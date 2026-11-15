"""Class-A decode attention on the round-5 tree, us per layer (KV ring from DRAM in one CUDA graph,
bench_attention.timeit), workspace and grids sized for PI_DEC_CAP keys per request (8192, the engine's --max-seq-len):
  fi / fi_tc  FlashInfer's decode (CUDA cores / tensor cores)
  v1          decode_kernel (the only class-A decode kernel before round 5)
  v0          the planner (decode_plan), told the longest request as the engine tells it: decode16_kernel (plan v3)
              or decode_kernel (plan v1)
Interleaved over ROUNDS rounds, medians. Run from kernels/. usage: decode_plan_bench.py [rounds] [hq] [hkv]"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import torch, flashinfer
import bench_attention as B
ext, dev, D, PAGE = B.ext, "cuda", 128, 16
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
HQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CAP = int(os.environ.get("PI_DEC_CAP", "8192"))
clock = lambda: subprocess.run("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader".split(), capture_output=True, text=True).stdout.strip()


def bench(batch, kv):
    kvs, indptr, indices, last, lens = B.make_cache(batch, kv, HKV)
    q = torch.randn(batch, HQ, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    fis = {}
    for name, tc in (("fi", False), ("fi_tc", True)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        w.plan(indptr.cpu(), indices, last.cpu(), HQ, HKV, D, PAGE, q_data_type=torch.bfloat16)
        fis[name] = w
    max_kv = max(kv, CAP)
    rows = batch * -(-max_kv // ext.GROUP_KEYS)
    ws_o = torch.full((rows, HQ, D), float("nan"), device=dev)
    ws_ml = torch.full((rows, HQ, 2 * ext.DECODE_WARPS), float("nan"), device=dev)
    cnt = torch.zeros(rows, HQ, ext.DECODE_WARPS, dtype=torch.int32, device=dev)

    def pi(var):
        ext.set_decode_variant(var)
        t = B.timeit(lambda i: ext.decode(q, *kvs[i], indptr, indices, lens, ws_o, ws_ml, cnt, max_kv, 1 / math.sqrt(D), kv), len(kvs))
        ext.set_decode_variant(0)
        return t
    r = {}
    for _ in range(ROUNDS):
        for name, w in fis.items():
            r.setdefault(name, []).append(B.timeit(lambda i: w.run(q, kvs[i]), len(kvs)))
        r.setdefault("v1", []).append(pi(1))
        r.setdefault("v0", []).append(pi(0))
    m = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
    fi = min(m["fi"], m["fi_tc"])
    print(f"DEC hq={HQ} hkv={HKV} b={batch:3d} kv={kv:5d} us/layer: " + " ".join(f"{k} {v:6.1f}" for k, v in m.items())
          + f" | plan v{ext.decode_plan(batch, HKV, max_kv, kv, HQ)} | v1/fi {m['v1'] / fi:.2f} v0/fi {m['v0'] / fi:.2f} v0/v1 {m['v0'] / m['v1']:.3f} | {clock()}", flush=True)


B.warm_gpu()
for batch in (1, 2, 4, 8, 13, 16, 32, 64):
    for kv in (96, 192, 320, 512, 640, 768, 1024, 2176, 4200, 8000):
        bench(batch, kv)
