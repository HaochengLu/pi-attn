"""Class-A decode attention, us per layer (KV ring from DRAM in one CUDA graph, bench_attention.timeit), workspace
and grids sized for PI_DEC_CAP keys per request (8192, the engine's --max-seq-len):
  fi / fi_tc  FlashInfer's decode (CUDA cores / tensor cores)
  v1          decode_kernel (shipped)
  v3          decode16_kernel: a fold group per CTA on 16 warps, the keys in the mma's 16-row operand
  v4          the same without helper CTAs (shown where the launch takes more than one wave)
  v0          the planner (decode_plan), told the longest request as the engine tells it (PI_R5_PLAN=1)
Interleaved over ROUNDS rounds, medians. Run from kernels/. usage: r5_decode_bench.py [rounds] [hq] [hkv]"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import torch, flashinfer
import bench_attention as B
from build import build
ext, dev, D, PAGE = B.ext, "cuda", 128, 16
HALF = None
if os.environ.get("PI_R5_HALF", "1") == "1":
    os.environ["PI_ATTN_SUFFIX"] = "_half"; os.environ["PI_ATTN_CFLAGS"] = "-DPI_PROBE_HALF_MMA"
    HALF = build()
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
HQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
HKV = int(sys.argv[3]) if len(sys.argv) > 3 else 8
CAP = int(os.environ.get("PI_DEC_CAP", "8192"))
SMS = torch.cuda.get_device_properties(0).multi_processor_count
PLAN = os.environ.get("PI_R5_PLAN") == "1"  # v0: the planner (decode_plan), told the longest request
ONLY_PLAN = os.environ.get("PI_R5_ONLY_PLAN") == "1"  # v1 and v0 only
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
    def pi(e, var):
        e.set_decode_variant(var)
        t = B.timeit(lambda i: e.decode(q, *kvs[i], indptr, indices, lens, ws_o, ws_ml, cnt, max_kv, 1 / math.sqrt(D), kv), len(kvs))
        e.set_decode_variant(0)
        return t
    fits = batch * HKV <= SMS  # decode16_kernel can run (else variant 3 falls back to decode_kernel)
    onewave = batch * HKV * -(-kv // ext.GROUP_KEYS) <= SMS
    r = {}
    for _ in range(ROUNDS):
        for name, w in fis.items():
            r.setdefault(name, []).append(B.timeit(lambda i: w.run(q, kvs[i]), len(kvs)))
        r.setdefault("v1", []).append(pi(ext, 1))
        if PLAN: r.setdefault("v0", []).append(pi(ext, 0))
        if fits and not ONLY_PLAN: r.setdefault("v3", []).append(pi(ext, 3))
        if fits and not onewave and not ONLY_PLAN: r.setdefault("v4", []).append(pi(ext, 4))
        if HALF:
            r.setdefault("h1", []).append(pi(HALF, 1))
            if fits and onewave: r.setdefault("h3", []).append(pi(HALF, 3))
    m = {k: sorted(v)[len(v) // 2] for k, v in r.items()}
    fi = min(m["fi"], m["fi_tc"])
    print(f"DEC hq={HQ} hkv={HKV} b={batch:3d} kv={kv:5d} us/layer: " + " ".join(f"{k} {v:6.1f}" for k, v in m.items())
          + f" | v1/fi {m['v1']/fi:.2f}" + (f" v3/v1 {m['v3']/m['v1']:.3f}{'' if onewave else ' (>1 wave)'}" if "v3" in m else "")
          + (f" h1/v1 {m['h1']/m['v1']:.3f}" if "h1" in m else "") + (f" h3/v3 {m['h3']/m['v3']:.3f}" if "h3" in m else "") + (f" v4/v1 {m['v4']/m['v1']:.3f}" if "v4" in m else "")
          + (f" | plan v{ext.decode_plan(batch, HKV, max_kv, kv, HQ)} v0/v1 {m['v0']/m['v1']:.3f} v0/fi {m['v0']/fi:.2f}" if "v0" in m else "")
          + f" | {clock()}", flush=True)

B.warm_gpu()
GRID = os.environ.get("PI_R5_GRID")
if GRID:
    for item in GRID.split(";"):
        bs, kvs = item.split(":")
        for batch in map(int, bs.split(",")):
            for kv in map(int, kvs.split(",")):
                bench(batch, kv)
else:
    for batch in (1, 2, 4, 8, 13, 16, 32, 64):
        for kv in (96, 192, 320, 512, 640, 1024, 2176, 4200, 8000):
            bench(batch, kv)
