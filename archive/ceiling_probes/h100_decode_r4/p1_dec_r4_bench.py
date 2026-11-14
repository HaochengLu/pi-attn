"""Decode attention on Hopper, us per layer (KV ring from DRAM in one CUDA graph, bench_attention.timeit):
FlashInfer's decode (CUDA-core and tensor-core), decode_kernel (v1), decode90_kernel (v2), decode90c_kernel (v3;
v4 = one arrival per warp) wherever they can run, and the planner (v0), which is told the longest request as
the engine tells it. The workspace and the grids are sized for PI_DEC_CAP keys per request (default 8192, the
engine's --max-seq-len), as the engine's graphs are.
usage (from kernels/): dec_r4_bench.py [eq] [grid] [table] ; PI_DEC_VARIANTS=1,2,3,0 ; PI_DEC_CAP=n"""
import math, os, sys
sys.path.insert(0, os.getcwd())
import torch, flashinfer
import bench_attention as B
ext, dev, D, PAGE = B.ext, "cuda", 128, 16
VARS = [int(v) for v in os.environ.get("PI_DEC_VARIANTS", "1,2,3,0").split(",")]
CAP = int(os.environ.get("PI_DEC_CAP", "8192"))
EQ_VARS = [int(v) for v in os.environ.get("PI_EQ_VARIANTS", "2,3,0").split(",")]
HQ, HKV = 16, 8


def pi_args(batch, hq, max_kv):
    rows = batch * -(-max_kv // ext.GROUP_KEYS)
    return (torch.full((rows, hq, D), float("nan"), device=dev),
            torch.full((rows, hq, 2 * ext.DECODE_WARPS), float("nan"), device=dev),
            torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device=dev))


def equal():
    """Every decode kernel and the planner against decode_kernel, bitwise: mixed lengths in one batch, workspaces
    sized for longer requests, requests of more fold groups than a CTA column has CTAs, each launch twice (the
    second on the shared memory the first left)."""
    torch.manual_seed(0)
    bad = {v: 0 for v in EQ_VARS}
    n = 0
    for hq, hkv in ((16, 8), (32, 8), (8, 8), (16, 1)):
        for lens in ([1], [31], [32], [33], [500], [512], [513], [1023], [2049], [4200], [1, 17, 64, 511, 512, 513, 700, 1300, 2049],
                     list(range(1, 130, 7)), [3000] * 3 + [5] * 5, [9000, 40, 5000], [70000 if hkv == 1 else 20000],
                     [600, 300], [1500, 1024, 200, 513], [520] * 8):
            batch, mx = len(lens), max(lens)
            pp = -(-mx // PAGE)
            npg = batch * pp + 1
            k = torch.randn(npg, PAGE, hkv, D, device=dev).bfloat16()
            v = torch.randn(npg, PAGE, hkv, D, device=dev).bfloat16()
            q = (torch.randn(batch, hq, D, device=dev) * 3).bfloat16()
            perm = (torch.randperm(npg - 1, device=dev).int() + 1).contiguous()
            indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pp
            kl = torch.tensor(lens, device=dev, dtype=torch.int32)
            for max_kv in (mx, mx + 1500, mx + 40000):
                outs = {}
                for var in [1] + EQ_VARS:
                    ext.set_decode_variant(var)
                    for rep in range(2):
                        ws_o, ws_ml, cnt = pi_args(batch, hq, max_kv)
                        out = ext.decode(q, k, v, indptr, perm, kl, ws_o, ws_ml, cnt, max_kv, 1 / math.sqrt(D), mx)
                        assert cnt.count_nonzero().item() == 0
                        if var == 1 and rep == 0:
                            outs[1] = out
                        elif not torch.equal(out.view(torch.int16), outs[1].view(torch.int16)):
                            bad[var] += 1
                            print("MISMATCH", hq, hkv, lens[:4], max_kv, var, rep,
                                  (out.view(torch.int16) != outs[1].view(torch.int16)).sum().item(), flush=True)
                n += 2
    ext.set_decode_variant(0)
    for var in EQ_VARS:
        print(f"EQ decode variant {var} vs decode_kernel: {n} comparisons, {bad[var]} mismatches", flush=True)


def bench(batch, kv, tag="DEC"):
    kvs, indptr, indices, last, lens = B.make_cache(batch, kv, HKV)
    q = torch.randn(batch, HQ, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    r = {}
    for name, tc in (("fi", False), ("fi_tc", True)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        w.plan(indptr.cpu(), indices, last.cpu(), HQ, HKV, D, PAGE, q_data_type=torch.bfloat16)
        r[name] = B.timeit(lambda i: w.run(q, kvs[i]), len(kvs))
    max_kv = max(kv, CAP)
    ws_o, ws_ml, cnt = pi_args(batch, HQ, max_kv)
    for var in VARS:
        ext.set_decode_variant(var)
        r[f"v{var}"] = B.timeit(lambda i: ext.decode(q, *kvs[i], indptr, indices, lens, ws_o, ws_ml, cnt, max_kv,
                                                     1 / math.sqrt(D), kv), len(kvs))
    ext.set_decode_variant(0)
    plan = ext.decode_plan(batch, HKV, max_kv, kv)
    fi = min(r["fi"], r["fi_tc"])
    best = min((v for v in VARS if v), key=lambda v: r[f"v{v}"])
    print(f"{tag} b={batch:2d} kv={kv:4d} us/layer: " + " ".join(f"{k} {v:6.1f}" for k, v in r.items())
          + f" | plan v{plan} best v{best}" + (f" | v0 / fi {r['v0'] / fi:.2f}" if "v0" in r else ""), flush=True)


for _ in range(sys.argv.count("eq")):
    equal()
B.warm_gpu()
if "grid" in sys.argv:  # where each kernel wins: the planner's rule
    for batch in (1, 2, 4, 8, 16):
        for kv in (96, 128, 160, 192, 320, 512, 640, 1024, 2176, 4096, 4200, 8000):
            bench(batch, kv)
if "table" in sys.argv:
    for batch in (1, 2, 4, 8, 16, 32):
        for kv in (96, 320, 640, 1280, 2176, 4200):
            bench(batch, kv, "TABLE")
