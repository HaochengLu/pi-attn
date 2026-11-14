"""Decode attention on Hopper, us per layer (KV ring from DRAM in one CUDA graph, bench_attention.timeit):
FlashInfer's decode (CUDA-core and tensor-core), decode_kernel (v1), decode90_kernel wherever it can run (v2) and
the planner's choice (v0). The archived diffs of the earlier designs carry their own variant ids.
usage (from kernels/): dec90_bench.py [eq] [quick|full|grid2] ; PI_DEC_VARIANTS=1,2,0 ; PI_DEC_SLACK=n (workspace
and grid sized for n times the keys, as the engine's graphs are: --max-seq-len against the running lengths)"""
import math, os, sys
sys.path.insert(0, os.getcwd())
import torch, flashinfer
import bench_attention as B
ext, dev, D, PAGE = B.ext, "cuda", 128, 16
VARS = [int(v) for v in os.environ.get("PI_DEC_VARIANTS", "1,2").split(",")]
SLACK = int(os.environ.get("PI_DEC_SLACK", "1"))
HQ, HKV = 16, 8


def pi_args(batch, kv_len, hq, max_kv):
    rows = batch * -(-max_kv // ext.GROUP_KEYS)
    return (torch.full((rows, hq, D), float("nan"), device=dev),
            torch.full((rows, hq, 2 * ext.DECODE_WARPS), float("nan"), device=dev),
            torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device=dev))


def equal():
    """decode90_kernel against decode_kernel, bitwise: mixed lengths in one batch, workspaces sized for longer
    requests, requests of more fold groups than a CTA column has CTAs."""
    torch.manual_seed(0)
    bad = n = 0
    for hq, hkv in ((16, 8), (32, 8), (8, 8), (16, 1)):
        for lens in ([1], [31], [32], [33], [500], [512], [513], [1023], [2049], [4200], [1, 17, 64, 511, 512, 513, 700, 1300, 2049],
                     list(range(1, 130, 7)), [3000] * 3 + [5] * 5, [9000, 40, 5000], [70000 if hkv == 1 else 20000]):
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
                for var in (1, 2):
                    ext.set_decode_variant(var)
                    ws_o, ws_ml, cnt = pi_args(batch, mx, hq, max_kv)
                    outs[var] = ext.decode(q, k, v, indptr, perm, kl, ws_o, ws_ml, cnt, max_kv, 1 / math.sqrt(D))
                    assert cnt.count_nonzero().item() == 0
                for var in (2,):
                    n += 1
                    if not torch.equal(outs[var].view(torch.int16), outs[1].view(torch.int16)):
                        bad += 1
                        print("MISMATCH", hq, hkv, lens[:4], max_kv, var,
                              (outs[var].view(torch.int16) != outs[1].view(torch.int16)).sum().item(), flush=True)
    ext.set_decode_variant(0)
    print(f"EQ decode90_kernel vs decode_kernel: {n} comparisons, {bad} mismatches", flush=True)


def bench(batch, kv):
    kvs, indptr, indices, last, lens = B.make_cache(batch, kv, HKV)
    q = torch.randn(batch, HQ, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    r = {}
    for name, tc in (("fi", False), ("fi_tc", True)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        w.plan(indptr.cpu(), indices, last.cpu(), HQ, HKV, D, PAGE, q_data_type=torch.bfloat16)
        r[name] = B.timeit(lambda i: w.run(q, kvs[i]), len(kvs))
    max_kv = kv * SLACK
    ws_o, ws_ml, cnt = pi_args(batch, kv, HQ, max_kv)
    for var in VARS:
        ext.set_decode_variant(var)
        r[f"v{var}"] = B.timeit(lambda i: ext.decode(q, *kvs[i], indptr, indices, lens, ws_o, ws_ml, cnt, max_kv, 1 / math.sqrt(D)),
                                len(kvs))
    ext.set_decode_variant(0)
    best = min(VARS, key=lambda v: r[f"v{v}"])
    print(f"DEC b={batch:2d} kv={kv:4d} us/layer: " + " ".join(f"{k} {v:6.1f}" for k, v in r.items())
          + f" | best v{best} / fi {r[f'v{best}'] / r['fi']:.2f}", flush=True)


for _ in range(sys.argv.count("eq")):
    equal()
B.warm_gpu()
if "quick" in sys.argv:
    for batch in (1, 8, 32):
        for kv in (96, 640, 2176):
            bench(batch, kv)
if "full" in sys.argv:
    for batch in (1, 4, 8, 16, 32, 64):
        for kv in (96, 320, 640, 1280, 2176, 4200):
            bench(batch, kv)
if "grid2" in sys.argv:  # the batches around the planner's rule
    for batch in (1, 2, 3, 4):
        for kv in (96, 192, 320, 640, 2176, 4200, 9000):
            bench(batch, kv)
