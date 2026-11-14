import math, os, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
warm_gpu()
VARS = [int(v) for v in os.environ.get("PI_PRE_VARIANTS", "5,6").split(",")]
SHAPES = [(1, 512, 0), (1, 1024, 0), (1, 2048, 0), (1, 4096, 0), (8, 1024, 0), (8, 2048, 0), (1, 2048, 2048), (1, 128, 4096)]
if os.environ.get("PI_PRE_SHORT"): SHAPES = [(1, 1024, 0), (1, 4096, 0), (8, 1024, 0), (1, 128, 4096)]
if os.environ.get("PI_PRE_ENGINE"): SHAPES = [(1, 64, 0), (8, 64, 0), (32, 64, 0), (2, 1024, 0), (4, 512, 0), (8, 256, 0), (16, 128, 0), (8, 64, 1024), (32, 32, 96), (2, 1000, 24)]
HEADS = [(16, 8)] if os.environ.get("PI_PRE_ENGINE") else [(16, 8), (32, 8)]
for hq, hkv in HEADS:
    for batch, q_len, prefix in SHAPES:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        fi = {}
        for be in ("fa2", "fa3"):
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD", backend=be)
            w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
            fi[be] = timeit(lambda i: w.run(q, kv[i]), len(kv))
            del w
        best = min(fi.values())
        print(f"PRE hq={hq} b={batch} q={q_len} prefix={prefix}: fa2 {fi['fa2']:.1f} fa3 {fi['fa3']:.1f}", end="", flush=True)
        for v in VARS:
            x = timeit(lambda i: ext.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, v), len(kv))
            print(f" | v{v} {x:.1f} ({x / best:.2f})", end="", flush=True)
        print(flush=True)
        del kv
        torch.cuda.empty_cache()
