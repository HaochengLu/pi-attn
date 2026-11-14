"""H100 prefill, us per call (KV ring in one CUDA graph): FlashInfer, the mma.sync kernel (variant 1) and the
wgmma kernel with the group totals in shared memory / in registers (variants 5 = the device's, 6 = the other)."""
import math, os, sys
sys.path.insert(0, os.getcwd())
import flashinfer, torch
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
warm_gpu()
VARS = [int(v) for v in os.environ.get("PI_PRE_VARIANTS", "1,5,6").split(",")]
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in [(1, 512, 0), (1, 1024, 0), (1, 2048, 0), (1, 4096, 0), (8, 1024, 0), (8, 2048, 0), (1, 2048, 2048), (1, 128, 4096)]:
        kv_len = q_len + prefix
        kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda i: w.run(q, kv[i]), len(kv))
        t = {v: timeit(lambda i: ext.prefill(q, *kv[i], qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, v), len(kv)) for v in VARS}
        print(f"PRE hq={hq} b={batch} q={q_len} prefix={prefix}: fi {fi:7.1f} | " + " | ".join(f"v{v} {x:7.1f}" for v, x in t.items())
              + (f" | v6/v5 {t[6] / t[5]:.3f}" if 5 in t and 6 in t else ""), flush=True)
        del kv, w
        torch.cuda.empty_cache()
