"""Prefill with 32-row warps (variants 3 pipelined / 4 not) vs the shipped sm_80 kernel (v2,
QREG), FlashInfer for scale; planner splits (s0) for all; outputs bit-compared to v2. us."""
import math, torch, flashinfer
from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu
warm_gpu()
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in [(1, 512, 0), (1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (32, 512, 0), (1, 512, 8192), (1, 128, 16384), (4, 2048, 2048)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)), iters=20)
        run = lambda var: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, var)
        ref = run(2)
        same = all(torch.equal(run(var).view(torch.int16), ref.view(torch.int16)) for var in (3, 4))
        r = {f"v{var}": timeit(lambda: run(var), iters=20) for var in (2, 3, 4)}
        print(f"hq={hq} b={batch} q={q_len} prefix={prefix}: fi {fi:7.1f} | " + " | ".join(f"{c} {t:7.1f} ({t/fi:.2f})" for c, t in r.items()) + f" | same={same}", flush=True)
