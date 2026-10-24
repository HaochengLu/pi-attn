"""Planner (s0) vs forced splits on short extends over long prefixes (the wave branch), variant 2."""
import math

import flashinfer
import torch

from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu

warm_gpu()
for hq, hkv in [(16, 8), (32, 8)]:
    for b, q_len, prefix in [(1, 128, 4096), (1, 128, 16384), (1, 512, 8192), (1, 256, 30000), (2, 512, 4096), (4, 128, 8192)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(b, kv_len, hkv)
        q = torch.randn(b * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, b + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)), iters=30)
        r = {f"s{sp}": timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), sp, 2), iters=30) for sp in (0, 1, 2, 4, 8, 16, 32, 64)}
        print(f"hq={hq} b={b} q={q_len} prefix={prefix}: fi {fi:7.1f} | " + " | ".join(f"{c} {t:7.1f} ({t/fi:.2f})" for c, t in r.items()), flush=True)
