"""Planner vs forced splits on the chunked-prefill shapes of long-context serving (q=2048
chunks over growing prefixes), variant 2."""
import math

import flashinfer
import torch

from bench_attention import PAGE, D, dev, ext, make_cache, timeit, warm_gpu

warm_gpu()
for hq, hkv in [(16, 8), (32, 8)]:
    for q_len, prefix in [(2048, 0), (2048, 2048), (2048, 6144), (2048, 14336), (2048, 26624), (1024, 0), (1024, 7168)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(1, kv_len, hkv)
        q = torch.randn(q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.tensor([0, q_len], device=dev, dtype=torch.int32)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)), iters=30)
        r = {f"s{sp}": timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), sp, 2), iters=30) for sp in (0, 1, 2, 4, 8)}
        print(f"hq={hq} q={q_len} prefix={prefix}: fi {fi:7.1f} | " + " | ".join(f"{c} {t:7.1f} ({t/fi:.2f})" for c, t in r.items()), flush=True)
