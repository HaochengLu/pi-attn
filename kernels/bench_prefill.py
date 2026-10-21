"""Prefill latency of PI-Attn at forced split counts and the planner (0) vs FlashInfer."""
import math

import flashinfer
import torch

from bench_attention import PAGE, D, dev, ext, make_cache, timeit

SPLITS = (0, 1, 2, 4, 8)
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in [(1, 512, 0), (1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (32, 512, 0), (1, 512, 8192), (1, 128, 4096), (1, 128, 16384), (4, 2048, 2048)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)), iters=20)
        r = {}
        for sp in SPLITS:
            r[sp] = timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), sp), iters=20)
        print(f"hq={hq} b={batch} q={q_len} prefix={prefix}: fi {fi:7.1f} | " + " | ".join(f"s{c} {t:7.1f} ({t/fi:.2f})" for c, t in r.items()), flush=True)
