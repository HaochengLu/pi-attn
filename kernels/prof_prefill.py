"""One PI prefill (8 warps) + one FlashInfer prefill call, for ncu."""
import math

import flashinfer
import torch

from bench_attn import PAGE, D, dev, ext, make_cache

hq, hkv, q_len = 16, 8, 8192
k, v, indptr, indices, last, lens = make_cache(1, q_len, hkv)
q = torch.randn(q_len, hq, D, device=dev, dtype=torch.bfloat16)
qo = torch.tensor([0, q_len], device=dev, dtype=torch.int32)
w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
for _ in range(2):
    ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, 1 / math.sqrt(D), 8)
    w.run(q, (k, v))
torch.cuda.synchronize()
