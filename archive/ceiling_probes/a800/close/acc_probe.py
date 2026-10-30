"""Accuracy vs float64 at L=4096 for PI, torch SDPA (each backend), FlashInfer single prefill."""
import math, sys
import torch
sys.path.insert(0, '.')
import test_invariance as T
import flashinfer
from torch.nn.attention import sdpa_kernel, SDPBackend
dev = 'cuda'
for hq, hkv, qs in [(16, 8, 1.0), (16, 8, 6.0)]:
    torch.manual_seed(1)
    T.HQ, T.HKV = hq, hkv
    L = 4096
    q = torch.randn(L, hq, 128, device=dev).bfloat16() * qs
    k = torch.randn(L, hkv, 128, device=dev).bfloat16()
    v = torch.randn(L, hkv, 128, device=dev).bfloat16()
    pool = T.Pool(pages=L // 16 + 8)
    pages = pool.alloc(L)
    pool.write(pages, 0, k, v)
    T.KVSPLITS, T.VARIANT = 1, 0
    pi = T.prefill(pool, [(q, pages, L)])[0].float()
    ex = T.exact(q, k, v).float()
    def err(o):
        e = (o.float() - ex).abs()
        return f"mean {e.mean():.3e} max {e.max():.3e}"
    print(f"hq={hq} qscale={qs} PI: {err(pi)}")
    for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel([be]):
                kk, vv = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
                o = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), kk.transpose(0, 1), vv.transpose(0, 1), is_causal=True).transpose(0, 1)
            print(f"  SDPA {be.name}: {err(o)}")
        except Exception as e:
            print(f"  SDPA {be.name}: {type(e).__name__}")
    o = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True, enable_gqa=True).transpose(0, 1)
    print(f"  SDPA default enable_gqa: {err(o)}")
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    print(f"  FlashInfer single_prefill: {err(o)}")
    # fp32 reference P path: exact softmax in fp32 then bf16 output
    print(f"  bf16(exact) rounding only: {err(ex.bfloat16())}")
