"""Segment-chain ids 58/59 vs the best other candidates and cuBLAS, graph bursts (gemm_burst.py)."""
import sys

import torch

import test_gemm as G
from gemm_burst import burst, POOL

for M in (1, 4, 8):
    for name in ("0.6B o", "0.6B down", "0.6B qkv", "1.7B o", "1.7B down", "4B o", "8B o"):
        N, K = G.SHAPES[name]
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, POOL // (N * K * 2)))]
        X = torch.randn(M, K, device="cuda").bfloat16()
        ref = G.ext.linear(X, Ws[0], 13, False)
        same = {k: (lambda y: y is not None and torch.equal(y.view(torch.int16), ref.view(torch.int16)))(G.ext.linear(X, Ws[0], k, False)) for k in (58, 59)}
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        res = {k: burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X) for k in G.candidates(M, N, K)}
        top = sorted((k for k in res if k not in (58, 59)), key=res.get)[:3]
        print(f"M={M} {name:10s} cublas {cb:6.2f} | " + " ".join(f"k{k} {res[k]:6.2f}" for k in top)
              + " | " + " ".join(f"k{k} {res.get(k, float('nan')):6.2f} same={same[k]}" for k in (58, 59)), flush=True)
        del Ws; torch.cuda.empty_cache()
