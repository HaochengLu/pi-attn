"""Probe 17: is a small kernel slower when it follows a PI GEMM than when it follows cuBLAS?
Graph of n x (gemm, fused_add_rmsnorm) minus graph of n x gemm, per pair."""
import sys
sys.path.insert(0, "kernels")
import torch
import test_gemm as G
from gemm_burst import burst
from flashinfer.norm import fused_add_rmsnorm

POOL = 600 << 20
for M in (1,):
    for name, (N, K) in G.SHAPES.items():
        if name.endswith("lm_head"):
            continue
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        X = torch.randn(M, K, device="cuda").bfloat16()
        h = torch.randn(M, N, device="cuda").bfloat16(); r = torch.randn(M, N, device="cuda").bfloat16()
        wn = torch.ones(N, device="cuda").bfloat16()
        def pair(f):
            def g(x, w):
                y = f(x, w)
                fused_add_rmsnorm(h, r, wn, 1e-6)
                return y
            return g
        cb = lambda x, w: torch.matmul(x, w.T)
        ks = G.candidates(M, N, K)
        out = [f"cublas {burst(pair(cb), Ws, X) - burst(cb, Ws, X):5.2f}"]
        for k in ks:
            f = lambda x, w, k=k: G.ext.linear(x, w, k, False)
            out.append(f"k{k} {burst(pair(f), Ws, X) - burst(f, Ws, X):5.2f}")
        print(f"{name:12s} norm cost after: " + " ".join(out), flush=True)
        del Ws; torch.cuda.empty_cache()
