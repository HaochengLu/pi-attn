"""0.6B segmented shapes (o, down), burst timing over a 600 MB weight ring: cuBLAS vs the
best previous id vs the segment-parallel ids (48, 49; a 64x128 id 50 was tried and never won)."""
import sys
import torch
import test_gemm as G
from gemm_cold import KERNELS, usable
from gemm_burst import burst, POOL

for M in (16, 32, 64, 128, 256, 512, 1024, 2048):
    for name, (N, K) in {"0.6B o": (1024, 2048), "0.6B down": (1024, 3072)}.items():
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, POOL // (N * K * 2)))]
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        res = {k: burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X) for k in KERNELS if usable(k, M, N, K)}
        old = min((k for k in res if k < 48), key=res.get)
        new = {k: res[k] for k in res if k >= 48}
        print(f"M={M:4d} {name:9s} cublas {cb:6.2f} | old k{old} {res[old]:6.2f} ({res[old]/cb:.2f}) | "
              + " ".join(f"k{k} {t:6.2f}" for k, t in new.items())
              + (f" ({min(new.values())/cb:.2f})" if new else ""), flush=True)
