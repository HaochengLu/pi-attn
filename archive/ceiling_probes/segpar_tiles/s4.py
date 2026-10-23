"""Probe 18b: 0.6B o/down with S = 4 segments (prototype rule), graph-timed vs cuBLAS."""
import sys
sys.path.insert(0, "kernels")
import torch
import test_gemm as G
from gemm_burst import burst

POOL = 600 << 20
for M in [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048]:
    for name in ("0.6B o", "0.6B down"):
        N, K = G.SHAPES[name]
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        res = {k: burst(lambda x, w, k=k: G.ext.linear(x, w, k, False), Ws, X) for k in G.candidates(M, N, K)}
        top = sorted(res, key=res.get)[:4]
        print(f"M={M:4d} {name:10s} cublas {cb:7.2f} | " + " ".join(f"k{k} {res[k]:7.2f}" for k in top)
              + f" | ratio {res[top[0]] / cb:.2f}", flush=True)
        del Ws; torch.cuda.empty_cache()
