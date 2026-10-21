"""Decode-shaped GEMM time in a burst of back-to-back launches, each over another
weight copy (so every launch streams from DRAM and launch gaps overlap as in a
forward): invariant kernels (each id) vs cuBLAS. Usage: python gemm_burst.py [M ...]."""
import sys

import torch

import test_gemm as G
from gemm_cold import KERNELS, SHAPES, usable

POOL = 600 << 20  # > 8x L2: every launch streams its weights from DRAM


def burst(fn, Ws, X, n=64, reps=5):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n):
            fn(X, Ws[i % len(Ws)])
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]


for M in [int(a) for a in sys.argv[1:]] or (1,):
    for name, (N, K) in SHAPES.items():
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda x, w: torch.matmul(x, w.T), Ws, X)
        res = {k: burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X) for k in KERNELS if usable(k, M, N, K)}
        top = sorted(res, key=res.get)[:4]
        print(f"M={M} {name:12s} cublas {cb:6.2f}us | " + " ".join(f"k{k} {res[k]:6.2f}" for k in top)
              + f" | ratio {res[top[0]] / cb:.2f}", flush=True)
        del Ws; torch.cuda.empty_cache()
