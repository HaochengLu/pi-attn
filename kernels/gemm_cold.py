"""Decode-shaped GEMM time with a cold L2 (weights streamed from DRAM, as in a
real decode step): invariant kernels (each id) vs cuBLAS, Qwen3-0.6B/1.7B shapes."""
import torch

import test_gemm as G

flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def cold(fn, iters=20):
    ts = []
    for _ in range(iters + 3):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts = sorted(ts[3:])
    return ts[len(ts) // 2]


SHAPES = {"0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
          "0.6B lm_head": (151936, 1024), "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144)}
for M in (1, 8, 32):
    for name, (N, K) in SHAPES.items():
        W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = cold(lambda: torch.matmul(X, W.T))
        res = {k: cold(lambda: G.ext.linear(X, W, k, False)) for k in (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13)}
        best = min(res, key=res.get)
        mb = N * K * 2 / 1e6
        print(f"M={M:2d} {name:13s} {mb:6.1f}MB cublas {cb:7.1f}us ({mb / cb * 1e3:5.0f} GB/s) | best pi k{best} {res[best]:7.1f}us "
              f"({mb / res[best] * 1e3:5.0f} GB/s) ratio {res[best] / cb:.2f} | k1 {res[1]:.1f} k6 {res[6]:.1f} k13 {res[13]:.1f}", flush=True)
