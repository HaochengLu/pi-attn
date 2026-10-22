"""Decode-shaped GEMM time with a cold L2 (weights streamed from DRAM, as in a
real decode step): the engine's candidate kernel ids vs cuBLAS, every Qwen3 projection.
Usage: python gemm_cold.py [M ...] (default 1 8 32)."""
import sys

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


SHAPES = G.SHAPES


def main():
    for M in [int(a) for a in sys.argv[1:]] or (1, 8, 32):
        for name, (N, K) in SHAPES.items():
            W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
            X = torch.randn(M, K, device="cuda").bfloat16()
            cb = cold(lambda: torch.matmul(X, W.T))
            res = {k: cold(lambda: G.ext.linear(X, W, k, False)) for k in G.candidates(M, N, K)}
            best = min(res, key=res.get)
            mb = N * K * 2 / 1e6
            print(f"M={M:2d} {name:13s} {mb:6.1f}MB cublas {cb:7.1f}us ({mb / cb * 1e3:5.0f} GB/s) | best pi k{best} {res[best]:7.1f}us "
                  f"({mb / res[best] * 1e3:5.0f} GB/s) ratio {res[best] / cb:.2f} | bal {min((res[k], k) for k in res if 30 <= k <= 33)[1] if any(30 <= k <= 33 for k in res) else 0}:{min((res[k] for k in res if 30 <= k <= 33), default=0):.1f} k13 {res.get(13, 0):.1f}", flush=True)


if __name__ == "__main__":
    main()
