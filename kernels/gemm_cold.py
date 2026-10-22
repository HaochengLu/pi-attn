"""Decode-shaped GEMM time with a cold L2 (weights streamed from DRAM, as in a
real decode step): invariant kernels (each id) vs cuBLAS, Qwen3 0.6B/1.7B/8B shapes.
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


SHAPES = {"0.6B qkv": (4096, 1024), "0.6B o": (1024, 2048), "0.6B gateup": (6144, 1024), "0.6B down": (1024, 3072),
          "0.6B lm_head": (151936, 1024), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048),
          "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144), "1.7B lm_head": (151936, 2048),
          "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B gateup": (19456, 2560), "4B down": (2560, 9728),
          "4B lm_head": (151936, 2560),
          "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
          "8B lm_head": (151936, 4096)}
KERNELS = (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49, 50)
BK = {20: 128, 21: 256, 22: 256, 23: 256, 24: 128, 25: 128, 30: 256, 31: 128, 32: 128, 33: 128,
      40: 128, 41: 128, 42: 128, 43: 256}
MAX_M = {30: 16, 31: 16, 32: 32, 33: 64, 40: 8, 41: 8, 42: 8, 43: 8}


def usable(k, M, N, K):  # the kernel ids that run this shape (stream-K: S = 1 only; segment-parallel: S > 1 only)
    if k in (48, 49):
        return G.segpar_fits(k, M, N, K)
    return K % BK.get(k, 64) == 0 and M <= MAX_M.get(k, 1 << 30) and not (k >= 40 and G.gemm_segments(N, K) > 1)


def main():
    for M in [int(a) for a in sys.argv[1:]] or (1, 8, 32):
        for name, (N, K) in SHAPES.items():
            W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
            X = torch.randn(M, K, device="cuda").bfloat16()
            cb = cold(lambda: torch.matmul(X, W.T))
            res = {k: cold(lambda: G.ext.linear(X, W, k, False)) for k in KERNELS if usable(k, M, N, K)}
            best = min(res, key=res.get)
            mb = N * K * 2 / 1e6
            print(f"M={M:2d} {name:13s} {mb:6.1f}MB cublas {cb:7.1f}us ({mb / cb * 1e3:5.0f} GB/s) | best pi k{best} {res[best]:7.1f}us "
                  f"({mb / res[best] * 1e3:5.0f} GB/s) ratio {res[best] / cb:.2f} | bal {min((res[k], k) for k in res if 30 <= k <= 33)[1] if any(30 <= k <= 33 for k in res) else 0}:{min((res[k] for k in res if 30 <= k <= 33), default=0):.1f} k13 {res.get(13, 0):.1f}", flush=True)


if __name__ == "__main__":
    main()
