"""Segment count as a canonical choice: best engine time per (shape, S, M) against cuBLAS, single GPU.
Cold L2 (a ring of weight copies) for M <= 512 for both, as the engine's tuner times decode.
usage (from kernels/): python ../logs/narrow/scan_s.py"""
import subprocess, sys
import torch
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit

SHAPES = {  # single-GPU projections whose decode / mid-M is off cuBLAS, and the 70B row-parallel shapes
    '0.6B o': (1024, 2048), '0.6B down': (1024, 3072), '1.7B o': (2048, 2048), '1.7B down': (2048, 6144),
    '4B o': (2560, 4096), '4B down': (2560, 9728), '8B o': (4096, 4096), '8B down': (4096, 12288),
    '8B qkv': (6144, 4096), '3.5-4B down': (2560, 9216), '70B o': (8192, 8192), '70B down': (8192, 28672)}
MS = [1, 8, 32, 128, 256, 512, 1024, 4096, 16384]
print('clock:', subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.applications.graphics', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip(), flush=True)
torch.manual_seed(0)
for name, (N, K) in SHAPES.items():
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ring = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    S_list = [S for S in (1, 2, 4, 8) if K % (S * 256) == 0 and K // S >= 1024]
    for M in MS:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ws = ring if M <= 512 else [W]
        tc = timeit(lambda w: torch.matmul(X, w.T), ws)
        line = f'{name:12s} M={M:5d} cuBLAS {tc:8.1f}us:'
        for S in S_list:
            t, k = min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
            line += f' S={S} {t / tc:.2f}(k{k})'
        print(line, flush=True)
    del W, ring
    torch.cuda.empty_cache()
