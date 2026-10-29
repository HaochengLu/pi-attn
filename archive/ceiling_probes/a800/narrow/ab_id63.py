"""Gain of stream-K id 63 (16 X rows) at M = 9..16: best engine id with vs without it, same session.
usage (from kernels/): python ../logs/narrow/ab_id63.py"""
import subprocess, sys, torch
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit
SHAPES = [('70B down t8', 1024, 28672, 2), ('70B o t8', 1024, 8192, 2), ('70B down t4', 2048, 28672, 2),
          ('8B down t8', 512, 12288, 4), ('8B o t8', 512, 4096, 4), ('4B down t8', 320, 9728, 2),
          ('0.6B o', 1024, 2048, 2), ('0.6B down', 1024, 3072, 2), ('3.5-4B ba', 64, 2560, 2)]
torch.manual_seed(0)
for name, N, K, S in SHAPES:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    line = f'{name:12s} N={N} K={K} S={S}:'
    for M in [9, 12, 16]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        t = {k: timeit(lambda w: G.ext.linear(X, w, k, False, S), ws) for k in G.candidates(M, N, K, S)}
        without = min((v, k) for k, v in t.items() if k != 63)
        line += f' M={M} {without[0]:.1f}(k{without[1]}) -> k63 {t.get(63, float("nan")):.1f} ({t.get(63, float("nan")) / without[0]:.2f})'
    print(line, flush=True)
print('SM clock under load:', subprocess.run(['nvidia-smi', '--query-gpu=clocks.applications.graphics,clocks.max.sm', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())
