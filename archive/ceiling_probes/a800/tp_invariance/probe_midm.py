"""Per-id times of narrow column shards at decode M (cold L2), to see which kernels limit M = 8..128.
usage (from kernels/): python ../logs/tp/probe_midm.py"""
import sys, torch
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit
torch.manual_seed(0)
for name, N, K, S in [('70B down t8', 1024, 28672, 2), ('70B o t8', 1024, 8192, 2), ('8B down t8', 512, 12288, 4)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    bw = N * K * 2 / 1.6e6  # us at 1.6 TB/s
    for M in [8, 16, 32, 64, 128]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        r = sorted((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
        print(f'{name} M={M} bw-floor {bw:.1f}us chain-floor {4 + 0.028 * K / S / 16:.1f}us: ' + ' '.join(f'k{k}:{t:.1f}' for t, k in r), flush=True)
