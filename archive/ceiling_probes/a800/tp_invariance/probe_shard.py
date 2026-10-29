"""Per-rank GEMM time of a row-parallel layer under two shardings, from the S=1 build of probe_segs.py:
row (standard Megatron: rank holds W[:, K/t], partial sums all-reduced) vs col (rank holds W[N/t, :] with the full
K: every output element keeps the TP=1 chain, inputs all-gathered). Best tuned id per shape, cold L2 for M <= 512.
Also: S=1 time of the 128x128 tile (id 11) against the best id at large M.
usage (from logs/tpprobe): python probe_shard.py"""
import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
import flashinfer
major, minor = torch.cuda.get_device_capability()
dst = os.path.join(HERE, 'S1')
ext = load(name='pi_gemm_probe_S1', sources=[os.path.join(dst, 'gemm_torch.cu')],
           extra_include_paths=[dst, os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
           build_directory=dst)
SMALL = set(ext.small_kernels())
IDS = list(ext.small_kernels()) + list(ext.large_kernels())


def timeit(fn, ws, iters=20):
    for i in range(3):
        fn(ws[i % len(ws)])
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(iters):
        fn(ws[i % len(ws)])
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


def best(x, W, ids=None):
    cold = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)] if x.shape[0] <= 512 else [W]
    r = None
    for k in ids or IDS:
        if x.shape[0] > 64 and k in SMALL:
            continue
        if ext.linear(x, W, k, False) is None:
            continue
        t = timeit(lambda w: ext.linear(x, w, k, False), cold)
        r = (t, k) if r is None or t < r[0] else r
    return r


SHAPES = [('qwen3.5-4B o/gdn-out', 2560, 4096), ('qwen3.5-4B down', 2560, 9216), ('qwen3-8B o', 4096, 4096),
          ('qwen3-8B down', 4096, 12288), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]
torch.manual_seed(0)
for name, N, K in SHAPES:
    for M in [2048, 8192]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        W = torch.randn(N, K, device='cuda').bfloat16() * 0.02
        b = best(X, W)
        t11 = best(X, W, [11])
        print(f'TILE {name:22s} M={M:5d}: best {b[0]:.0f}us(k{b[1]}) k11 {t11[0]:.0f}us = {t11[0] / b[0]:.2f}x', flush=True)
    for t in [2, 4, 8]:
        if (K // t) % 256 or (N // t) % 64:
            continue
        line = f'SHARD {name:22s} N={N:5d} K={K:5d} t={t}:'
        Wr = torch.randn(N, K // t, device='cuda').bfloat16() * 0.02
        Wc = torch.randn(N // t, K, device='cuda').bfloat16() * 0.02
        for M in [1, 8, 32, 128, 512, 2048, 8192]:
            Xr = torch.randn(M, K // t, device='cuda').bfloat16()
            Xc = torch.randn(M, K, device='cuda').bfloat16()
            r, c = best(Xr, Wr), best(Xc, Wc)
            line += f' M={M}: row {r[0]:.1f} col {c[0]:.1f} ({c[0] / r[0]:.2f}x)'
        print(line, flush=True)
    torch.cuda.empty_cache()
