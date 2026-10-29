"""Narrow-N probe: every config of narrow_probe.cu against the best engine id, bits checked.
usage (from kernels/): python ../logs/narrow/probe_narrow.py"""
import os, sys
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
KDIR = os.path.abspath('.')
ext = load(name='narrow_probe', sources=[os.path.join(HERE, 'narrow_probe.cu')], build_directory=HERE,
           extra_include_paths=[KDIR, os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
CFG = ext.configs()
SHAPES = [  # (name, N, K, S): column shards at t_max = 8, and single-GPU S = 2 shapes of small models
    ('70B down t8', 1024, 28672, 2), ('70B o t8', 1024, 8192, 2), ('8B down t8', 512, 12288, 4),
    ('8B o t8', 512, 4096, 4), ('4B down t8', 320, 9728, 2),
    ('0.6B o', 1024, 2048, 2), ('0.6B down', 1024, 3072, 2), ('1.7B down', 2048, 6144, 1)]
torch.manual_seed(0)
for name, N, K, S in SHAPES:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    for M in [1, 8, 16, 32, 64, 128]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ref = G.ext.linear(X, W, 1, False, S)
        eng = min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
        res, bad = [], []
        for i, c in enumerate(CFG):
            y = ext.narrow(X, W, i, S)
            if y is None:
                continue
            if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                bad.append(c)
                continue
            res.append((timeit(lambda w: ext.narrow(X, w, i, S), ws), c))
        res.sort()
        top = ' '.join(f'[{c}]{t:.1f}' for t, c in res[:3])
        print(f'{name:12s} M={M:3d} engine k{eng[1]} {eng[0]:.1f}us | narrow best {res[0][0]:.1f}us '
              f'({res[0][0] / eng[0]:.2f}x) {top}{" BITS-DIFFER " + str(bad) if bad else ""}', flush=True)
