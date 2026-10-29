"""Systolic relay probe: every (config, slice, column groups Q) of systolic_probe.cu against the best
engine id, bits checked against the engine. usage (from kernels/): python ../logs/narrow/probe_systolic.py"""
import os, sys
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='systolic_probe', sources=[os.path.join(HERE, 'systolic_probe.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
CFG = ext.configs()
SHAPES = [('70B down t8', 1024, 28672, 2), ('70B o t8', 1024, 8192, 2), ('8B down t8', 512, 12288, 4),
          ('4B down t8', 320, 9728, 2), ('0.6B down', 1024, 3072, 2), ('1.7B down', 2048, 6144, 1)]
torch.manual_seed(0)
for name, N, K, S in SHAPES:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    for M in [1, 8, 16, 32, 64]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ref = G.ext.linear(X, W, 1, False, S)
        eng = min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
        res, bad = [], []
        for i, c in enumerate(CFG):
            for sl in [256, 512, 768, 1024, 1536, 2048, 3072]:
                for Q in [1, 2, 4, 5, 8, 16]:
                    y = ext.systolic(X, W, i, S, sl, Q)
                    if y is None:
                        continue
                    if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                        bad.append((c, sl, Q))
                        continue
                    res.append((timeit(lambda w: ext.systolic(X, w, i, S, sl, Q), ws), f'{c}|sl{sl}|Q{Q}|P{K // S // sl}'))
        res.sort()
        top = ' '.join(f'[{c}]{t:.1f}' for t, c in res[:3])
        print(f'{name:12s} M={M:3d} engine k{eng[1]} {eng[0]:.1f}us | systolic best '
              f'{res[0][0] if res else float("nan"):.1f}us ({res[0][0] / eng[0] if res else float("nan"):.2f}x) {top}'
              f'{" BITS-DIFFER " + str(bad[:4]) if bad else ""}', flush=True)
