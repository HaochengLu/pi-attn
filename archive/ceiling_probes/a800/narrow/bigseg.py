"""Large-tile segment-parallel GEMM at mid M against cuBLAS and the engine's S = 1 best (warm, as test_gemm --bench).
usage (from kernels/): python ../logs/narrow/bigseg/bigseg.py"""
import os, re, shutil, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
HERE = os.path.dirname(os.path.abspath(__file__))
inc = os.path.join(HERE, 'inc')
os.makedirs(inc, exist_ok=True)
for f in ['gemm_dispatch.cuh', 'gemm_cutlass.cuh', 'gemm_inv.cuh', 'pi_attn.cuh']:
    shutil.copy(os.path.realpath(f), inc)
p = os.path.join(inc, 'gemm_cutlass.cuh')
s = open(p).read()
s = s.replace('constexpr int kSktMaxWorkers = 256, kSktMaxTile = 64 * 128;', 'constexpr int kSktMaxWorkers = 512, kSktMaxTile = 128 * 256;')
s = s.replace('constexpr int kSegMaxTiles = 1024;', 'constexpr int kSegMaxTiles = 4096;')
open(p, 'w').write(s)
major, minor = torch.cuda.get_device_capability()
ext = load(name='bigseg', sources=[os.path.join(HERE, 'bigseg.cu')], build_directory=HERE,
           extra_include_paths=[inc, os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
NAMES = ['segpar 128x128', 'segpar 128x256', 'segpar 128x64', 'segpar 64x128', 'segpar 64x64', 'sk 128x128', 'sk 128x256', 'sk 128x64']
torch.manual_seed(0)
for name, N, K in [('8B down', 4096, 12288), ('8B o', 4096, 4096), ('8B qkv', 6144, 4096), ('4B down', 2560, 9728),
                   ('3.5-4B down', 2560, 9216), ('1.7B down', 2048, 6144), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    for M in [128, 256, 512, 1024]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        tc = G.timeit(lambda: torch.matmul(X, W.T))
        e1 = min((G.timeit(lambda: G.ext.linear(X, W, k, False, 1)), k) for k in G.candidates(M, N, K, 1))
        line = f'{name:11s} M={M:4d}: engine S=1 {e1[0] / tc:.2f}x (k{e1[1]})'
        for S in (1, 2, 4):
            if K % (S * 256) or K // S < 1024:
                continue
            ref = G.ext.linear(X, W, 1, False, S)
            best, bad = None, []
            for c, nm in enumerate(NAMES):
                y = ext.run(X, W, c, S)
                if y is None:
                    continue
                if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                    bad.append(nm); continue
                t = G.timeit(lambda: ext.run(X, W, c, S))
                best = (t, nm) if best is None or t < best[0] else best
            if best:
                line += f' | S={S} {best[1]} {best[0] / tc:.2f}x{" BITS " + str(bad) if bad else ""}'
        print(line, flush=True)
    del W
    torch.cuda.empty_cache()
