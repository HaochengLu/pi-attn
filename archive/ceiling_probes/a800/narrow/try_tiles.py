"""Small CUTLASS stream-K tiles on narrow shards: bits vs the engine, time vs the best engine id.
usage (from kernels/): python ../logs/narrow/tiles/try_tiles.py"""
import os, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='try_tiles', sources=[os.path.join(HERE, 'try_tiles.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
NAMES = ['32x16 w1', '16x32 w1', '32x32 w1']
torch.manual_seed(0)
for name, N, K, S in [('70B down t8', 1024, 28672, 2), ('70B o t8', 1024, 8192, 2), ('8B down t8', 512, 12288, 4)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    for M in [16, 32, 64]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ref = G.ext.linear(X, W, 1, False, S)
        eng = min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
        line = f'{name} M={M}: engine k{eng[1]} {eng[0]:.1f}'
        for c, nm in enumerate(NAMES):
            for cps in (2, 4, 6):
                y = ext.run(X, W, c, S, cps)
                if y is None:
                    continue
                ok = torch.equal(y.view(torch.int16), ref.view(torch.int16))
                t = timeit(lambda w: ext.run(X, w, c, S, cps), ws)
                line += f' | {nm} cps{cps} {t:.1f}{"" if ok else " BITS!"}'
        print(line, flush=True)
