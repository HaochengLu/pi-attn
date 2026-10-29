"""Mid-M (128-1024) stream-K tiles at 1-3 CTAs per SM vs the best engine id and cuBLAS (warm, as test_gemm --bench).
usage (from kernels/): python ../logs/narrow/tiles/try_midm.py"""
import os, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='try_midm', sources=[os.path.join(HERE, 'try_midm.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
NAMES = ['64x64x64', '128x64x32', '128x128x32', '64x128x32', '128x64x64']
torch.manual_seed(0)
for name, N, K in [('8B down', 4096, 12288), ('4B down', 2560, 9728), ('1.7B down', 2048, 6144), ('8B o', 4096, 4096),
                   ('8B qkv', 6144, 4096), ('0.6B down', 1024, 3072)]:
    S = G.gemm_segments(N, K)
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    for M in [128, 256, 512, 1024]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ref = G.ext.linear(X, W, 1, False, S)
        tc = G.timeit(lambda: torch.matmul(X, W.T))
        eng = min((G.timeit(lambda: G.ext.linear(X, W, k, False, S)), k) for k in G.candidates(M, N, K, S))
        best, bad = None, []
        for c, nm in enumerate(NAMES):
            for cps in (1, 2, 3):
                y = ext.run(X, W, c, S, cps)
                if y is None:
                    continue
                if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                    bad.append((nm, cps)); continue
                t = G.timeit(lambda: ext.run(X, W, c, S, cps))
                best = (t, nm, cps) if best is None or t < best[0] else best
        print(f'{name:10s} M={M:4d}: engine k{eng[1]} {eng[0] / tc:.2f}x cuBLAS | new best {best[1]} cps{best[2]} '
              f'{best[0] / tc:.2f}x{" BITS " + str(bad) if bad else ""}', flush=True)
