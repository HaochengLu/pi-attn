"""Is the 32x16 stream-K bit difference specific to S > 1? Same tile, relay exercised (chains > workers), S = 1 and 2.
usage (from kernels/): python ../logs/narrow/tiles/bits_32x16.py"""
import os, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='try_tiles', sources=[os.path.join(HERE, 'try_tiles.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
torch.manual_seed(0)
for N, K in [(1024, 8192), (1024, 4096), (2048, 4096)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    for S in (1, 2):
        for M in (32, 64, 96, 128):
            X = torch.randn(M, K, device='cuda').bfloat16()
            ref = G.ext.linear(X, W, 1, False, S)
            res = []
            for c, nm in enumerate(['32x16', '16x32', '32x32']):
                for cps in (1, 2):
                    y = ext.run(X, W, c, S, cps)
                    if y is None:
                        res.append(f'{nm}/{cps}:refused'); continue
                    d = (y.view(torch.int16) != ref.view(torch.int16))
                    res.append(f'{nm}/{cps}:{"ok" if not d.any() else "DIFF rows " + str(sorted(set(d.nonzero()[:, 0].tolist()))[:6]) + " cols " + str(sorted(set((d.nonzero()[:, 1] // 16).tolist()))[:6])}')
            print(f'N={N} K={K} S={S} M={M}: ' + ' '.join(res), flush=True)
