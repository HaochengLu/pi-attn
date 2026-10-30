"""Mid-M GEMM: candidate CUTLASS configurations (gemm_cfg.cu) vs the best id the engine's tuner has, single GPU,
engine regime (bench_tp.timeit). Every candidate's bits must equal id 13's.
usage (from kernels/): python ../archive/ceiling_probes/a800/close/probe_gemm_cfg.py"""
import os, subprocess, sys
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
ext = load(name='gemm_cfg_probe', sources=[os.path.join(HERE, 'gemm_cfg.cu')], build_directory=HERE,
           extra_include_paths=[os.path.abspath('.'), os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
print('clock:', subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.applications.graphics', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip(), flush=True)
SHAPES = {k: G.SHAPES[k] for k in ['0.6B qkv', '0.6B down', '1.7B qkv', '1.7B o', '1.7B gateup', '1.7B down', '4B qkv', '4B o',
                                   '4B gateup', '4B down', '8B qkv', '8B o', '8B gateup', '8B down']}
torch.manual_seed(0)
for name, (N, K) in SHAPES.items():
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    S = G.gemm_segments(N, K)
    for M in [128, 256, 512, 1024]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)] if M <= 512 else [W]
        ref = G.ext.linear(X, W, 13, False, S)
        tc = timeit(lambda w: torch.matmul(X, w.T), ws)
        tb, kb = min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))
        line = f'{name:12s} M={M:5d} cuBLAS {tc:7.1f}us best id {kb} {tb / tc:.2f}x |'
        for c in range(ext.count()):
            y = ext.run(X, W, c, S)
            if y is None:
                continue
            assert torch.equal(y.view(torch.int16), ref.view(torch.int16)), f'cfg {c} bits differ'
            t = timeit(lambda w: ext.run(X, w, c, S), ws)
            line += f' c{c} {t / tc:.2f}' + ('*' if t < 0.97 * tb else '')
        print(line, flush=True)
    del W
    torch.cuda.empty_cache()
