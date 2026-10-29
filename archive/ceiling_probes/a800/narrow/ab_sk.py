"""A/B: warp stream-K ids before/after the XR generalization (old gemm_inv.cuh in ab/old, new = engine).
usage (from kernels/): python ../logs/narrow/ab_sk.py"""
import os, shutil, sys, torch
from torch.utils.cpp_extension import load
sys.path.insert(0, '.')
import test_gemm as G
from bench_tp import timeit
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
old = os.path.join(HERE, 'ab_old')
os.makedirs(old, exist_ok=True)
for f in ['gemm_torch.cu', 'gemm_dispatch.cuh', 'gemm_cutlass.cuh', 'pi_attn.cuh']:
    shutil.copy(os.path.realpath(f), old)
shutil.copy(os.path.join(HERE, 'gemm_inv_old.cuh'), os.path.join(old, 'gemm_inv.cuh'))
src = open(os.path.join(old, 'gemm_dispatch.cuh')).read()
import re
src = re.sub(r'    case 6[3-9]: return run_streamk<[^\n]*\n', '', src)
src = '\n'.join(l.replace(', dev.sms, dev.smem_optin, st);', ', dev.sms, st);') if 'run_streamk<' in l else l for l in src.split('\n'))
open(os.path.join(old, 'gemm_dispatch.cuh'), 'w').write(src)
ext_old = load(name='pi_gemm_ab_old', sources=[os.path.join(old, 'gemm_torch.cu')], build_directory=old,
               extra_include_paths=[old, os.path.join(os.path.dirname(__import__('flashinfer').__file__), 'data', 'cutlass', 'include')],
               extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
torch.manual_seed(0)
for name, N, K, S in [('70B down t8', 1024, 28672, 2), ('8B down t8', 512, 12288, 4), ('0.6B down', 1024, 3072, 2)]:
    W = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    X = torch.randn(8, K, device='cuda').bfloat16()
    for rep in range(2):
        line = f'{name} M=8 rep{rep}:'
        for k in [43, 51, 61, 62]:
            to = timeit(lambda w: ext_old.linear(X, w, k, False, S), ws)
            tn = timeit(lambda w: G.ext.linear(X, w, k, False, S), ws)
            line += f' k{k} old {to:.1f} new {tn:.1f}'
        print(line, flush=True)
