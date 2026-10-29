"""Cost of K segments on row-parallel shapes (the ones tensor parallelism splits along K).
Builds the engine's GEMM with gemm_segments forced to S for every shape whose K allows it
(K % (S * 256) == 0) and times the best of the tuner's candidate kernel ids per M, cold L2
for M <= 512 (rotating weight copies, as the engine tuner), checking that every id gives the same bits.
usage (from logs/tpprobe): python probe_segs.py S"""
import os, re, shutil, sys
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..')
S = int(sys.argv[1])
src = os.path.join(REPO, 'llm_infer_rust', 'src', 'kernels', 'pi')
dst = os.path.join(HERE, f'S{S}')
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith('.cuh'):
        shutil.copy(os.path.join(src, f), dst)
shutil.copy(os.path.join(REPO, 'kernels', 'gemm_torch.cu'), dst)
inv = open(os.path.join(dst, 'gemm_inv.cuh')).read()
a = 'constexpr int kSegMaxN = 1024, kSegMax = 2;'
b = 'return N <= kSegMaxN && K >= 2048 && K % 512 == 0 ? 2 : 1;'
assert inv.count(a) == 1 and inv.count(b) == 1
inv = inv.replace(a, 'constexpr int kSegMaxN = 8192, kSegMax = 8;').replace(b, f'return K % ({S} * 256) == 0 ? {S} : 1;')
open(os.path.join(dst, 'gemm_inv.cuh'), 'w').write(inv)
import flashinfer
major, minor = torch.cuda.get_device_capability()
ext = load(name=f'pi_gemm_probe_S{S}', sources=[os.path.join(dst, 'gemm_torch.cu')],
           extra_include_paths=[dst, os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
           build_directory=dst)

SHAPES = [('qwen3.5-4B o/gdn-out', 2560, 4096), ('qwen3.5-4B down', 2560, 9216), ('qwen3-8B o', 4096, 4096),
          ('qwen3-8B down', 4096, 12288), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]
MS = [1, 8, 32, 128, 512, 2048, 8192]
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
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


torch.manual_seed(0)
for name, N, K in SHAPES:
    W = torch.randn(N, K, device='cuda').bfloat16() * 0.02
    cold = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    X = torch.randn(max(MS), K, device='cuda').bfloat16()
    segs = S if K % (S * 256) == 0 else 1
    line = f'S={S} {name:22s} N={N:5d} K={K:5d} segs={segs}:'
    for M in MS:
        x = X[:M].contiguous()
        ref, best = None, None
        for k in IDS:
            if M > 64 and k in SMALL:
                continue
            y = ext.linear(x, W, k, False)
            if y is None:
                continue
            if ref is None:
                ref = y
            elif not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                print('BITS DIFFER', name, M, k, flush=True)
            t = timeit(lambda w: ext.linear(x, w, k, False), cold if M <= 512 else [W])
            if best is None or t < best[0]:
                best = (t, k)
        line += f' M={M}:{best[0]:.1f}us(k{best[1]})'
    print(line, flush=True)
    del W, cold, X
    torch.cuda.empty_cache()
