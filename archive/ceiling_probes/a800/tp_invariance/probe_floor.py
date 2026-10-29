"""Critical path of one canonical chain: M = 1 GEMM time of the S = 1 build (best id, cold L2) as N shrinks at
fixed K. Once N / 8 chains no longer fill the SMs' bandwidth, the time stops falling: the floor is one chain of
K / 16 dependent mma steps. usage (from logs/tpprobe, after probe_segs.py): python probe_floor.py"""
import os, torch, flashinfer
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
dst = os.path.join(HERE, 'S1')
ext = load(name='pi_gemm_probe_S1', sources=[os.path.join(dst, 'gemm_torch.cu')], build_directory=dst,
           extra_include_paths=[dst, os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
IDS = list(ext.small_kernels()) + list(ext.large_kernels())
def best(x, W):
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    r = None
    for k in IDS:
        if ext.linear(x, W, k, False) is None: continue
        for i in range(3): ext.linear(x, ws[i % len(ws)], k, False)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(20): ext.linear(x, ws[i % len(ws)], k, False)
        e1.record(); torch.cuda.synchronize()
        t = e0.elapsed_time(e1) / 20 * 1000
        r = (t, k) if r is None or t < r[0] else r
    return r
for K in [4096, 8192, 16384, 28672]:
    x = torch.randn(1, K, device='cuda').bfloat16()
    line = f'K={K:5d}:'
    for N in [8192, 4096, 2048, 1024, 512, 256, 128]:
        W = torch.randn(N, K, device='cuda').bfloat16()
        t, k = best(x, W)
        line += f' N={N} {t:.1f}us({N * K * 2 / t / 1e6:.2f}TB/s,k{k})'
    print(line, flush=True)
