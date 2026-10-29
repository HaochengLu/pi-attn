"""E1: the chain floor against K segments. M = 1 (and 8) GEMM time, best id, cold L2, for narrow N (a column shard)
and K in 4096..28672, from the S = 1, 2, 4, 8 builds of probe_segs.py (S applies when K % (S * 256) == 0).
Prints time and achieved weight bandwidth. usage (from logs/tpprobe, after probe_segs.py): python probe_floor_s.py"""
import os, torch, flashinfer
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
major, minor = torch.cuda.get_device_capability()
CUT = os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')
def ext(S):
    d = os.path.join(HERE, f'S{S}')
    return load(name=f'pi_gemm_probe_S{S}', sources=[os.path.join(d, 'gemm_torch.cu')], build_directory=d,
                extra_include_paths=[d, CUT],
                extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'])
EXTS = {S: ext(S) for S in [1, 2, 4, 8]}
def best(e, x, W):
    ids = list(e.small_kernels()) + list(e.large_kernels())
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)]
    r = None
    for k in ids:
        if e.linear(x, W, k, False) is None: continue
        for i in range(3): e.linear(x, ws[i % len(ws)], k, False)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(20): e.linear(x, ws[i % len(ws)], k, False)
        e1.record(); torch.cuda.synchronize()
        t = e0.elapsed_time(e1) / 20 * 1000
        r = (t, k) if r is None or t < r[0] else r
    return r
torch.manual_seed(0)
for M in [1, 8]:
    for N in [512, 1024, 2048]:
        for K in [4096, 8192, 16384, 28672]:
            x = torch.randn(M, K, device='cuda').bfloat16()
            W = torch.randn(N, K, device='cuda').bfloat16()
            line = f'M={M} N={N:5d} K={K:5d}:'
            for S, e in EXTS.items():
                if K % (S * 256): continue
                t, k = best(e, x, W)
                line += f' S={S} {t:6.1f}us {N * K * 2 / t / 1e6:.2f}TB/s(k{k})'
            print(line, flush=True)
