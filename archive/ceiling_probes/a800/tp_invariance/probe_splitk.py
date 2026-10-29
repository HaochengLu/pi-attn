"""Leaves as split-K: S S=1 GEMMs over K/S column slices with fp32 outputs plus one combine pass
(int64 fixed-point sum of the S fp32 partials, bf16 out), against the S=1 GEMM (best id each).
Uses the S=1 build of probe_segs.py.  usage (from logs/tpprobe): python probe_splitk.py"""
import os, sys
import torch, triton, triton.language as tl
from torch.utils.cpp_extension import load
from triton.language.extra.cuda import libdevice

HERE = os.path.dirname(os.path.abspath(__file__))
import flashinfer
major, minor = torch.cuda.get_device_capability()
dst = os.path.join(HERE, 'S1')
ext = load(name='pi_gemm_probe_S1', sources=[os.path.join(dst, 'gemm_torch.cu')],
           extra_include_paths=[dst, os.path.join(os.path.dirname(flashinfer.__file__), 'data', 'cutlass', 'include')],
           extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
           build_directory=dst)


@triton.jit
def combine(P, Y, n, S: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    acc = tl.zeros([BLOCK], tl.int64)
    for s in tl.static_range(S):
        p = tl.load(P + s * n + o, mask=m, other=0.)
        acc += libdevice.llrint(p.to(tl.float64) * 268435456.0)
    tl.store(Y + o, (acc.to(tl.float64) * (1.0 / 268435456.0)).to(tl.bfloat16), mask=m)


def timeit(fn, iters=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


LARGE = list(ext.large_kernels())
SHAPES = [('qwen3.5-4B o/gdn-out', 2560, 4096), ('qwen3.5-4B down', 2560, 9216), ('qwen3-8B o', 4096, 4096),
          ('qwen3-8B down', 4096, 12288), ('70B o', 8192, 8192), ('70B down', 8192, 28672)]
torch.manual_seed(0)
for name, N, K in SHAPES:
    W = torch.randn(N, K, device='cuda').bfloat16() * 0.02
    for M in [2048, 8192]:
        X = torch.randn(M, K, device='cuda').bfloat16()
        def best(x, w, fp32):
            r = None
            for k in LARGE:
                if ext.linear(x, w, k, fp32) is None:
                    continue
                t = timeit(lambda: ext.linear(x, w, k, fp32))
                r = (t, k) if r is None or t < r[0] else r
            return r
        base = best(X, W, False)
        line = f'{name:22s} N={N:5d} K={K:5d} M={M:5d}: S=1 {base[0]:.0f}us(k{base[1]})'
        for S in [2, 4, 8]:
            ks = K // S
            if ks % 256:
                continue
            xs = [X[:, s * ks:(s + 1) * ks] for s in range(S)]
            ws = [W[:, s * ks:(s + 1) * ks].contiguous() for s in range(S)]
            leaf = best(xs[0], ws[0], True)
            P = torch.empty(S, M, N, device='cuda')
            Y = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
            def run():
                for s in range(S):
                    P[s] = ext.linear(xs[s], ws[s], leaf[1], True)
            t_leaves = timeit(run)
            t_comb = timeit(lambda: combine[(triton.cdiv(M * N, 1024),)](P, Y, M * N, S=S, BLOCK=1024))
            line += f' | S={S} leaves {t_leaves / base[0]:.2f}x + combine {t_comb / base[0]:.2f}x = {(t_leaves + t_comb) / base[0]:.2f}x'
            del P
        print(line, flush=True)
        del X
    del W
    torch.cuda.empty_cache()
