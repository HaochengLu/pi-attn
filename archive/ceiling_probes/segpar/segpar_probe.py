"""Segment-parallel 32x64x64 (id 48) with the arrival flag in static shared memory vs
broadcast by __syncthreads_or (see segpar_probe.cu), hot weights, burst timing (us).
Control: on an S = 1 shape (N 1024, K 1536), mode 1 does exactly the plain CUTLASS
tile's work (id 14), so any gap there is the kernel wrapper itself."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch, flashinfer
from torch.utils.cpp_extension import load
import test_gemm as G
from gemm_burst import burst
H = os.path.dirname(os.path.abspath(__file__))
B = "/uniparser/users/luhaocheng/pi_attn/build/segpar_probe"
os.makedirs(B, exist_ok=True)
sp = load(name="segpar_probe", sources=[os.path.join(H, "segpar_probe.cu")],
          extra_include_paths=[KD, os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")],
          extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89"],
          build_directory=B)
for (N, K, M) in [(1024, 3072, 16), (1024, 3072, 64), (1024, 1536, 128)]:
    W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    X = torch.randn(M, K, device="cuda").bfloat16()
    ref = G.ext.linear(X, W, 14, False)
    for flag in (True, False):
        assert torch.equal(sp.run(X, W, 0, flag), ref)
    t = {(m, f): burst(lambda x, w: sp.run(x, w, m, f), [W], X) for m in (0, 1) for f in (True, False)}
    print(f"N={N} K={K} M={M:3d} S={G.gemm_segments(N, K)} | plain id 14 {burst(lambda x, w: G.ext.linear(x, w, 14, False), [W], X):6.2f} | "
          f"static flag: full {t[0, True]:6.2f} mainloop+epilogue {t[1, True]:6.2f} | "
          f"__syncthreads_or: full {t[0, False]:6.2f} mainloop+epilogue {t[1, False]:6.2f}", flush=True)
