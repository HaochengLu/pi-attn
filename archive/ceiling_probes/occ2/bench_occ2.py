"""Mid-M probe 2: 8-warp / two-CTA-per-SM CUTLASS tiles (occ_probe2.cu) in all three modes,
bitwise vs the engine, vs the engine's best id and cuBLAS. Cold bursts for M <= 512 (the
tuner's regime), hot bursts above."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch, flashinfer
from torch.utils.cpp_extension import load
import test_gemm as G
from gemm_cold import KERNELS, usable
from gemm_burst import burst, POOL
H = os.path.dirname(os.path.abspath(__file__))
B = "/uniparser/users/luhaocheng/pi_attn/build/occ_probe2"
os.makedirs(B, exist_ok=True)
op = load(name="occ_probe2", sources=[os.path.join(H, "occ_probe2.cu")],
          extra_include_paths=[KD, os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")],
          extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89"],
          build_directory=B)
NCFG = 8
shapes = {"0.6B o": (1024, 2048), "0.6B down": (1024, 3072), "1.7B down": (2048, 6144), "8B o": (4096, 4096),
          "8B down": (4096, 12288)}
fails = []
for name, (N, K) in shapes.items():
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, POOL // (N * K * 2)))]
    for M in (128, 256, 512, 1024):
        X = (torch.randn(M, K, device="cuda") * (1 + 10 * torch.rand(M, 1, device="cuda"))).bfloat16()
        ref = G.ext.linear(X, Ws[0], 13, False)
        ring = Ws if M <= 512 else Ws[:1]
        cb = burst(lambda x, w: torch.matmul(x, w.T), ring, X)
        res = {k: burst(lambda x, w: G.ext.linear(x, w, k, False), ring, X) for k in KERNELS if usable(k, M, N, K)}
        kb = min(res, key=res.get)
        new = {}
        for c in range(NCFG):
            for mode in (0, 1, 2):
                y = op.run(X, Ws[0], c, mode)
                if y is None:  # this mode does not run this shape
                    continue
                if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                    fails.append((name, M, c, mode))
                new[(c, mode)] = burst(lambda x, w: op.run(x, w, c, mode), ring, X)
        top = sorted(new, key=new.get)[:3]
        print(f"{name:9s} M={M:4d} cublas {cb:7.1f} | engine k{kb} {res[kb]:7.1f} ({res[kb]/cb:.2f}) | probe "
              + " ".join(f"c{c}m{m} {new[(c, m)]:.1f}" for c, m in top) + f" ({new[top[0]]/cb:.2f})", flush=True)
    del Ws; torch.cuda.empty_cache()
print("BITWISE FAILURES:", len(fails), fails[:10])
