"""Deterministic stream-K probe: bitwise vs the canonical kernel, then cold-burst time vs
the best engine kernel and cuBLAS (M = 1, 4, 8)."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch
from torch.utils.cpp_extension import load
import test_gemm as G
KERNELS = [1, 2, 4, 6, 7, 8, 9, 20, 21, 22, 24, 25, 30, 31, 32, 33]  # engine small_kernels()
BK = {k: 256 if k in (21, 22, 30) else 128 if 20 <= k <= 33 else 64 for k in KERNELS}  # kernel_bk

H = os.path.dirname(os.path.abspath(__file__))
os.makedirs("/tmp/skp_build", exist_ok=True)
sk = load(name="skp", sources=[os.path.join(H, "streamk_probe.cu")], extra_include_paths=[KD],
          extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89"],
          build_directory="/tmp/skp_build")


def burst(fn, Ws, n=32, reps=7):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n): fn(Ws[i % len(Ws)])
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]


CFGS = [(c, wp, wpsm) for c in (0, 1, 2, 3) for wp, wpsm in ((4, 4), (2, 2), (4, 8), (3, 3), (6, 6))]
SMEM = {0: 4 * 16 * 136 * 2, 1: 3 * 16 * 264 * 2, 2: 6 * 16 * 136 * 2, 3: 8 * 16 * 72 * 2}
CFGS = [c for c in CFGS if SMEM[c[0]] * c[1] <= 99 * 1024 and c[1] <= c[2]]
shapes = {"1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048), "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144),
          "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B down": (2560, 9728),
          "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B gateup": (24576, 4096), "8B down": (4096, 12288),
          "0.6B lm_head": (151936, 1024)}
fails = []
for name, (N, K) in shapes.items():
    nb = N * K * 2
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    for M in (1, 4, 8):
        X = (torch.randn(M, K, device="cuda") * (1 + 10 * torch.rand(M, 1, device="cuda"))).bfloat16()
        ref = G.ext.linear(X, Ws[0], 1, False)
        for c in CFGS:
            if K % (128 if c[0] in (0, 2) else 256 if c[0] == 1 else 64): continue
            for rep in range(3):
                if not torch.equal(sk.run(X, Ws[0], *c).view(torch.int16), ref.view(torch.int16)):
                    fails.append((name, M, c)); break
        if M == 4: continue
        cb = burst(lambda w: torch.matmul(X, w.T), Ws)
        res = {k: burst(lambda w: G.ext.linear(X, w, k, False), Ws) for k in KERNELS
               if K % BK.get(k, 64) == 0 and M <= {30: 16, 31: 16, 32: 32, 33: 64}.get(k, 1 << 30)}
        kb = min(res, key=res.get)
        skr = {c: burst(lambda w: sk.run(X, w, *c), Ws) for c in CFGS
               if not K % (128 if c[0] in (0, 2) else 256 if c[0] == 1 else 64)}
        cs = sorted(skr, key=skr.get)[:3]
        print(f"{name:12s} M={M} cublas {cb:7.1f} | best pi k{kb} {res[kb]:7.1f} ({res[kb]/cb:.2f}) | streamK "
              + " ".join(f"{c} {skr[c]:.1f}" for c in cs) + f" ({skr[cs[0]]/cb:.2f})", flush=True)
    del Ws; torch.cuda.empty_cache()
print("BITWISE FAILURES:", len(fails), fails[:10])
