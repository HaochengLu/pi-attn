"""Mid-M probe: extra CUTLASS tiles, bitwise vs the engine kernel, warm time vs best engine kernel and cuBLAS."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch
from torch.utils.cpp_extension import load
import flashinfer
import test_gemm as G
CUT = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")
H = os.path.dirname(os.path.abspath(__file__))
os.makedirs("/tmp/ctp_build", exist_ok=True)
cp = load(name="ctp", sources=[os.path.join(H, "cutlass_probe.cu")], extra_include_paths=[KD, CUT],
          extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89"],
          build_directory="/tmp/ctp_build")
NCFG = 12

def burst(fn, Ws, n=32, reps=7):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n): fn(Ws[i % len(Ws)])
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]

shapes = {"0.6B o": (1024, 2048), "0.6B down": (1024, 3072), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048),
          "1.7B down": (2048, 6144), "4B o": (2560, 4096), "4B down": (2560, 9728), "8B o": (4096, 4096),
          "8B down": (4096, 12288)}
fails = []
for name, (N, K) in shapes.items():
    nb = N * K * 2
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    for M in (64, 128, 256, 512):
        X = (torch.randn(M, K, device="cuda") * (1 + 10 * torch.rand(M, 1, device="cuda"))).bfloat16()
        ref = G.ext.linear(X, Ws[0], 13, False)
        for c in range(NCFG):
            if not torch.equal(cp.run(X, Ws[0], c).view(torch.int16), ref.view(torch.int16)):
                fails.append((name, M, c))
        cold = M <= 128
        tm = (lambda f: burst(f, Ws)) if cold else (lambda f: burst(f, Ws[:1], n=20))
        cb = tm(lambda w: torch.matmul(X, w.T))
        ks = [10, 11, 12, 13] + ([24, 25, 33] if M <= 64 else [])
        ks = [k for k in ks if not (k == 33 and M > 64)]
        res = {k: tm(lambda w: G.ext.linear(X, w, k, False)) for k in ks}
        kb = min(res, key=res.get)
        new = {c: tm(lambda w: cp.run(X, w, c)) for c in range(NCFG)}
        cs = sorted(new, key=new.get)[:3]
        print(f"{name:10s} M={M:3d} cublas {cb:7.1f} | pi k{kb} {res[kb]:7.1f} ({res[kb]/cb:.2f}) | new "
              + " ".join(f"c{c} {new[c]:.1f}" for c in cs) + f" ({new[cs[0]]/cb:.2f})", flush=True)
    del Ws; torch.cuda.empty_cache()
print("BITWISE FAILURES:", len(fails), fails[:10])
