"""Segment rule sweep: per projection shape, best invariant-kernel time for each
PI_SEG_UNITS build (decode M cold in bursts, prefill M warm) vs cuBLAS."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch
from torch.utils.cpp_extension import load
import flashinfer

CUT = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")
TS = [int(t) for t in sys.argv[1:]] or [256, 1024, 2048]
exts = {}
for T in TS:
    bd = f"/tmp/segsweep_{T}"
    os.makedirs(bd, exist_ok=True)
    exts[T] = load(name=f"pi_gemm_seg{T}", sources=[os.path.join(KD, "gemm_torch.cu")], extra_include_paths=[KD, CUT],
                   extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89",
                                      f"-DPI_SEG_UNITS={T}"], build_directory=bd)


def segs(N, K, T):
    s = 1
    while s < 4 and N // 8 * s < T and K % (2 * s * 256) == 0 and K // (2 * s) >= 1024:
        s *= 2
    return s


def burst(fn, Ws, n=32, reps=7):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n): fn(Ws[i % len(Ws)])
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]


DEC = [1, 2, 4, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25]
BAL = {30: 16, 31: 16, 32: 32, 33: 64}
PRE = [10, 11, 12, 13]
SHAPES = {"0.6B o": (1024, 2048), "0.6B down": (1024, 3072), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048),
          "1.7B down": (2048, 6144), "4B qkv": (6144, 2560), "4B o": (2560, 4096), "4B down": (2560, 9728),
          "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B down": (4096, 12288)}
for name, (N, K) in SHAPES.items():
    nb = N * K * 2
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    X = torch.randn(4096, K, device="cuda").bfloat16()
    out = []
    for M in (1, 8, 32, 512, 4096):
        Xm = X[:M].contiguous()
        cold = M <= 64
        tm = (lambda f: burst(f, Ws)) if cold else (lambda f: burst(f, Ws[:1], n=20))
        cb = tm(lambda w: torch.matmul(Xm, w.T))
        row = [f"M={M} cub {cb:.1f}"]
        for T in TS:
            ks = (DEC + [k for k, mm in BAL.items() if M <= mm]) if cold else PRE
            ks = [k for k in ks if not (k in (21, 22, 23, 30) and K % 256)]
            best = min((tm(lambda w: exts[T].linear(Xm, w, k, False)), k) for k in ks)
            row.append(f"T{T}/S{segs(N, K, T)} {best[0]:.1f}({best[0]/cb:.2f})")
        out.append(" ".join(row))
    print(f"== {name} N={N} K={K}\n   " + "\n   ".join(out), flush=True)
    del Ws; torch.cuda.empty_cache()
