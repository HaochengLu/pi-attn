"""Mid-M projections: best of the old kernel set vs with the new CUTLASS tiles (14, 15) and
stream-K over tiles (44..47), vs cuBLAS. Bursts over >= 600 MB of weight copies (every
launch streams from DRAM, as consecutive layers in a forward)."""
import os, sys
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, KD)
import torch
import test_gemm as G
from gemm_cold import usable

OLD = (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33)
NEW = (14, 15, 44, 45, 46, 47)


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
          "1.7B gateup": (12288, 2048), "1.7B down": (2048, 6144), "4B qkv": (6144, 2560), "4B o": (2560, 4096),
          "4B down": (2560, 9728), "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B down": (4096, 12288)}
Ms = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128, 256, 512, 1024, 2048]
for name, (N, K) in shapes.items():
    nb = N * K * 2
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    for M in Ms:
        X = torch.randn(M, K, device="cuda").bfloat16()
        cb = burst(lambda w: torch.matmul(X, w.T), Ws)
        old = {k: burst(lambda w: G.ext.linear(X, w, k, False), Ws) for k in OLD if usable(k, M, N, K)}
        new = {k: burst(lambda w: G.ext.linear(X, w, k, False), Ws) for k in NEW if usable(k, M, N, K)}
        ko = min(old, key=old.get)
        allr = {**old, **new}
        ka = min(allr, key=allr.get)
        print(f"{name:11s} M={M:4d} cublas {cb:7.1f} | old k{ko} {old[ko]:7.1f} ({old[ko]/cb:.2f}) | new k{ka} {allr[ka]:7.1f} "
              f"({allr[ka]/cb:.2f}) | " + " ".join(f"k{k} {v:.1f}" for k, v in new.items()), flush=True)
    del Ws; torch.cuda.empty_cache()
