"""M=1 decode GEMM: cuBLAS vs best PI kernel vs pure-stream ceiling, all cold (>= 600 MB pool)."""
import os, sys
sys.path.insert(0, "/uniparser/users/luhaocheng/pi_attn/kernels")
os.chdir("/uniparser/users/luhaocheng/pi_attn/kernels")
import torch
import test_gemm as G
from gemm_cold import BK, KERNELS
import bench_bw as B  # builds the stream probe; its own loop is skipped via env

def burst(fn, Ws, n=32, reps=7):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(n): fn(Ws[i % len(Ws)])
        e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / n)
    return sorted(ts[1:])[reps // 2]

M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
shapes = {"0.6B down": (1024, 3072), "1.7B qkv": (4096, 2048), "1.7B o": (2048, 2048), "1.7B gateup": (12288, 2048),
          "1.7B down": (2048, 6144), "8B qkv": (6144, 4096), "8B o": (4096, 4096), "8B gateup": (24576, 4096),
          "8B down": (4096, 12288)}
for name, (N, K) in shapes.items():
    nb = N * K * 2
    Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    X = torch.randn(M, K, device="cuda").bfloat16()
    cb = burst(lambda w: torch.matmul(X, w.T), Ws)
    lin = burst(lambda w: B.e.run(w, 0, 16, 128, 4), Ws)
    res = {k: burst(lambda w: G.ext.linear(X, w, k, False), Ws) for k in KERNELS
           if K % BK.get(k, 64) == 0 and M <= {30: 16, 31: 16, 32: 32, 33: 64}.get(k, 1 << 30)}
    top = sorted(res, key=res.get)[:3]
    gb = lambda us: nb / us / 1e3
    print(f"{name:12s} cublas {cb:6.1f} ({gb(cb):4.0f}) stream {lin:6.1f} ({gb(lin):4.0f}) | "
          + " ".join(f"k{k} {res[k]:6.1f}" for k in top) + f" ({gb(res[top[0]]):4.0f}) | pi/cublas {res[top[0]]/cb:.2f} pi/stream {res[top[0]]/lin:.2f}", flush=True)
    del Ws; torch.cuda.empty_cache()
