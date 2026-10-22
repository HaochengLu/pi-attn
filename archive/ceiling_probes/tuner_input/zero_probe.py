"""Does an all-zero X (as the tuner used) time GEMM tiles differently from random X? Warm
timings, random weights; prints zero/random us per candidate tile.
Usage (from kernels/): python ../archive/ceiling_probes/tuner_input/zero_probe.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../kernels"))
import test_gemm as G  # noqa: E402
def t(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1000
for name in ("8B gateup", "8B down", "4B gateup", "1.7B qkv", "0.6B gateup"):
    N, K = G.SHAPES[name]
    Wt = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    for M in (512, 2048, 4096):
        Xz = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16); Xr = torch.randn(M, K, device="cuda").bfloat16()
        row = []
        for k in G.candidates(M, N, K)[:6]:
            tz = t(lambda: G.ext.linear(Xz, Wt, k, False)); tr = t(lambda: G.ext.linear(Xr, Wt, k, False))
            row.append(f"k{k} {tz:.0f}/{tr:.0f}({tr / tz:.3f})")
        print(f"{name:12s} M={M:5d} zero/random us: " + " ".join(row), flush=True)
