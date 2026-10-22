"""Is the n8-tile weight stream slower for long K at a fixed weight size? The same 96 MB
(8B down) as [N, K] with K = 4096 .. 24576: linear stream, cuBLAS GEMV, and the best
tile/warp pattern (CTA or warp owns n8 units, K in cp.async stages) over stage widths up
to 2048 columns and 8..48 rows in flight per CTA. No MMA; bandwidth only.
Usage: python archive/ceiling_probes/long_k/bench_k.py"""
import os
import torch
from torch.utils.cpp_extension import load

H = os.path.dirname(os.path.abspath(__file__))
e = load(name="bwp_long_k", sources=[os.path.join(H, "bw_probe.cu")],
         extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89"])
BURST = 32


def burst(fn, copies, reps=7):
    ts = []
    for _ in range(reps):
        s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(BURST):
            fn(copies[i % len(copies)])
        en.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(en) * 1000 / BURST)
    return sorted(ts)[len(ts) // 2]


for K in (4096, 8192, 12288, 24576):
    N = 4096 * 12288 // K
    nb = N * K * 2
    copies = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(max(4, (600 << 20) // nb))]
    X = torch.randn(1, K, device="cuda").bfloat16()
    res = {"cublas": burst(lambda W: torch.matmul(X, W.T), copies), "linear": burst(lambda W: e.run(W, 0, 16, 128, 4), copies)}
    for bk, s in [(256, 4), (512, 2), (512, 3), (1024, 2), (1024, 3), (2048, 2)]:
        for rows in (8, 16, 32, 48):
            if s * rows * (bk + 8) * 2 <= 99 * 1024:
                res[f"tile bk{bk} s{s} r{rows}"] = burst(lambda W: e.run(W, 1, bk, s, rows), copies)
    for bk, s in [(256, 4), (512, 2), (512, 3), (1024, 2), (1024, 3)]:
        for wp in (2, 4, 6, 8):
            if wp * s * 8 * (bk + 8) * 2 <= 99 * 1024:
                res[f"warp bk{bk} s{s} w{wp}"] = burst(lambda W: e.run(W, 2, bk, s, wp), copies)
    pat = {k: v for k, v in res.items() if k.startswith(("tile", "warp"))}
    top = sorted(pat.items(), key=lambda kv: kv[1])[:4]
    gb = lambda us: nb / us / 1e3
    print(f"K={K:5d} N={N:5d}: linear {res['linear']:.1f} ({gb(res['linear']):.0f} GB/s) cublas {res['cublas']:.1f} "
          f"({gb(res['cublas']):.0f}) | best pattern " + " | ".join(f"{k} {v:.1f} ({gb(v):.0f})" for k, v in top), flush=True)
    del copies
    torch.cuda.empty_cache()
