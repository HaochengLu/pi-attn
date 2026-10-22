"""One decode step's GEMMs (Qwen3-0.6B: 28 layers x qkv/o/gate_up/down + lm_head,
independent weights, 1.2 GB streamed from DRAM) captured in a CUDA graph, M = 1/8/32:
cuBLAS vs each fixed invariant kernel id vs best-per-shape (tuned as the engine does:
bursts over the 28 layer copies, every launch streaming from DRAM)."""
import torch

import test_gemm as G
from gemm_cold import KERNELS, usable

LAYERS = 28
SHAPES = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)]
HEAD = (151936, 1024)
weights = [[(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for (N, K) in SHAPES] for _ in range(LAYERS)]
head = (torch.randn(*HEAD, device="cuda") * 0.02).bfloat16()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def step(M, pick):
    xs = {K: torch.randn(M, K, device="cuda").bfloat16() for (_, K) in SHAPES + [HEAD]}

    def run():
        for layer in weights:
            for w, (N, K) in zip(layer, SHAPES):
                pick(xs[K], w, N, K)
        pick(xs[HEAD[1]], head, *HEAD)

    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    ts = []
    for _ in range(15):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[7]


def burst(fn, ws, reps=7):
    ts = []
    for _ in range(reps + 1):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for w in ws: fn(w)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts[1:])[reps // 2]


for M in (1, 8, 32):
    best = {}
    for i, (N, K) in enumerate(SHAPES + [HEAD]):
        ws = [layer[i] for layer in weights] if i < len(SHAPES) else [head] * 4
        x = torch.randn(M, K, device="cuda").bfloat16()
        best[(N, K)] = min((k for k in KERNELS if usable(k, M, N, K)),
                           key=lambda k: burst(lambda w: G.ext.linear(x, w, k, False), ws))
    t_cublas = step(M, lambda x, w, N, K: torch.matmul(x, w.T))
    t_best = step(M, lambda x, w, N, K: G.ext.linear(x, w, best[(N, K)], False))
    # single-id policies: only ids that run every 0.6B shape (the stream-K ids 40+ skip the segmented o/down)
    fixed = {k: step(M, lambda x, w, N, K, k=k: G.ext.linear(x, w, k, False)) for k in ((13, 20, 21, 30, 31) if M <= 16 else (2, 13, 24, 32))}
    print(f"M={M:2d}: cublas {t_cublas:7.0f}us | pi best-per-shape {t_best:7.0f}us ({t_best / t_cublas:.3f}x) {best} | "
          + " ".join(f"k{k} {t:.0f}" for k, t in fixed.items()), flush=True)
