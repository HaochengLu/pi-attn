"""One decode step's GEMMs (Qwen3-0.6B: 28 layers x qkv/o/gate_up/down + lm_head,
independent weights, 1.2 GB streamed from DRAM) captured in a CUDA graph, M = 1/8/32:
cuBLAS vs each fixed invariant kernel id vs best-per-shape (cold-L2 tuned)."""
import torch

import test_gemm as G

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


def cold(fn):
    ts = []
    for _ in range(7):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[3]


for M in (1, 8, 32):
    best = {}
    for (N, K), w in zip(SHAPES + [HEAD], weights[0] + [head]):
        x = torch.randn(M, K, device="cuda").bfloat16()
        cands = (1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25)
        best[(N, K)] = min(cands, key=lambda k: cold(lambda: G.ext.linear(x, w, k, False)))
    t_cublas = step(M, lambda x, w, N, K: torch.matmul(x, w.T))
    t_best = step(M, lambda x, w, N, K: G.ext.linear(x, w, best[(N, K)], False))
    fixed = {k: step(M, lambda x, w, N, K, k=k: G.ext.linear(x, w, k, False)) for k in ((1, 13, 20, 21, 22, 23) if M <= 16 else (1, 2, 13, 24, 25) if M <= 32 else (13, 12))}
    print(f"M={M:2d}: cublas {t_cublas:7.0f}us | pi best-per-shape {t_best:7.0f}us ({t_best / t_cublas:.3f}x) {best} | "
          + " ".join(f"k{k} {t:.0f}" for k, t in fixed.items()), flush=True)
