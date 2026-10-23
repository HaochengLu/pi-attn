"""Probe 16: is PI's M = 1 projection deficit inside a decode graph (gemm_step_graph.txt,
1.04x cuBLAS) a property of the kernels or of running different kernels back to back?
Qwen3-0.6B, 28 layers of independent weights, M = 1. For cuBLAS and for the engine's tuned
choice per shape: one graph per shape (28 launches of that shape only) and the full mixed
step (28 x qkv/o/gate_up/down). If the mixed step costs more than the sum of the
per-shape graphs, the extra is switching between kernels, not streaming weights.
Usage (from kernels/): python ../archive/ceiling_probes/graph_mix/graph_mix.py"""
import sys

import torch

sys.path.insert(0, ".")
import test_gemm as G

LAYERS = 28
SHAPES = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)]
PI = {(4096, 1024): 7, (1024, 2048): 31, (6144, 1024): 1, (1024, 3072): 31}  # results/e2e_matrix/tuning.txt, M = 1
weights = [[(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for (N, K) in SHAPES] for _ in range(LAYERS)]
xs = {K: torch.randn(1, K, device="cuda").bfloat16() for (_, K) in SHAPES}
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def graph_time(run):
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    ts = []
    for _ in range(21):
        flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[10]


def mm(name, x, w, N, K):
    return torch.matmul(x, w.T) if name == "cublas" else G.ext.linear(x, w, PI[(N, K)], False)


for name in ("cublas", "pi"):
    per = []
    for i, (N, K) in enumerate(SHAPES):
        per.append(graph_time(lambda: [mm(name, xs[K], layer[i], N, K) for layer in weights]))
    mixed = graph_time(lambda: [mm(name, xs[K], w, N, K) for layer in weights for w, (N, K) in zip(layer, SHAPES)])
    print(f"{name:6s} per-shape graphs " + " ".join(f"{t:6.1f}" for t in per)
          + f" | sum {sum(per):7.1f} us | mixed step {mixed:7.1f} us | mixed - sum {mixed - sum(per):+6.1f} us", flush=True)
