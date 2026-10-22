"""Tuner bucket probe. The engine tunes one kernel per (N, K, M bucket) and runs it for every
M in the bucket. This times every candidate kernel on a grid of M values, then scores bucket
policies by the loss of the kernel each would run at an M against the best kernel there:

  pow2         buckets (2^(k-1), 2^k], tuned at the top (the engine before this probe)
  pow2/minimax same buckets, the kernel with the smallest worst ratio over bottom and top
  graph        the CUDA-graph capture sizes (powers of two to 128, then steps of 64 to 512,
               128 to 1024, 256 to 2048), tuned at the top (the engine after this probe)

scored over the graph sizes (the M every graphed forward runs at) and over the whole grid
(eager forwards). Projections only (the LM head runs at the decode batch size). Timing as
gemm_burst.py: back-to-back launches over weight copies, so each launch reads DRAM.
Usage (from kernels/): python ../archive/ceiling_probes/tuner_edges/gemm_grid.py [out.json]
       python ../archive/ceiling_probes/tuner_edges/gemm_grid.py --analyze out.json"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../kernels"))

POW2 = [16, 32, 64, 128, 256, 512, 1024, 2048]
GRAPH = [1, 2, 4, 8, 16, 32, 64, 128] + list(range(192, 513, 64)) + list(range(640, 1025, 128)) + list(range(1280, 2049, 256))
HALF = sorted(POW2 + [24, 48, 96, 192, 384, 768, 1536])
GRID = sorted({m for b in HALF for m in (b, b + 1)} | {(a + b) // 2 for a, b in zip(HALF, HALF[1:])} | {17})
GRID = [m for m in GRID if 17 <= m <= 2048]


def measure(path):
    import torch
    import test_gemm as G
    from gemm_burst import POOL, burst
    table = {}
    for name, (N, K) in G.SHAPES.items():
        if "lm_head" in name:
            continue
        copies = max(4, POOL // (N * K * 2))
        Ws = [(torch.randn(N, K, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        table[name] = {}
        for M in GRID:
            X = torch.randn(M, K, device="cuda").bfloat16()
            table[name][M] = {k: burst(lambda x, w: G.ext.linear(x, w, k, False), Ws, X, n=16)
                              for k in G.candidates(M, N, K)}
        print(name, "done", flush=True)
        del Ws
        torch.cuda.empty_cache()
        with open(path, "w") as f:
            json.dump(table, f)
    return table


def policy_loss(rows, tops, minimax):
    """Loss (time / best - 1) at every grid M for kernels chosen per bucket (lo, hi]."""
    losses = {}
    for lo, hi in zip([16] + tops, tops):
        pts = [m for m in rows if lo < m <= hi]
        if not pts:
            continue
        ends = [hi] if not minimax else [m for m in (lo + 1, hi) if m in rows]
        common = set.intersection(*(set(rows[m]) for m in ends))
        pick = min(common, key=lambda k: max(rows[m][k] / min(rows[m].values()) for m in ends))
        for m in pts:
            losses[m] = rows[m][pick] / min(rows[m].values()) - 1 if pick in rows[m] else float("nan")
    return losses


def analyze(table):
    graph = [m for m in GRAPH if m >= 17]
    policies = {"pow2": (POW2, False), "pow2/minimax": (POW2, True), "graph": (graph, False)}
    for title, keep in (("the graph sizes", set(graph)), ("every grid M", set(GRID))):
        print(f"mean / max loss at {title} in 17..2048")
        print(f"{'shape':12s} " + " ".join(f"{p:>18s}" for p in policies))
        total = {p: [] for p in policies}
        for name, rows in table.items():
            rows = {int(m): {int(k): t for k, t in r.items()} for m, r in rows.items()}
            cells = []
            for p, (tops, mm) in policies.items():
                ls = [v for m, v in policy_loss(rows, tops, mm).items() if m in keep]
                total[p] += ls
                cells.append(f"{100 * sum(ls) / len(ls):5.1f}% / {100 * max(ls):5.1f}%")
            print(f"{name:12s} " + " ".join(f"{c:>18s}" for c in cells))
        print(f"{'all':12s} " + " ".join(f"{100 * sum(v) / len(v):5.1f}% / {100 * max(v):5.1f}%".rjust(18)
                                         for v in total.values()))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--analyze"]:
        with open(sys.argv[2]) as f:
            analyze(json.load(f))
    else:
        analyze(measure(sys.argv[1] if len(sys.argv) > 1 else "grid.json"))
