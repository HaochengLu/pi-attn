"""Cost of TP-degree invariance by column sharding, per row-parallel projection (A800, best kernel id per shape).

A model served at up to t_max GPUs computes every projection with S = gemm_segments(N, K, t_max) K segments
(test_gemm.py: the bits of a column shard are those of the whole weight for any t <= t_max). Two costs:
  tp1   the whole weight at S(t_max) against S(1): what declaring t_max costs a single-GPU deployment.
  rank  one rank's GEMM at degree t: the column shard W[N/t, K] at S(t_max) (invariant) against the
        Megatron row shard W[N, K/t] at S = 1 (not invariant: its partial sums are all-reduced). Same weight
        bytes; the column shard runs longer chains on fewer columns.
Communication is not measured (one GPU): the column layout all-gathers the layer input instead of
all-reducing the output (README).
Times: best id over the engine's candidates (test_gemm.candidates), cold L2 for M <= 512 (a ring of weight
copies), as the engine's tuner does.
usage: python bench_tp.py [t_max]"""
import sys

import torch

import test_gemm as G

T_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MS = [1, 8, 32, 128, 512, 2048, 8192]


def timeit(fn, ws, iters=20):
    for i in range(3):
        fn(ws[i % len(ws)])
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(iters):
        fn(ws[i % len(ws)])
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


def best(X, W, S):
    M, (N, K) = X.shape[0], W.shape
    ws = [W] + [W.clone() for _ in range(max(1, -(-160 * 2**20 // W.nbytes)) - 1)] if M <= 512 else [W]
    return min((timeit(lambda w: G.ext.linear(X, w, k, False, S), ws), k) for k in G.candidates(M, N, K, S))


def main():
    torch.manual_seed(0)
    for name, (N, K) in G.TP_SHAPES.items():
        S = G.gemm_segments(N, K, T_MAX)
        W = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        Xs = {M: torch.randn(M, K, device="cuda").bfloat16() for M in MS}
        line = f"tp1  {name:12s} N={N:5d} K={K:5d} S(1)={G.gemm_segments(N, K)} S({T_MAX})={S}:"
        for M in MS:
            a, b = best(Xs[M], W, G.gemm_segments(N, K)), best(Xs[M], W, S)
            line += f" M={M} {a[0]:.1f}/{b[0]:.1f}={b[0] / a[0]:.2f}"
        print(line, flush=True)
        for t in [2, 4, 8]:
            if t > T_MAX:
                continue
            Wc = W[: N // t].contiguous()
            Wr = (torch.randn(N, K // t, device="cuda") * 0.02).bfloat16()
            line = f"rank {name:12s} t={t}:"
            for M in MS:
                c, r = best(Xs[M], Wc, S), best(Xs[M][:, : K // t].contiguous(), Wr, 1)
                line += f" M={M} col {c[0]:.1f}(k{c[1]}) row {r[0]:.1f}(k{r[1]}) {c[0] / r[0]:.2f}"
            print(line, flush=True)
            del Wc, Wr
        del W, Xs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
