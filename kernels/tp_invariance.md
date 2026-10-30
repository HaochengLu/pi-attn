# Tensor-parallel degree invariance

The engine's outputs are a bitwise function of the model and the request whatever the batch
(README). This extends that to the tensor-parallel degree: with the layout below, a model computed
on t = 1, 2, 3, ..., 8 ranks produces the same bits. The ranks are emulated on one GPU (no multi-GPU
machine was available; see "Not claimed"). All numbers are from one A800 80GB (SXM, 108 SMs); every
one traces to a log under `results/a800/kernels/` or `archive/ceiling_probes/a800/{tp_invariance,narrow}/`.

## Mechanism: no cross-rank reduction

In this engine, tensor parallelism can change bits only where it splits a reduction: every kernel
is already invariant to the batch and to how its work is partitioned. Attention heads, column-parallel
projections (QKV, gate/up, LM head) and all elementwise and per-row operations run the same program
on the same inputs on every rank. The only reductions TP splits are the K sums of the row-parallel
projections (attention output, MLP down; in Megatron each rank sums K / t and an all-reduce adds the
partials, so the summation order depends on t).

Here no projection splits K. Every projection is column-parallel: rank r holds the weight rows of
its output columns over the whole K, the layer input is all-gathered first, and the output is
all-gathered after. An all-gather copies bits, so each output element is computed by exactly the
program it runs at t = 1. This holds for any t, including degrees that do not divide the head or
column counts evenly (3, 5, 6, 7).

The GEMM program is the engine's (`gemm_inv.cuh`): per output element, one fp32 MMA chain over
ascending k16 steps per K segment, S segments left-folded from +0. S is part of the model's
canonical program, not of the launch: every launch takes it as an argument, and a launch over a
column shard passes the whole weight's S. Any of the 51 kernel ids the tuner picks from is legal
on any shard.

## The segment count and the chain floor

A chain is a sequence of dependent MMAs, about 28 ns per k16 step on A800 (`out_floor.txt`: at
M = 1, narrowing N stops helping at 11.4 µs for K = 4096 and 54.4 µs for K = 28672). A column shard
keeps the whole K on fewer columns, so a narrow shard at decode is bounded by the chain, not by
weight bandwidth: 70B down at t = 8 (N = 1024 columns, K = 28672) runs 67.2 µs at S = 1 against
40.6 µs for the Megatron row shard of the same bytes (`out_floor_s.txt`; the row shard from
`out_shard.txt`). Segments shorten the chain: S = 2 brings it to 44.3 µs.

So the canonical S depends on the largest degree the model will be served at, t_max:
`gemm_segments(N, K, t_max)` doubles S while the narrowest shard, N / t_max columns, is at most
1024 / S wide, up to 8, as long as K stays divisible into segments that are multiples of 256 and at
least 1024 long (K % (512 S) == 0 and K / (2 S) >= 1024 for the next doubling). It applies to every
projection, not only o / down (Qwen3-8B at t_max = 8, q/k/v separate as in its checkpoint:
q/k/v/o/down get S = 4, gate/up/LM head 1; a fused qkv [6144, 4096] gets S = 2).
t_max is part of the model's canonical program: for a fixed t_max the bits are the same for every
t (also t > t_max, only slower); two deployments that declare different t_max compute different
bits. With t_max = 1 every Qwen3 / Qwen3.5 projection keeps the S the engine always had (asserted
in `test_gemm.py`); the rule differs from the old one only for N <= 512 with K >= 4096 and
K % 1024 == 0 (S = 4 or 8 instead of 2), a shape none of these models has.

## Verification

| Test | What | Result |
|---|---|---|
| `kernels/test_gemm.py` | 7 row-parallel shapes (Qwen3 4B/8B, Qwen3.5-4B, 70B o 8192x8192 and down 8192x28672) at t_max = 8: Y over the whole weight equals the concatenated Y of the column shards for t = 1..8 at M = 1, 8, 33, 300 and t = 1, 3, 8 at M = 4096, every kernel id that runs the shard; rows equal across M; S > 1 equals the explicit left fold of single-segment chains | 0 failures (`test_gemm_tp.log`) |
| `kernels/test_invariance.py` (heads) | PI-Attn prefill and decode over the head shards of t = 2, 3, 4, 8 ranks, and t = 16 (kv heads replicated, smaller GQA group per rank), against the all-heads call | 0 failures (`test_invariance_heads.log`) |
| `kernels/test_tp_model.py` | Qwen3-8B, real weights, t ranks emulated on one GPU (column layout, all-gathers by concatenation, every projection cycling through the kernel ids the tuner may pick for its shard): 4 prompts of random token ids (37-1029 tokens) prefilled as one batch, then 8 greedy decode steps; every logit of every step at t = 2, 3, 4, 8 against t = 1 | bitwise equal. Negative controls, Megatron row-parallel o/down at t = 2: bf16 partial sums change 77-87% of the logits, fp32 partials summed in fp32 (only the order differs) 79-93% (`test_tp_model.log`) |

## Cost (A800, best kernel id per shape, cold L2 for M <= 512; `bench_tp.log`, ratios within one run)

Only o / down are measured; the other projections whose S changes at t_max = 8 (q/k/v of small
models) are not. Declaring t_max = 8 on a single GPU (whole weight at S(8) against S(1)):

| Projection (N x K) | S(1) -> S(8) | M = 1 | 8 | 16 | 32 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|---|---|---|
| 70B o (8192 x 8192) | 1 -> 2 | 1.05 | 1.00 | 1.01 | 1.00 | 1.14 | 1.21 | 1.13 | 1.07 |
| 70B down (8192 x 28672) | 1 -> 2 | 1.02 | 1.02 | 1.02 | 1.00 | 1.06 | 1.06 | 1.10 | 1.09 |
| 8B o (4096 x 4096) | 1 -> 4 | 1.10 | 1.09 | 1.09 | 1.10 | 1.16 | 1.36 | 1.22 | 1.27 |
| 8B down (4096 x 12288) | 1 -> 4 | 1.05 | 1.02 | 1.03 | 1.03 | 1.08 | 1.31 | 1.13 | 1.10 |
| 4B o (2560 x 4096) | 1 -> 4 | 1.08 | 0.93 | 0.94 | 0.93 | 1.12 | 1.21 | 1.16 | 1.32 |
| 4B down (2560 x 9728) | 1 -> 2 | 1.03 | 0.86 | 0.84 | 0.85 | 1.03 | 1.04 | 1.08 | 1.06 |
| Qwen3.5-4B down (2560 x 9216) | 1 -> 4 | 1.11 | 0.95 | 0.87 | 0.86 | 1.04 | 1.06 | 1.07 | 1.17 |
Per-rank GEMM time of the invariant column shard W[N/t, K] at S(8) against the Megatron row shard
W[N, K/t] at S = 1 (same weight bytes; the row shard's sum is not invariant). Every shape at t = 8,
and the 70B shapes at t = 2 (t = 2 and 4 for all shapes are in the log):

| Projection | t | M = 1 | 8 | 16 | 32 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|---|---|---|
| 70B o | 2 | 1.07 | 1.07 | 1.08 | 1.07 | 1.04 | 1.16 | 1.19 | 1.11 |
| 70B down | 2 | 1.03 | 1.02 | 1.02 | 1.02 | 1.03 | 1.26 | 1.14 | 1.12 |
| 70B o | 8 | 1.25 | 1.36 | 1.45 | 1.46 | 1.52 | 1.16 | 1.07 | 1.14 |
| 70B down | 8 | 1.12 | 1.18 | 1.49 | 1.49 | 1.52 | 1.31 | 1.34 | 1.26 |
| 8B o | 8 | 1.28 | 1.24 | 1.34 | 1.60 | 1.57 | 1.23 | 1.39 | 1.24 |
| 8B down | 8 | 1.26 | 1.27 | 1.50 | 1.55 | 1.66 | 1.27 | 1.28 | 1.33 |
| 4B o | 8 | 1.34 | 1.41 | 1.40 | 1.69 | 1.47 | 1.42 | 1.16 | 1.28 |
| 4B down | 8 | 2.12 | 2.16 | 2.40 | 2.37 | 1.95 | 1.48 | 1.24 | 1.36 |
| Qwen3.5-4B down | 8 | 1.42 | 1.39 | 1.66 | 1.66 | 1.55 | 1.39 | 1.30 | 1.39 |
Where the gaps come from (the floors are measured in `archive/ceiling_probes/a800/narrow/`):
- Independent chains. One n8 chain advances one k16 step per ~28 ns (`out_floor.txt`), 8-9 GB/s
  of weight, so a weight streams at the bandwidth (up to 1.73 TB/s on wide weights,
  `out_floor.txt`) only with several chains per SM; a micro-benchmark puts it at 3-4
  (`out_micro.txt`, run during the 1140 MHz period, so its 6 GB/s per chain is low). A shard has
  N / t / 8 x S chains: 256 on 70B down at t = 8, 2.4 per SM: 49 µs at M = 8 against a
  34-37 µs bandwidth floor. 4B down (K = 9728) gets only S = 2 (S = 4 needs
  K % 1024 == 0), so its 320-column shard has 80 chains for 108 SMs: 2.1-2.4x at M <= 32. Small
  models at t = 8 are not a realistic deployment, but the case shows the limit. More chains need a
  larger S, which costs prefill: a canonical trade, not a kernel one.
- M = 16-128 on narrow shards. Within a segment the chain is sequential, so the work splits only
  over (columns x segments), and an SM that covers few columns re-reads X from L2 for each of them.
  Warp stream-K with 16 X rows (id 63) and CUTLASS stream-K over (tile, segment) chains (id 64,
  32x32 tiles) took 70B down t = 8 from 68.4 to 62.2-62.8 µs at M = 16-32; X re-reads and the
  chain count bound the rest (`midm3.txt`; five other designs are in the archive, all slower).
- M = 512-2048: a narrow shard has few large tiles for 108 SMs, plus the segment cost. Stream-K
  over 128x128 / 128x256 tiles (ids 65 / 66, with S > 1 through per-device scratch) brought 70B o
  t = 8 at M = 2048 from 1.39x to 1.07x and 70B o / down t = 2 at M = 512 to 1.16x / 1.26x
  (`archive/ceiling_probes/a800/narrow/bench_tp_before_workspace.log` vs `bench_tp.log`); the
  row shard gains from the same ids at S = 1, so 1.16-1.48x remains at M = 512 for t = 8. The
  64-deep k-stage tiles (ids 67-72, `archive/ceiling_probes/a800/close/probe_gemm_cfg.txt`) speed
  up both layouts at M = 128-2048; at M = 128 the row shard gains more (70B down t = 8: 1.43x to
  1.52x), so those ratios rose. Against the previous run, column shards at M >= 8 are at most 3.3%
  slower (70B o t = 2 at M = 2048); at M = 1, where the new ids are not chosen, up to 9% (8B o t = 4:
  11.9 to 13.0 µs) is run-to-run variation.
- M = 8192: the segment cost. Ids 17-19 keep the segment total in an L2 slot instead of a second
  register fragment (bitwise equal, persistent CTAs); they take 70B down's S = 2 prefill cost from
  1.24x to 1.09x at M = 8192 (`archive/.../bench_tp_before_slot.log` vs `bench_tp.log`).

Communication is not measured (one GPU). Per layer the column layout all-gathers the attention
output, the o_proj output, the MLP intermediate and the down output (4 all-gathers) where Megatron
all-reduces the o_proj and down outputs (2 all-reduces). Bytes per rank per token, ring algorithms:
Megatron 2 x 2(t-1)/t x H bf16 elements; column layout (t-1)/t x (A + H + I + H), with A the
attention output width. Llama-3-70B (H = A = 8192, I = 28672) at t = 8: 57 KB vs 93 KB per token per layer, 1.63x
(o_proj alone: equal bytes, one more collective; down: 2.25x). Its end-to-end cost at TP = 8 needs
a multi-GPU measurement.

## Tried and rejected (`archive/ceiling_probes/a800/tp_invariance/`)

- K leaves as the TP mechanism (the canonical fixes S = t_max leaves aligned with rank boundaries;
  ranks exchange fp32 leaf partials; Fable round-1 plan). At t = 1 every leaf is a K segment:
  in-register fold +2-38% at M = 8192 for S = 2-8 (`out_S*.txt`), an int64
  fixed-point slot +11-184%, separate split-K launches with an int64 combine 1.08-3.3x
  (`out_splitk.txt`). And it supports only t dividing S, with S/t fp32 partials per element on the
  wire for t < S. Accuracy is not the issue: an int64 fixed-point or fp32 left fold of 8 leaves
  mismatches the fp64 reference as rarely as the single chain (0.03-0.17% of bf16 outputs; a bf16
  ring all-reduce 37-59%; `out_prec.txt`).
- A pipeline-drain-free segmented mainloop (`seg_drainfree/`, 2026-10-27): no gain.

## Prior art

Batch-invariant TP work keeps the all-reduce and fixes its order: TBIK (arXiv 2511.17826) matches a
binary tree over K tiles inside the GEMM with a tree all-reduce (power-of-two degrees, 22-63%
end-to-end on Qwen3-8B TP = 4); Megatron-LM PR #8421 orders an fp32 reduction for one fixed degree;
vLLM's batch-invariant mode is not invariant under TP (issue #51290). Exact long accumulators
(ExBLAS, ReproBLAS, det-train's int64 Kulisch sums) and Ozaki-scheme integer GEMMs make the sum
order-free at several times the cost or with a different numerical contract. We found no system
that removes the K split instead. Column-parallel second projections are a known layout (Megatron
rejected it for its communication); the contribution is using it as a determinism mechanism with
a canonical that is independent of the shard, and quantifying its price: the chain floor and the
parallelism floor that fix S per model.

## Not claimed

- Multi-GPU runs: communication time and end-to-end TP timings are unmeasured.
- Model-level invariance was run on Qwen3-8B; the 70B shapes are covered at the GEMM level only.
- The engine and the vLLM integration run on one GPU; neither has a TP executor yet.
