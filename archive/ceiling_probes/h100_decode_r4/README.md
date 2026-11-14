# H100 decode, round 4: kernel choice by the step's longest request; a 2-CTA cluster per fold group (lost)

H100 (Modal, CUDA 12.6), Qwen3 shapes (16 query heads, 8 kv heads, head dim 128, page 16), one CUDA graph over
a ring of per-layer K/V caches, microseconds per layer. Unlike round 3's tables, the workspace and the grids are
sized for 8192 keys per request, as the engine's graphs are (`--max-seq-len 8192`). $1.25 in three sandboxes.

| file | what |
|---|---|
| `validate.sh`, `validate_H100.log` | the shipped tree: test_invariance, EQ, attention hashes, decode mutants, the table below, `pi_launch.cu` compiled for sm_90a ($0.65) |
| `dec_r4_bench.py`, `attn_hashes.py`, `decode_mutants.py` | what `validate.sh` runs |
| `p1_cluster2_k_v_split.diff`, `p1_cluster2_k_v_split_H100.log`, `p1_*.py`, `p1_probe.sh` | the cluster kernel (against faa5f15, on top of the shipped scheduling) and its probe ($0.26) |
| `l4_engine_compile.sh`, `l4_engine_compile_L4.log` | `cargo build --release` and `cargo check --release --tests` of the engine with `FLASHINFER_CUDA_ARCH=90` on an L4 ($0.34; the tree of the p1 diff: the Rust is the shipped Rust, `pi_launch.cu` still had the cluster branch) |

## Shipped: the planner is told the longest request

`decode_plan` (pi_attn_kernels.cu, replacing `decode90_plan`) takes an upper bound of the launch's kv lengths.
`decode90_kernel` runs when every fold group of that length gets a CTA of its own in one wave
(batch x kv heads x groups <= SMs) and the bound is at least 192 keys (320 when the launch takes more than half
the SMs). The grid still depends on (batch, capacity) only, and a wrong bound costs time, never bits.

Engine: `pi_attn_decode` takes `max_kv_len` (the scheduler's `max_seqlen`, the longest running request of the
step), so eager decode chooses per launch. For graphs, `pi_attn_decode_plan` gives the length ranges that share a
kernel; `DecodeGraphRunner` captures one graph per (batch size, kernel) and replays the one of the step's
`max_seqlen` (`ModelExecutor::decode_graph_plans`; backends without plans keep one graph per batch size). At
`--max-seq-len 8192` that is a second graph for batch sizes 1 to 16.

Planner before (round 3 at this capacity: `decode90_kernel` for one request, else `decode_kernel`) and after,
us per layer, and the ratio to the better FlashInfer kernel (`TABLE` lines; batch 32 is unchanged):

| batch | kv 96 | 320 | 640 | 2176 | 4200 |
|---|---|---|---|---|---|
| 1, FlashInfer | 4.5 | 8.1 | 8.7 | 10.3 | 12.3 |
| 1, before -> after | 6.3 -> 5.4 (1.20) | 10.0 -> 9.8 (1.21) | 14.3 (1.64) | 15.1 (1.47) | 17.8 (1.45) |
| 2, FlashInfer | 4.4 | 8.8 | 9.4 | 12.5 | 17.8 |
| 2, before -> after | 5.4 (1.23) | 13.9 -> 10.1 (1.15) | 21.8 -> 14.7 (1.56) | 22.7 -> 17.7 (1.42) | 25.1 (1.41) |
| 4, FlashInfer | 5.7 | 9.4 | 10.4 | 18.4 | 30.4 |
| 4, before -> after | 6.0 (1.05) | 13.9 -> 10.8 (1.15) | 22.0 -> 16.1 (1.55) | 25.6 (1.39) | 32.7 (1.08) |
| 8, FlashInfer | 6.0 | 10.6 | 12.9 | 31.3 | 53.0 |
| 8, before -> after | 6.4 (1.07) | 14.4 -> 12.3 (1.16) | 23.3 -> 18.5 (1.43) | 33.8 (1.08) | 54.7 (1.03) |
| 16, FlashInfer | 6.4 | 13.2 | 20.5 | 54.9 | 98.0 |
| 16, before -> after | 6.9 (1.08) | 15.8 -> 15.0 (1.14) | 27.6 (1.35) | 55.4 (1.01) | 102.9 (1.05) |

Also 512 keys: batch 2 / 4 / 8 / 16 go 20.6 / 20.6 / 21.3 / 23.5 -> 12.6 / 13.6 / 15.8 / 19.3.

Bits (`validate_H100.log`):

    test_invariance exit=0   checks failed: 0 []        (DECODE_VARIANTS [1, 2])
    EQ decode variant 2 vs decode_kernel: 432 comparisons, 0 mismatches
    EQ decode variant 0 vs decode_kernel: 432 comparisons, 0 mismatches      (the planner, told the longest request)
    attention hashes equal to results/modal/H100_sm90/hashes_H100.json: True  (decode v0, v1, v2; prefill v0)
    mutant decode_fold_unfused: 78 failed checks, kinds ['path']
    mutant decode90_d_unfused: 15 failed checks, kinds ['path']
    UNCAUGHT DECODE MUTANTS: 0 []
    pi_launch.cu sm_90a compile exit=0

Not run here: the engine end to end (rollouts, detcheck, throughput) and the Rust unit tests; the engine was
only compiled.

## Lost: one fold group on a cluster of two CTAs (K on one SM, V on the other)

`p1_cluster2_k_v_split.diff`: CTA 0 takes K and forms S, the maxima, the scales and P, writes P, the scales and m
into CTA 1's shared memory and arrives at its mbarrier; CTA 1 takes V and runs the row sums, d, the o chain and
the end of the group. One hand-off, a cluster barrier per group. v3 arrives once per thread (512), v4 once per warp.

Bits: equal to `decode_kernel` (432 comparisons each, 0 mismatches, v3 and v4; both attention decode hashes equal).

Time (`DEC` lines of the p1 log), us per layer, v2 = `decode90_kernel`:

| batch | keys | FlashInfer | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|---|
| 1 | 96 | 4.4 | 5.4 | 6.2 | 7.5 | 7.6 |
| 1 | 320 | 7.8 | 13.6 | 9.7 | 10.9 | 10.9 |
| 1 | 512 | 8.5 | 20.6 | 12.5 | 13.2 | 13.0 |
| 1 | 640 | 8.6 | 21.4 | 14.2 | 15.0 | 15.0 |
| 1 | 2176 | 10.2 | 22.0 | 15.2 | 16.9 | 16.8 |
| 2 | 512 | 9.1 | 20.2 | 13.5 | 13.4 | 13.4 |
| 4 | 640 | 10.5 | 22.0 | 15.9 | 17.1 | 17.1 |
| 8 | 512 | 11.8 | 21.2 | 15.6 | 17.3 | 17.2 |

The cluster costs 1.3 us before any key (96 keys: 7.5 against 6.2) and gives back 0.6 us on a full group. The
review's 3 us assumed the 256 KB of a group are taken in serially before the work; `decode90_kernel` already
runs S and P while V is in flight, so splitting only removes V's share of the intake while K lands. The number of
arrivals does not matter (v3 = v4). Replacing the mbarrier by the cluster barrier would save at most one of the
two 0.45 us hand-offs and still not win. Not tried: keys in the 16-row mma operand (measured bit-identical in
h100_review; it shortens S and the o chain's issue, 0.5 us at the most by the stage times of
h100_decode_cluster).

## What is left

Batch 1 to 8 at 512 to 2176 keys is 1.3-1.6x FlashInfer: a full 512-key group takes about 12.5 us on its SM
against FlashInfer's 9, plus 2 us of partial, arrival and fold for more than one group. A group cannot be spread
over SMs for less than the hand-offs cost (p1 here, p1 to p6 of h100_decode_cluster), so the remaining distance
is the price of one 512-key group per SM: the group size is a contract constant, the 32-key step is not what
binds. From batch 16 at 640 keys on, the SMs are full and `decode_kernel` is within 1.00-1.35x.
