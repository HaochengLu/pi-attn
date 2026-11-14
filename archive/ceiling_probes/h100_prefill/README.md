# H100 prefill attention: what closed the gap to FlashInfer, and what did not

Measured on H100 (Modal, CUDA 12.6.3, FlashInfer 0.6.18) against pi-attn @ 61ff596. Times are µs per call from `pre_bench.py`: one CUDA graph over a ring of per-layer KV caches, as `kernels/bench_attention.py`. FlashInfer is measured in the same call with both backends; `fa3` is its sm_90 kernel and what `auto` picks.

## Result

`prefill90w_kernel` replaces `prefill90_kernel` as the Hopper prefill kernel. Outputs are bit-identical (`test_invariance` 0 failures with variants 1–7, `test_directed` 0, the attention hashes of `results/modal/H100_sm90/hashes_H100.json`).

| Shape | FlashInfer (best backend) | Before (61ff596) | After | After / FlashInfer |
|---|---|---|---|---|
| hq=16 b=1 q=512 | 15.3 | 21.6 | 19.2 | 1.25 |
| hq=16 b=1 q=1024 | 22.7 | 42.7 | 33.4 | 1.47 |
| hq=16 b=1 q=2048 | 62.9 | 131.9 | 70.9 | 1.13 |
| hq=16 b=1 q=4096 | 174.6 | 405.0 | 223.0 | 1.28 |
| hq=16 b=8 q=1024 | 118.1 | 222.3 | 155.3 | 1.31 |
| hq=16 b=8 q=2048 | 358.6 | 724.1 | 494.8 | 1.38 |
| hq=16 b=1 q=2048 prefix=2048 | 127.5 | 307.7 | 183.5 | 1.44 |
| hq=16 b=1 q=128 prefix=4096 | 31.9 (fa2) | 36.3 | 31.4 | 0.98 |
| hq=32 b=1 q=512 | 15.9 | 24.6 | 19.5 | 1.23 |
| hq=32 b=1 q=1024 | 37.2 | 72.1 | 41.1 | 1.10 |
| hq=32 b=1 q=2048 | 100.0 | 215.8 | 123.2 | 1.23 |
| hq=32 b=1 q=4096 | 314.8 | 711.5 | 406.2 | 1.29 |
| hq=32 b=8 q=1024 | 214.3 | 415.9 | 293.6 | 1.37 |
| hq=32 b=8 q=2048 | 700.9 | 1388.6 | 940.8 | 1.34 |
| hq=32 b=1 q=2048 prefix=2048 | 219.3 | 534.3 | 324.6 | 1.48 |
| hq=32 b=1 q=128 prefix=4096 | 49.4 (fa2) | 62.5 | 52.6 | 1.06 |

"Before" is `logs/05_longest_first_and_exit_probes.log` (column `v5`), "After" and FlashInfer are the final validation run (`logs/10_validation.log`, column `v0`). The target of 1.15× is met on four rows and missed on twelve (1.23–1.48×).

## What moved the time

| Change | Effect (hq=32 b=1 q=4096 unless noted) | Log |
|---|---|---|
| Longest-first grid: row tiles descending in the slowest grid index | 711 → 638 on the old kernel; 72 → 52 at q=1024. Causal row tiles cost their position; in ascending order the last SMs run the longest CTAs alone. A dispatch simulation gives 1.15× (q=4096) to 1.66× (q=1024) of the ideal makespan ascending and 1.00–1.02× descending. | `05` |
| `zero_reg()`: the O accumulator is reset by an add on an opaque register, not by the constant | ptxas stops serialising wgmma (C7511 gone): 8 HGMMA per arrive / wait instead of one. 454 → 399 with three warp groups, 711 → 638 on the old kernel in its original order. The trigger is the constant reset, not the register count (`c7511/`). | `08`, `c7511/run7.log` |
| Three consumer warp groups per SM (192 rows per CTA) instead of two | 481 → 399. One warp group is not faster when the other is idle (`06`: 602 with the second one idle against 546 with both working), so a third adds throughput; with three, each takes about a third longer per tile. | `06`, `07`, `08` |
| P rounded once per pair into the A-fragment word | 746 → 671 on the two-warp-group kernel: `cvt.rn.bf16.f32` is a slow instruction | `03` |
| Two warp groups when the 192-row CTAs do not fill the SMs twice | hq=16 q=1024: 41 → 33 | `08`, `10` |

## What did not

| Probe | Result | Log / diff |
|---|---|---|
| FlashAttention-3 structure as such (producer warp group, mbarrier rings, two consumer warp groups, softmax of tile t + 1 under P V(t), S two stages ahead) | 748 against 711: no gain while ptxas serialised | `02`, `two_wg_softmax_ahead_not_shipped.diff` |
| The same overlap after the `zero_reg` fix (S per tile, softmax on a copy of S) | Slower than the serial loop with two and with three warp groups (457 against 403, 519 against 479) | `09`, `overlapped_per_tile_and_dev_flags_not_shipped.diff` |
| Ping-pong of the two warp groups' wgmma bursts (named barriers) | 834 against 746 | `03` (flag 4) |
| S from a zero-scaled first wgmma step instead of zeroed registers | Bit-identical, no gain (741 against 746; 397 against 399) | `03`, `08` |
| One mbarrier arrival per consumer warp instead of per thread | No change alone (213 against 213 in the skeleton); kept, it is the simpler count | `04` |
| S for four tiles in one chain with a warp-group vote on `scale == 1` | Not built: a row's maximum moves in tile t of a group with probability about 1/t on the benchmark's data, so 64 rows agree with probability (1 − 1/t)^64, under 2% at t = 16 | — |

## Where the time is now

Ablations of the shipped three-warp-group kernel at hq=32 b=1 q=4096, 399 µs in total (`08`): removing the softmax saves 89, the wgmma 51, the folds and the output 50, the rescale of O 18, the K/V copies 15, the divisions 6. With softmax, wgmma, copies and rescale all removed the kernel still takes 135. FlashInfer's whole kernel takes 320.

The 32-key contract costs CUDA-core instructions, not tensor-core time: per tile and thread about 235 instructions against roughly 100 per 32 keys in FlashAttention-3 (one rescale of 64 accumulators per 128 keys instead of four, no bf16 row sum, the scale fused into the exponent). A warp group is bound by its own instruction stream, and a third warp group per SM is the last that fits: 152 registers per thread for three, about 120 for four, where O alone is 64.
