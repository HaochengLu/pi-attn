# H100 round 3: independent ceiling review (2026-11-14)

One H100 sandbox, $0.17 (`rev_probe.cu`, `rev_probe.py`, `intake_sym_H100.log`); $0.05 on an L4 compile that failed
(sm_90-only PTX). No shipped code changed. Times of this sandbox are about 40% above `h100_decode_cluster`'s for the
same code (contiguous bulk 5.3 against 2.9 us), so only ratios within the log count.

## Measured

| Question | Result | Lines |
|---|---|---|
| Is `mma.sync m16n8k16` symmetric in its operands: D[i][j] of (A = X, B = Y, C) against D'[j][i] of (A = Y, B = X, C^T)? | Bit-identical, 0 of 6,400,000 elements over five magnitude regimes. Decode could put the keys in the 16-row operand: 16 instead of 32 mma per tile for S and for the o chain. Not built, not timed. | `SYM` |
| Does a TMA tensor box per page (2-D map, 256 B x 16 slots, 4 KB per instruction) beat 16 B `cp.async` for a group's 256 KB? `copy_probe.cu` only measured a contiguous 8 KB copy and one 256 B copy per key. | No. Pages in order: 8.2 against 8.4 us; pages shuffled: 13.7 against 9.0-10.6. The smem image equals cp.async's (`CHECK`). TMA stays rejected. | `COPY ... TMA box page` |
| What sets the time of one CTA taking in a group? | Bytes per CTA, linearly: 128 KB 4.0-4.2 us, 256 KB 8.3-8.4 us, from L2 as from DRAM (7.9-8.4 against 8.4-8.8), 4 or 16 warps alike. That is 30-45 GB/s per SM. | `t16` against `t32` rows |

## What this changes in the claims

- Decode at batch 1: the body of a 512-key group is dominated by one SM's intake of 256 KB (about 6 us of the roughly
  10 us above an empty launch in `h100_decode_cluster`), not by the 32-key online step. "One group on one CTA" is the
  kernel's design, not the contract. Not tried: a 2-CTA cluster per group, one CTA taking K (S, maxima, scales, P,
  row sums, d), the other V and the o chain, P (2 real rows x 512 keys = 2 KB) and 32 scales pushed once through
  distributed shared memory: one hand-off (0.45 us) for half the intake per SM (about 3 us). p1-p6 measured other
  splits (dim slices, per-tile rings) with 4-warp bodies.
- Decode at batch 2-8 in the engine runs `decode_kernel`, not the planner column of the round-3 table: the engine's
  capacity is `--max-seq-len` (16 groups at 8192), `decode90_plan` then takes batch 1 only. `SLACK13` of
  `h100_decode_cluster/validate_rule_batch2_H100.log`: batch 8, 640 keys, v0 = 23.4 us (FlashInfer 13.1, 1.79x), where
  `decode90_kernel` measures 17.9-18.6. Capturing the decode graph per (batch, capacity bucket) and choosing by the
  lengths the host knows is bit-preserving (`EQ ... 180 comparisons, 0 mismatches`) and worth about
  28 x 5 us = 0.14 ms of a 2.87 ms step at 512/256 c8.
