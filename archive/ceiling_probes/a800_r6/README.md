# A800 round 6: where prefill time goes, three scheduling levers, and what is left (2026-11-15)

One A800-SXM4-80GB (sm_80, 108 SMs of four sub-partitions, 40 MiB L2), driver 535.129.03, CUDA 12.6, starting from
98aff15. SM clock 1380-1410 MHz under load in every log (the `MHz` column); comparisons are within one process or
one interleaved session. Hardware counters are not available in the container (`ncu`: ERR_NVGPUCTRPERM), so every
attribution is a timing ablation, a micro-benchmark, or the compiled SASS. Kernel times are microseconds per call,
one CUDA graph over a ring of per-layer K/V caches larger than L2 (`bench_attention.timeit`).

The two gaps this round set out to shrink (paper §5.3, `results/a800_r5`): time to first token +11-18% on 16K-30K
prompts, and prefill-heavy Qwen3-0.6B at 64 requests at -6.4% throughput.

| # | question | answer | shipped |
|---|---|---|---|
| 1 | Which half of PI causes the two gaps? | Attention, all of it: with PI attention alone 16K / 30K time to first token is +13.9% / +19.8% and 0.6B 2048/128 c64 is -5.9%; with the PI GEMM alone -1.7% / -1.3% and -0.1%. | - |
| 2 | What do the engine's prefill-phase batches at 64 requests look like? | One new 2048-token prompt and 58 decode rows (one-token requests). In one prefill launch the decode rows took 464 128-row CTAs for 2 real rows each, beside 14,384 empty CTAs. | the one-token tail of a batch runs through the decode kernel: 742 -> 464 us per layer (FlashInfer 500) |
| 3 | Why is a long chunk over a long prefix 1.3x FlashInfer? | A third of it is wave quantisation: 256 equal CTAs on 108 SMs run three waves with the third 37% full. | `prefill_wave_head`: the tokens whose CTAs fill whole waves, then the rest as an extend: 0.70-0.97x on every shape the rule cuts |
| 4 | Grid order when K/V exceed L2? | Descending within a (request, kv head) band: 0.965-1.002x of ascending. | for one-split launches that are not longest-first |
| 5 | What bounds a 32-key step? | Two units, each about three quarters loaded and imperfectly overlapped: the tensor cores (64 HMMA per warp, 8.1 cycles each per sub-partition) and the scalar pipes (about 320 other instructions per warp, a contract-fixed 177 of them floating point). No single component removed buys more than 20%. | integer and move overhead of the step cut from 588 to about 450 instructions: 3-4% |
| 6 | Is `ldmatrix` the bottleneck (0.5 per MMA against FlashInfer's 0.31)? | No. Alone it costs 20 cycles per sub-partition, but under two HMMA per load it is hidden; replacing all 32 by 16-byte `ld.shared` (wrong bits) changes the kernel by -0.8% to +2%. | - |
| 7 | What would a 64-key online step (a contract change) buy? | 1-3% from halving the per-step softmax work; with the barrier and copy cadence it also halves, at most about 6%. | no (report only) |

Not shipped, all bit-identical unless marked: deeper K/V prefetch through rings that reuse the Q tile (0-3% slower),
a conditional rescale (+4.8% on random scores), a mask fast path inside the step (+1.5-3%), copies split between
warp groups (+7%), three CTAs per SM (18-40% slower), CTAs with fewer active warps to fit waves (2-6% slower),
fragment-order staging for `ld.shared` loads (wrong-bits bound: no gain, +3-6% with its staging cost).

Bits of the shipped kernel against the 98aff15 build (`logs/eq_final_A800.log`):

    EQ prefill repo vs probe (variants 0-4, splits 0/1/2/3/7, both grid orders, 8 input regimes, 17 shapes, 2 seeds): 13600 launch pairs, each also against the reference build's first launch of the shape, 0 mismatches
    PACK NVIDIA A800-SXM4-80GB: 4294967296 first operands (each also as the second operand of a hashed pattern), 16777214 NaN inputs: 0 mismatches

The suites, kernel hashes, rollouts and detcheck on the shipped tree are in `results/a800_r6/`.

## Files

| path | what |
|---|---|
| `engine/` | `abl0/` (component ablation on the round-5 engine), `ab1/` (FlashInfer mode, round-5 binary, decode tail only), `c128/` (512/256 at 128 requests three times in one server), `ab2/` (FlashInfer mode, round-5 binary, the shipped engine): per-launch JSON, loadgen logs, SM clocks, `SUMMARY.md` |
| `scripts/` | `r6_abl.sh`, `r6_abl_sum.py` (engine A/B launcher and summary), `r6_probe_bench.py` (probe builds interleaved in one process), `r6_eq.py` (bit equality of two builds), `r6_nan_diag.py` (#8), `stepcount.py`, `warpsim.py` (SASS of a step: instruction histogram; an in-order single-warp timing model) |
| `probes/` | `ablation_flags_98aff15.diff` (removal flags on the round-5 kernel), `ring_additions_occupancy.diff` (K/V rings, added-instruction flags, three CTAs per SM; against the shipped kernel), `step64_warps_ldshared.diff` (the 64-key-step, active-warp and `ld.shared` probes) |
| `micro/` | `mma_rate.cu`, `op_rate.cu`, `lat.cu`, `smem_rate.cu` (throughput and latency per instruction kind by warps per SM), `mma_negzero.cu`, `pack_exhaustive.cu`, `run.sh`, `out_A800.log` |
| `logs/` | one log per probe, named in the sections below |

## 1. Attribution in the engine (`engine/abl0/SUMMARY.md`)

Round-5 engine, one launch per mode, `--attention-backend flashinfer` / `--attention-backend pi` /
`--row-invariant-gemm` / `--deterministic`; change against FlashInfer mode:

| workload | metric | PI attention only | PI GEMM only | both |
|---|---|---|---|---|
| 1.7B 16384/256 c2 | time to first token | +13.9% | -1.7% | +11.4% |
| 1.7B 30000/128 c1 | time to first token | +19.8% | -1.3% | +17.9% |
| 1.7B 4096/4 c4 | time to first token | +5.6% | +3.5% | +4.5% |
| 0.6B 2048/128 c64 | output tok/s | -5.9% | -0.1% | -6.2% |
| 0.6B 2048/128 c64 | time per output token | +6.5% | +0.5% | +6.8% |
| 0.6B 512/256 c128 | time to first token | +16.8% | +3.9% | +13.8% |

## 2. The one-token tail of a prefill-phase batch

The scheduler appends the running requests' decode rows to a batch of new prompts, and both backends ran them
through their prefill kernel. The 0.6B server log at 64 requests alternates `prefill_requests=1 decode_requests=58
prefill_tokens=2048` with single-prompt batches. `prefill_kernel`'s grid is dense in (row tile, kv head, request):
32 row tiles x 8 x 59 = 15,104 CTAs of which 720 work, and each decode row runs every key tile on eight warps for
its two GQA rows. The decode kernels give a one-token request the same bits (one canonical sequence per row), so
the engine now runs the trailing one-token requests of a batch through `pi_attn_decode`, on a grid for their longest
request instead of the page table's capacity, and the rest through prefill.
`logs/mixed_batches_A800.log` (us per layer, 16 query heads; the two PI outputs asserted bitwise equal):

| batch | FlashInfer | one prefill launch | prefill + decode |
|---|---|---|---|
| 1 x 2048 + 58 rows at 2100 keys | 499.9 | 742.1 (1.48) | 464.2 (0.93) |
| 2 x 2048 + 62 at 1500 | 516.1 | 756.9 (1.47) | 523.7 (1.01) |
| 8 x 2048 + 56 at 1500 | 1090.3 | 1539.2 (1.41) | 1387.4 (1.27) |
| 1 x 2048 (+4096) + 63 at 1500 | 892.8 | 1086.1 (1.22) | 980.2 (1.10) |
| 4 x 512 + 124 at 700 | 274.2 | 482.2 (1.76) | 291.6 (1.06) |
| 16 x 512 + 48 at 700 | 263.6 | 393.4 (1.49) | 329.3 (1.25) |
| 1 x 512 + 7 at 700 | 56.3 | 58.4 (1.04) | 51.7 (0.92) |
| 1 x 2048 + 1 at 2100 | 171.5 | 248.0 (1.45) | 177.7 (1.04) |
| 4 x 512 + 4 at 300 | 61.2 | 73.6 (1.20) | 76.1 (1.24) |
| 1 x 30 + 31 at 600 | 62.5 | 109.6 (1.75) | 66.7 (1.07) |

The one loss (four rows of 300 keys beside four prompts, +3%) is the second launch; the rule has no threshold.
The shipped tree's numbers, with the tail's grid sized as the engine sizes it: `results/a800_r6/kernels/mixed_batch.txt`.

## 3. Wave split

A 2048-token chunk of a 16-head model is 256 128-row CTAs, one per SM. Over a long prefix they all run about as
long, so the launch is three waves of 108, 108 and 40. `logs/wave_heads_A800.log` times the launch against every
cut into a head whose CTAs are a whole number of waves or half-waves and the rest as an extend (which the planner
splits over fold groups); the split is asserted bitwise equal. One launch = 1.000:

| chunk + prefix, heads | CTAs (waves) | whole-wave head | other heads |
|---|---|---|---|
| 2048 + 28000, 16 | 256 (2.37) | 1728: 0.892 | 0.979-1.093 |
| 2048 + 14336, 16 | 256 (2.37) | 1728: 0.902 | 0.986-1.216 |
| 2048 + 2048, 16 | 256 (2.37) | 1728: 0.954 | 1.004-1.133 |
| 2048 + 0, 16 (causal triangle) | 256 (2.37) | 1728: 1.215 | 1.049-1.204 |
| 1792 + 8192, 16 | 224 (2.07) | 1728: 0.771 | 0.979-1.052 |
| 1024 + 7168, 16 | 128 (1.19) | 832: 0.825 | 1.036 |
| 1280 + 8192, 16 | 160 (1.48) | 832: 0.928 | 1.194 |
| 1536 + 8192, 16 | 192 (1.78) | 832: 1.128 | 1.211-1.305 |
| 2048 + 14336, 32 | 512 (4.74) | 1728: 1.049 | 1.017-1.219 |
| 1792 + 8192, 32 | 448 (4.15) | 1728: 0.890 | 0.965-1.065 |
| 1024 + 7168, 32 | 256 (2.37) | 864: 0.901 | 0.993-1.184 |
| 512 + 8192, 32 | 128 (1.19) | 416: 0.701 | 0.890 |
| 4096 + 12288, 32 | 1024 (9.48) | 3872: 1.031 | 0.991-1.105 |
| 8192 + 0, 16 | 1024 (9.48) | 7744: 1.077 | 1.008-1.129 |

The rule (`prefill_wave_head`): one request, a prefix at least as long as the chunk, one to five waves, a last wave
at most half full; cut at the last whole wave. It cuts ten of the 34 shapes of the log's two head counts, all of
them faster (0.70-0.97x), and leaves the 24 others, of which three would have gained (1024 + 7168 and 1280 + 8192 with
16 heads, 512 + 8192 with 32, all cut by the shipped rule after this log: the first version required 128-row CTAs).
A 32-head model's 2048-token chunk (512 CTAs, a last wave of 80) is not cut: every head measured slower.

Partially filled SMs do not help: an eight-warp CTA with seven, six or five warps computing takes 1.02-1.06x the
full CTA's time, not less (`logs/step64_warps_A800.log`), so CTAs of fewer rows to fill the waves were not built.

## 4. Grid order

`results/a800_r6/kernels/prefill_order.txt` has the shipped rule; the probe (`logs/order_A800.log`, 32 shapes x 2 head
counts, the shipped kernel): descending / ascending 0.965-1.002x wherever the rule does not take longest-first and
the launch is one split (1 x 16384: 0.965, 1 x 30000: 0.978, 4 x 2048 + 2048: 0.979, 64 x 2048: 1.001).

The order is one more kernel argument. Passed as a second flag with a select on `gridDim` it made the compiled step
loop 2-3% slower on every shape (`logs/codegen_A800.log`: `fin`, `alt4`, `alt7` against `o3nr`, the same step
without the argument); as one int that the CTA subtracts from its x index (`prefill_tile_flip`) it costs nothing
(`alt8`). Addressing and prologue order (`alt`, `alt2`) made no difference.

## 5. What a step costs

The 128-row CTA runs a 32-key step in about 2,180 cycles in steady state (fit over 32 x 1024 and 32 x 2048:
0.0143 us per CTA step and 0.086 us per CTA on 108 SMs; `logs/fit_A800.log`); a 64-row CTA alone on an SM, one warp
per sub-partition, still needs 1,780 (1,440 without loads and barrier), and two per SM 2,385 (`logs/lone_A800.log`).

Micro-benchmarks (`micro/out_A800.log`), cycles per instruction per sub-partition once two or more warps share it:
HMMA m16n8k16 8.1 (latency 31), FFMA or FADD 2.0 (latency 4), EX2 8.1 (19), `ldmatrix.x4` 20, `ld.shared` of 16
bytes 1.4-2.5, SHFL latency 26; `ldmatrix.x4` with two HMMA per load 16.4, the two HMMA alone 16.2.

Removal (wrong bits; `logs/ablation_removal_A800.log`, `logs/skeleton_A800.log`), time against the round-5 kernel:

| removed | 1 x 8192 | 1 x 16384 | 32 x 512 | 32 x 2048 |
|---|---|---|---|---|
| QK MMA and its ldmatrix | 0.80 | 0.80 | 0.86 | 0.82 |
| PV MMA and its ldmatrix | 0.85 | 0.84 | 0.98 | 0.87 |
| K/V loads after the first two tiles | 0.92 | 0.93 | 0.97 | 0.94 |
| ... and the barrier | 0.90 | 0.91 | 0.95 | 0.94 |
| rescale of o | 0.96 | 0.96 | 0.97 | 0.97 |
| exp and rounding of P | 0.97 | 0.97 | 0.97 | 0.97 |
| both MMAs (softmax, loads and barrier left) | | | | 0.73 |
| softmax, rescale, loads, barrier (both MMAs left) | | | | 0.83 |
| all of the above (mask, max, shuffles, sums left) | | | | 0.56 |

The parts do not add: either half alone takes three quarters of the time. Added work (`logs/ablation_addition_A800.log`,
per step and warp, against the shipped step): 16 `ldmatrix` consumed at once +24%, 16 HMMA +16%, 64 FFMA +9%,
128 integer instructions +12%, 16 EX2 +1.4%, 4 SHFL +1.2%.

The compiled step of the round-5 kernel was 588 instructions on a copying warp, 283 of them integer arithmetic,
moves and branches: each tile copy re-formed its swizzled addresses (about 45 instructions), the mask cost four
per element, and P was rounded twice. The shipped step is 448-454 (`scripts/stepcount.py`): addresses formed once
per thread, tiles every row of the warp sees in a loop of their own, one rounding per pair. 0.96-0.97x on every
shape (`logs/step_variants_A800.log`: `o2` against `base`; `o2ns` without the loop split, `o2np` without the pack).

A branch inside the step is not the same thing as a second loop: with the mask fast path as an `if`, the compiler
emits the 32 QK MMAs back to back and the softmax after the join (`logs/first_attempt_A800.log`, `o_mask`: +1.5-3%).

## 6. ldmatrix

Two fits said `ldmatrix` (0.5 per MMA; FlashInfer's 32-row warps need 0.31) explains the gap to FlashInfer. The
probe says no: `logs/ldshared_A800.log` replaces every K and V `ldmatrix` of the step by a 16-byte `ld.shared`
(garbage fragments): 0.992-1.021x; with the four `ldmatrix` and four stores per warp that a fragment-order staging
stage would add, 1.026-1.065x. Under two MMAs per load the tensor cores, not the load, set the pace.

## 7. A 64-key step

`probes/step64_warps_ldshared.diff`, `PROBE_STEP64` (wrong bits): every second tile runs without the row-max
exchange, the scale, the rescale of o and the cross-lane sum, which is what the second half of a 64-key step would
still cost. 1 x 16384: 0.988, 32 x 2048: 0.990, 2048 + 14336: 0.988, 104 equal CTAs over 8192 keys: 0.969
(`logs/step64_warps_A800.log`). The probe keeps a barrier and a copy per 32 keys; removing the barrier outright
was worth 2% (#5), so a 64-key step is worth about 1-3% and at most about 6% of prefill kernel time, i.e. at most
about two points of the long-prompt time to first token. It would change every attention bit.

## 8. Found on the way: non-finite V

`scripts/r6_nan_diag.py`: with +-inf or NaN injected into V (2% of the elements), the 64-row and the 128-row
prefill variants of 98aff15 disagree on 93-388 output elements of early tokens (finite in one, NaN in the other).
A masked key has P = 0, and 0 x inf is NaN in the PV MMA, so a row's output depends on which later keys share its
CTA. With finite K/V (0 x finite = 0, added to an accumulator that is never -0: `mma_negzero.cu`, 0 of 2.7e10) the
masked keys change nothing, which is what the contract and every test assume. Not changed here; the guarantee
should be read as one for finite K and V.

## Critic pass

Before calling the kernel bound, an independent review of the round's numbers proposed four untried, bit-identical
ideas. Chunk alignment to the SM count became the wave split (#3). `ldmatrix` one fragment ahead of its MMAs: the
compiler already schedules the loads 15-40 instructions early (`scripts/warpsim.py` finds no wait on a load).
Fragment-order staging: #6. An S-producer warp for 32 rows: not built; its premise was the `ldmatrix` cost that #6
refutes.
