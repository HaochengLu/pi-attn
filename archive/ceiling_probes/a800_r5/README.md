# A800 round 5: component ablation, prefill grid order, a 16-warp decode kernel, the mma operand swap (2026-11-14)

One A800-SXM4-80GB (sm_80, 108 SMs, 40 MiB L2), driver 535.129.03, CUDA 12.6, the tree of bf73578 and the
changes below; SM clock 1365-1410 MHz under load in every log (the `MHz` column). Kernel times are microseconds per
call, one CUDA graph over a ring of per-layer K/V caches larger than L2 (`bench_attention.timeit`), Qwen3 shapes
(16 or 32 query heads, 8 kv heads, head dim 128, page 16); decode workspaces and grids are sized for 8192 keys per
request, as the engine's graphs are (`--max-seq-len 8192`). Engine numbers are medians of three interleaved launches,
a fresh server per mode and launch.

| # | question | answer | shipped |
|---|---|---|---|
| 1 | Which half of PI causes each end-to-end gap on the A800? | On the one workload that lost more than 1% (512/256 at 8 requests, -5.6%), attention caused 70% of it and the GEMM 30%. §5 had ablated 1 and 32 requests only and said attention "never costs more than 0.3%". | the ablation's workloads, `eval/e2e_ablation.sh` |
| 2 | Does the H100's longest-first prefill grid help class A? | Yes for one to eight requests prefilled from an empty or short prefix (0.76-0.98x of the ascending grid's time), no when the launch's K/V exceed L2 or the launch is split (1.00-1.27x). | longest first when the launch is one split and its K/V fit L2 |
| 3 | Does any bit-identical class-A decode kernel beat `decode_kernel` somewhere? | A whole fold group per 16-warp CTA (decode90's program in sm_80's shared memory): 0.70-0.96x from 320 keys with a CTA per group in one wave. With the operand swap of #4: 0.48-0.97x from 96 keys, and 0.63-1.14x the better FlashInfer kernel's time where `decode_kernel` took 0.79-2.06x. | `decode16_kernel`, chosen per step by `decode_plan` |
| 4 | Is `mma.sync` symmetric in its operands on sm_80, and does putting the keys in the 16-row operand pay? | Symmetric: 0 of 1,689,600,000 elements differ. In `decode_kernel` (4 warps, a tile at a time) halving the mma buys nothing; in the 16-warp kernel the swapped program takes 0.69-0.84x the unswapped one's time. | the swapped program is `decode16_kernel` |

Not shipped: a third prefill order (row tiles descending, row tile fastest: 0.92-1.03x of ascending, never better
than the shipped rule where that rule changes the order), the unswapped 16-warp decode kernel, helper CTAs on a 1-D
grid (integer divisions per thread cost 0.2-0.7 us a launch), `decode16_kernel` beyond one short group more than
CTAs per column.

Bits, on the final tree (`bits_final_A800.log`; the same lines for every intermediate tree in `bits_*.log`):

    EQ prefill grid order (the device's choice, ascending and longest first, variants [0, 1, 2, 3, 4], splits 0/1/2/3/7): 4500 launches against the ascending one-split reference, 0 mismatches
    test_invariance.run_checks with prefill order 0, DECODE_VARIANTS [1, 3, 4]: checks failed: 0 []
    test_invariance.run_checks with prefill order 1, DECODE_VARIANTS [1, 3, 4]: checks failed: 0 []
    test_invariance.run_checks with prefill order 2, DECODE_VARIANTS [1, 3, 4]: checks failed: 0 []
    EQ decode variant 3 vs decode_kernel: 432 comparisons, 0 mismatches
    EQ decode variant 4 vs decode_kernel: 432 comparisons, 0 mismatches
    EQ decode variant 0 vs decode_kernel: 432 comparisons, 0 mismatches

The suites, the kernel hashes against the A800 reference, the engine's rollouts and detcheck on the shipped tree are
in `results/a800_r5/`.

## Files

| path | what |
|---|---|
| `component_ablation/` | `abl.sh` (the launcher of every engine A/B here), `abl_sum.py`, `ab_sum.py`; `before_bf73578/`: the four modes on the round-4 tree, per-launch JSON, loadgen logs and SM clocks; `schedule_ab/`: FlashInfer mode and the round-5 engine under four schedules (`det_old`: `PI_DECODE90=0 PI_PREFILL_ORDER=1`, the round-4 schedule; `det_decode` / `det_prefill`: one of the two new) |
| `prefill_order/` | `r5_prefill_bench.py` + `orders_A800.log` (ascending, longest first, descending; 21 shapes x 2 head counts), `r5_prefill_rule.py` + `rule_A800.log` (the shipped rule, 27 x 2 shapes) |
| `decode16/` | `r5_decode_bench.py`, `r5_wrap_ab.py`; `unswapped_A800.log` (16-warp kernel against `decode_kernel` and FlashInfer; `h1` / `h3`: half the mma removed, wrong bits), `helpers_1d_A800.log`, `wrapper_ab.txt`, `swap_A800.log` (v1 `decode_kernel`, v3 unswapped, v5 swapped, B.v3 unswapped with half the mma), `layout_A800.log` and `last_group_A800.log` (the swapped kernel with and without helper rows, beyond one wave: the planner's rule); `pi_attn_decode16_unswapped.cuh`, `pi_attn_decode16s_probe.cuh` (the two kernels as probed), `probe_tree.diff` (bf73578 + three prefill orders + the unswapped kernel on a 1-D grid + `PI_PROBE_HALF_MMA`) |
| `mma_operand_swap/` | `sym_probe.cu`, `sym_probe.py`, `sym_A800.log` |
| `r5_bits.py`, `bits_*.log` | the bit checks above |

## 1. Component ablation (round-4 tree, Qwen3-1.7B)

`abl.sh` as `h100_component_ablation/abl.sh`: `--attention-backend flashinfer` (FI), `--attention-backend pi`,
`--row-invariant-gemm --attention-backend flashinfer`, `--deterministic`; 32 running requests (64 for the last
row), `--max-seq-len 8192 --chunked-prefill-size 2048`, greedy.

| workload | FI tok/s | PI attention only | PI GEMM only | both | TPOT: attention, GEMM, both (us) | TTFT: attention, GEMM, both |
|---|---|---|---|---|---|---|
| 64/256 c1 | 283 | +1.8% | +3.6% | +5.6% | -65, -124, -190 | +7.7%, +5.0%, +10.2% (5.6 ms) |
| 512/256 c8 | 1908 | -3.8% | -1.6% | -5.6% | +159, +71, +251 | +2.0%, -1.1%, +0.6% |
| 512/256 c32 | 5064 | +1.3% | -1.1% | -0.6% | -96, +94, +12 | +1.8%, -2.9%, +3.6% |
| 2048/256 c16 | 1900 | -0.9% | -0.2% | -0.8% | +13, +6, -3 | +6.0%, +1.8%, +7.3% |
| 4096/128 c4 | 541 | -1.4% | +2.2% | +0.8% | +55, -116, -67 | +3.9%, -3.1%, +0.7% |
| 2048/128 c8 | 1118 | -1.4% | +1.1% | -0.4% | +5, -39, -21 | +8.3%, -3.2%, +4.1% |
| 512/256 c64 | 7026 | +1.5% | -0.3% | +0.4% | -139, +64, -62 | +2.0%, -3.6%, +4.4% |

The halves add (the sum of the two is within 0.8 points of both). Share of the PI-vs-FI gap by time per output
token: 512/256 c8, the only workload more than 1% behind, 159 of 251 us from attention (63%; 70% by throughput);
everywhere else PI's time per token is within 67 us of FI's or below it, and the GEMM is the faster half at 1, 4 and
8 requests with long prompts, attention at 32 and 64. Time to first token is attention's: +2% to +8% with PI
attention alone, -3.6% to +1.8% with the PI GEMM alone on every prompt of 512 tokens or more. So §5's attribution
of the batched gaps to the GEMM did not hold at 8 requests (decode attention at 513-768 keys: `decode_kernel` 30 us
per layer against FlashInfer's 20, #3), and its attribution of the prefill gap to attention did.

## 2. Prefill grid order

`prefill_kernel`'s grid was (row tile, kv head, request x split) with the row tile fastest and ascending. Longest
first (`lf`): (kv head, request x split, row tile), row tiles descending in the slowest index, as
`prefill90w_kernel`'s on Hopper. Descending (`desc`): the old grid with the row tile reversed. us per call, 16
query heads; (x) = against FlashInfer's prefill:

| requests x new tokens (+ prefix) | FlashInfer | ascending | longest first | descending |
|---|---|---|---|---|
| 1 x 512 | 60.4 | 33.1 (0.55) | 29.5 (0.49) | 34.2 |
| 1 x 1024 | 71.1 | 74.4 (1.05) | 63.6 (0.89) | 68.2 |
| 1 x 2048 | 171.8 | 210.2 (1.22) | 160.6 (0.93) | 199.5 |
| 1 x 4096 | 500.8 | 642.3 (1.28) | 539.5 (1.08) | 601.1 |
| 1 x 8192 | 1659.0 | 2195.6 (1.32) | 2030.8 (1.22) | 2085.3 |
| 1 x 2048 + 2048 | 435.4 | 507.6 (1.17) | 458.9 (1.05) | 465.0 |
| 1 x 2048 + 6144 | 822.3 | 1099.6 (1.34) | 1050.2 (1.28) | 1059.9 |
| 2 x 1024 | 90.2 | 119.2 (1.32) | 95.9 (1.06) | 116.4 |
| 4 x 512 | 55.5 | 76.3 (1.37) | 64.8 (1.17) | 74.1 |
| 8 x 512 | 92.9 | 131.1 (1.41) | 117.5 (1.26) | 131.0 |
| 8 x 1024 | 257.8 | 370.3 (1.44) | 349.8 (1.36) | 371.2 |
| 32 x 512 | 296.6 | 457.5 (1.54) | 474.6 (1.60) | 457.5 |
| 16 x 2048 | 1644.7 | 2361.6 (1.44) | 2525.6 (1.54) | 2352.2 |
| 4 x 2048 + 2048 | 1188.2 | 1630.0 (1.37) | 1637.2 (1.38) | 1613.4 |
| 1 x 512 + 8192 | 255.9 | 400.7 (1.57) | 431.0 (1.68) | 405.2 |
| 1 x 128 + 4096 | 50.6 | 67.2 (1.33) | 77.5 (1.53) | 67.1 |
| 1 x 128 + 16384 | 114.9 | 210.0 (1.83) | 258.8 (2.25) | 209.9 |

Longest first loses in two regimes. (a) Launches whose K/V exceed L2 (40 MiB): the row tiles of a (request, kv head)
read the same K/V; ascending, they are adjacent in the grid and run together, so each byte comes from DRAM once;
longest first puts a whole row-tile level of every (request, kv head) between them, and with more K/V than L2 holds
every level streams it again (32 x 512: 64 MiB, +4-6%; 16 x 2048: 128 MiB, +7-9%). (b) Split launches (the planner
splits short extends over long prefixes across fold groups): their CTAs all run about as long, so there is nothing to
order, and longest first measured 0.5-27% slower. Descending keeps the tiles together and gains 0-8% on single long
requests only (0.92-1.03x over the 42 shapes). The shipped rule (`prefill_longest_first`): compute capability 8.0 (the only class-A GPU measured),
one split, and the launch's K/V, rounded up to fold groups, within L2. `rule_A800.log`, rule against ascending over
54 shapes: 0.76-1.02x; it reorders 29 of them, 0.76-0.98x on 27 and +0.8% and +1.6% on two (8 x 1024 and 16 x 512
with 32 query heads, 32 MiB of K/V). The +2.0% is 1 x 512 + 8192, a split launch that keeps the ascending order:
run-to-run noise. Against FlashInfer's prefill the 54 shapes go from 0.55-1.80x to 0.49-1.80x; the 1.80x is the
128-token extend over 16K keys, which is split and unchanged.

Before / after against FlashInfer's prefill over `bench_attention.py`'s 18 prefill shapes: results/a800_r5.

## 3. A 16-warp decode kernel for sm_80

`decode90_kernel` (Hopper) runs a fold group per CTA on 16 warps with cp.async and mma.sync only; it is Hopper-only
because it holds 220 KB of shared memory. sm_80 allows 164 KB. `pi_attn_decode16_unswapped.cuh`: the same program
with 16 tile slots instead of 24: K of the group lands first, V takes the slots once S is done (160 KB, one CTA per
SM). us per layer, 16 query heads (`unswapped_A800.log`):

| requests, keys | FlashInfer (better kernel) | `decode_kernel` | 16 warps, unswapped | 16 warps, swapped (#4) |
|---|---|---|---|---|
| 1, 96 | 6.8 | 6.3 | 9.6 | 6.0 |
| 1, 192 | 11.0 | 10.0 | 10.2 | 7.5 |
| 1, 320 | 11.4 | 15.9 | 13.8 | 10.0 |
| 1, 512 | 11.8 | 23.5 | 18.2 | 12.2 |
| 1, 1024 | 12.9 | 24.9 | 20.3 | 14.7 |
| 1, 2176 | 15.3 | 26.3 | 21.8 | 16.3 |
| 1, 4200 | 19.2 | 30.1 | 25.6 | 20.2 |
| 2, 512 | 12.9 | 26.6 | 18.6 | 12.7 |
| 4, 512 | 14.7 | 27.0 | 19.6 | 13.8 |
| 4, 1024 | 18.2 | 29.2 | 24.5 | 19.0 |
| 8, 512 | 18.7 | 28.0 | 22.8 | 17.3 |
| 13, 512 | 25.0 | 29.0 | 28.0 | 22.5 |

(FlashInfer and `decode_kernel` from `layout_A800.log`, the unswapped kernel from `unswapped_A800.log`, an earlier
process of the same session; `swap_A800.log` has the three PI kernels in one process.)

Launch layout. A CTA takes an SM, so a launch has `slots` = min(groups of the capacity, SMs / (requests x kv
heads)) CTAs per (request, kv head), and CTA j runs the fold groups j, j + slots, ... one after the other.

- *One wave* (every group a CTA of its own): the swapped kernel takes 0.48-0.97x `decode_kernel`'s time at every
  length measured, 96 keys included (`layout_A800.log`: 69 points, 1 to 13 requests, 96 to 6656 keys), and
  0.63-1.14x the better FlashInfer kernel's (`decode_kernel`: 0.79-2.06x; from 256 keys 0.80-1.14x against
  1.05-2.06x). The unswapped kernel needed 320 keys to win (0.70-0.96x; 1.23x at 96 keys).
- *One group more than slots.* The extra group is the request's last, a short one. With one CTA per column (7 to 13
  requests of 8 kv heads) and at most 256 keys in it: 0.67-0.93x. When the SMs the columns leave free hold a helper
  CTA for every two columns (7 and 8 requests: 5 or 6 rows of 8 helpers), the last groups run there, two per helper:
  8 requests at 640 keys 21.3 us against 25.2 without helpers and 30.2 for `decode_kernel` (FlashInfer 20.7).
  Helpers for only some of the requests measured slower than none (9 and 10 requests), so the layout has them only
  when they cover every request. With several CTAs per column: 0.89-0.96x at 128 keys in the last group, 0.95-1.02x
  at 256 (`last_group_A800.log`).
- *A full group more, or several:* 0.97-1.28x. `decode_kernel` keeps those launches. It is at 0.95-1.11x
  FlashInfer's time there, except just past one wave (1 request at 7168 keys, 2 at 3584, 7 or 8 at 896-1024:
  1.15-1.28x), which is what remains of the short-decode gap.

`decode_plan` returns 3 (`decode16_kernel`) for the first two cases, from the step's longest request, as the Hopper
plan does; the engine captures a decode graph per kernel (round 4's `DecodeGraphRunner`, unchanged).

The wrapper matters at this scale: helper CTAs on a 1-D grid needed four integer divisions per thread to find
(request, kv head, slot). Inline, the 512 threads' software divisions cost 0.2-0.7 us per launch against the
shipped wrapper (`wrapper_ab.txt`, run 1 B against run 2 A; `decode_kernel`'s column agrees within 0.1 us between
the runs); in a function that is not inlined, 1.0-1.4 us. That is more than the helpers returned below 512 keys.
The shipped grid is `decode_grid`'s with the helper rows after the requests in y: no division, and the same time
as `decode90_kernel`'s wrapper without helpers (run 2).

Planner against `decode_kernel` and FlashInfer on the shipped tree: results/a800_r5 (`kernels/decode_plan.txt`).

## 4. The mma operand swap

`sym_probe.cu`: D[i][j] of (A = X, B = Y, C) against D'[j][i] of (A = Y, B = X, C^T), `mma.sync.aligned.m16n8k16`
bf16 x bf16 + f32, single steps and chains of 2 and 8 steps with the accumulator carried through (as S over
head_dim and o over a tile's keys do). 11 input regimes: Gaussian at four magnitude pairs, exponents spread over
2^+-6, 2^+-20 and 2^+-60 (overflow and underflow), raw random bit patterns for both operands and the accumulator
(NaN, infinities, subnormals), Gaussian with 2% and with 30% special values injected (+-0, +-inf, two NaNs, subnormals,
the largest and smallest normals), and softmax-like P against Gaussian V.

    SYM TOTAL: 0 of 1689600000 differ (0 NaN-payload-only)

(51.2 million elements per regime and chain length; up to 51,199,998 NaN outputs in one of them.) This agrees with
the tensor-core models of paper §7, which group a step's products by k alone, and with the H100
(`h100_review/`: 0 of 6.4 million).

What it buys. Decode has `group` real query rows per kv head (2 on Qwen3-0.6B / 1.7B, 4 on 4B / 8B) in the mma's 16
rows. With the keys in the 16-row operand and the query rows in the 8 columns, S^T = K Q^T of a 32-key tile is 2 mma
chains instead of 4 and o^T = V^T P^T is one accumulator per 16 dims instead of two.

- *In `decode_kernel`* (4 warps, a tile at a time behind three barriers): removing half the mma of S and of the o
  chain outright (`PI_PROBE_HALF_MMA`, wrong bits) changes its time by -2.3% to +9% (`h1/v1` in
  `unswapped_A800.log`, 72 points). It is bound by its barriers and load latencies, not by its mma. Not built.
- *In the unswapped 16-warp kernel* the same removal gave 0.93-1.03x (`h3/v3`; 0.94-1.02x in `swap_A800.log`),
  which looked like the bound for the swap. It was not: that probe kept every ldmatrix, every fma rescale and the
  exchange of tile maxima.
- *The swapped program* (`pi_attn_decode16s_probe.cuh`, shipped as `pi_attn_decode16.cuh`): a warp runs a whole
  tile's S^T and has its exact tile maximum itself (no exchange through shared memory, one barrier less); P^T is 16
  bytes per key (8 KB a group instead of 20 KB) and feeds the o chain through one ldmatrix per tile; a warp of the
  o chain issues 2 mma, 3 ldmatrix and 4 fma per tile instead of 4, 4 and 8. us per layer (`swap_A800.log`):

| requests, keys | `decode_kernel` | 16 warps, unswapped | swapped | swapped / unswapped |
|---|---|---|---|---|
| 1, 320 | 16.1 | 14.3 | 10.1 | 0.71 |
| 1, 512 | 23.5 | 18.0 | 12.4 | 0.69 |
| 1, 1024 | 25.0 | 20.3 | 14.7 | 0.72 |
| 1, 4200 | 29.9 | 25.9 | 20.1 | 0.78 |
| 2, 192 | 12.1 | 11.1 | 8.3 | 0.75 |
| 2, 512 | 26.5 | 18.4 | 12.6 | 0.68 |
| 4, 512 | 27.1 | 19.6 | 13.7 | 0.70 |
| 8, 512 | 27.9 | 23.0 | 17.3 | 0.75 |
| 8, 640 (helpers) | 30.2 | 26.1 | 21.2 | 0.81 |
| 13, 512 | 29.2 | 27.4 | 22.9 | 0.84 |

(The log's `B.v3` column is the unswapped kernel with half its mma removed.) Every element still runs `decode_kernel`'s instruction sequence with the two mma operands exchanged: head_dim
k-steps ascending from +0 for S; fma rescale, then the two key k-steps ascending, for o; P rounded elementwise; the
row sums and d formed from P in `online_step`'s order. Bits: `EQ decode variant 5 vs decode_kernel: 432 comparisons,
0 mismatches` (`bits_swap_probe_A800.log`), then the lines at the top for the shipped file. The kernel needs a GQA
group of at most 8 (the 8 columns); `decode_plan` returns `decode_kernel` for larger groups. Two mutants of the
shipped kernel (d with two roundings; the two key k-steps of P V in descending order) fail `test_invariance`'s path
check (`kernels/test_mutants.py`).

Open, not measured here: the same swap in `decode90_kernel` on Hopper (paper §7 puts short decode there at
1.4-1.7x FlashInfer and attributes it to one SM's intake of a group; on the A800 the unswapped 16-warp kernel was
at 1.5-1.6x and the swap brought it to 1.0-1.1x); the swap in the M <= 8 GEMM, whose 16-row operand is as padded;
a 2-tile-per-iteration `decode_kernel` for launches beyond one wave.

## Engine: the round-5 schedule against the round-4 schedule, one binary

`component_ablation/schedule_ab/`: Qwen3-1.7B, the round-5 engine in FlashInfer mode and in `--deterministic` under
four schedules chosen by environment (`PI_DECODE90=0`: `decode_kernel` everywhere; `PI_PREFILL_ORDER=1`: ascending
prefill grids), three interleaved launches, change against FlashInfer mode (output tok/s; time per output token in
us; time to first token):

| workload | FlashInfer tok/s | round-4 schedule | round-5 schedule | only the decode plan new | only the prefill order new |
|---|---|---|---|---|---|
| 64/256 c1 | 283 | +5.9%, -200, +12.3% | +8.4%, -275, +9.5% | +8.4%, -278, +10.3% | +5.5%, -186, +10.1% |
| 512/256 c8 | 1909 | -5.7%, +252, +1.2% | -0.9%, +40, -0.2% | -0.6%, +22, +0.3% | -5.7%, +254, +0.2% |
| 512/256 c32 | 5065 | -0.5%, +29, +0.2% | -0.3%, +23, -0.6% | -0.4%, +33, -0.5% | -0.5%, +33, -0.1% |
| 2048/256 c16 | 1900 | -0.8%, +5, +6.5% | -0.4%, +11, +4.2% | -0.8%, +11, +5.7% | -0.3%, +23, +4.1% |
| 4096/128 c4 | 541 | +0.4%, -47, +1.4% | +0.9%, -66, -0.2% | +0.7%, -70, +1.3% | +1.2%, -88, -0.2% |
| 2048/128 c8 | 1118 | -0.5%, +14, +1.8% | +0.1%, -19, +1.1% | -0.4%, +10, +1.7% | -0.0%, -11, +1.1% |
| 512/256 c64 | 7032 | -0.0%, -69, +7.8% | +0.5%, -79, +3.8% | +0.5%, -76, -0.1% | +0.4%, -78, +4.7% |
| 1024/256 c1 | 273 | -3.3%, +124, +0.8% | +4.9%, -170, -1.0% | +4.7%, -166, +0.4% | -3.3%, +127, -0.7% |
| 1024/256 c2 | 526 | -6.9%, +287, -2.2% | +1.8%, -64, -4.0% | +1.8%, -65, -3.1% | -7.3%, +303, -2.5% |
| 1024/256 c4 | 971 | -3.9%, +167, +0.4% | +2.2%, -87, -0.8% | +2.6%, -105, -0.3% | -4.3%, +186, +0.0% |

The round-4 schedule in this binary reproduces the round-4 tree (512/256 c8: -5.7% here, -5.6% there). The decode
plan removes the 512/256 c8 gap (252 -> 40 us per token) and turns one to four requests on 1024-token prompts, a
regime §5 never measured, from 3.3-6.9% behind FlashInfer mode to 1.8-4.9% ahead. The prefill order takes 1-2
points off time to first token on 2048- and 4096-token prompts; the time-to-first-token entries at 64 requests
differ between modes with the same prefill order by as much (+7.8% and -0.1%), so they are noise at that load.
The full A800 evaluation of the shipped tree: results/a800_r5.
