# A800 ceiling probes (sm_80)

Hardware: A800-SXM4-80GB, 108 SMs, 1.41 GHz max SM clock; full environment in
`results/a800/ENV.txt`. Hardware performance counters are not available in the container
(ERR_NVGPUCTRPERM), so every attribution here comes from timing ablation builds.

The engine commit that ships is 489c9dc (091612d plus a planner change below). Every attempt below that is not in it was
reverted because of the measurement listed with it. Unless a file says otherwise, times
are in µs: median of CUDA-event timings, or graph-replay averages where the file says
"graph". Kernel output bits were checked in every case: the invariance, matrix and
directed tests, or the inline `same=` column.

## Starting point

`first_run_870c171_*.txt` hold the first full A800 run with the 4090-final engine
(870c171), copied from that run before its result tree was reset: end-to-end summaries
and kernel benchmarks. PI prefill was 1.36–1.89× FlashInfer's time; on the 4090 it was
about 1.0×. Long-context end to end: 30000/128 c1 was −20.3%, TTFT +41.8%.

## Prefill: what shipped and what did not

Variant labels in the files follow the experimental binding of the time.

`prefill_ablation_flags.txt` and `prefill_mt2_and_qreg.txt` (ring kernel, before
pipelining):
- v1 = 64 rows, 2 stages; v2 = 3 stages; v4/v5 = 128 rows (32-row warps, totals in smem,
  1 CTA/SM) with 2/3 stages; v6/v7 = Q in registers, totals in smem, 2/3 stages.
- Tags are build flags:
  - `nosm`: no softmax;
  - `noissue`: no loads after the prologue;
  - `mmaonly`: both;
  - `nopv` / `noqk`: no P·V or no QK MMAs.
- MMA plus ldmatrix alone (`mmaonly` v2) takes 3.1 ms for hq32/q8192, about FlashInfer's
  whole kernel (3.04 ms). Softmax and loads each add ~25%.
- 32-row warps need the fp32 fold totals in smem, which limits them to 1 CTA/SM; they lose
  (v4/v5).

Shipped in 091612d, in order of the files:

| file | change | hq32/q8192, ms |
|---|---|---|
| `prefill_one_key_loads.txt` | one key per thread in cp.async | 5.46 → 5.28 |
| `prefill_xor_addresses.txt` | precomputed XOR-swizzled ldmatrix addresses; Q in registers with totals in smem now pays off (v6) | 4.87 |
| `prefill_pipelined.txt` | S(t + 1) issued before tile t's softmax and P·V, two K and two V slots, one barrier per tile | 4.47 |
| `prefill_pingpong.txt` | S buffers alternate instead of being copied | 4.35 |

`prefill_final_091612d.txt` is the shipped kernel against FlashInfer: 1.10–1.48× on long
prefills, 0.41–0.99× on short ones.

Not shipped:
- `prefill_diag_mask_branch.txt`: masking only diagonal tiles is slower, because the
  branch splits the scheduling block.
- `prefill_online_step_forms.txt`: packing before rounding, and skipping the rescale when
  every scale is 1 (both bitwise exact), give no gain.
- `prefill_barrier_bound.txt`: removing the barrier entirely would buy only ~6%.
- `prefill_warp_pair.txt`, `attempts_deep_decode_warp_pair.cu`: warp-pair specialisation.
  - Warp A holds Q and runs S and softmax for 32 rows; warp B holds o and runs P·V.
  - 1.2–1.7× slower than the shipped kernel: A carries 32 rows of softmax plus 64 MMAs
    per tile and is the critical path.
  - Two split-group bit failures were left undebugged, since speed rules the design out.

Conclusion: at 16 rows per warp every ldmatrix'd K/V fragment feeds one MMA. The fp32
group totals of the canonical fold keep 32-row warps from two CTAs per SM. sm_89's
half-rate fp32-accumulate MMA hid this; sm_80 does not.

## Prefill split planner (engine 489c9dc)

`prefill_split_planner.txt` (scripts `probe_chunks.py`, `probe_wave.py`, run from
`kernels/`) compares the planner with forced split counts on the shipped kernel.
- Few-wave shapes, where the planner simulates the causal tail's dispatch: the
  simulation was fitted on sm_89, and on the A800 every split it picks is slower than one
  split. Examples: 2048-token chunk with 16 heads, 241.9 vs 205.2 µs; 1024-token chunk with
  32 heads, 163.2 vs 125.7 µs; two 512-token extends over 4096 keys with 32 heads, 772.6 vs
  631.7 µs.
- Short extends over long prefixes (the wave branch): within 6% of the best forced split
  on every shape, so they are left alone.
- Long chunks over long prefixes already run unsplit, because the split workspace would
  exceed the 256 MiB cap.

Engine 489c9dc therefore keeps one split in the few-wave case on sm_80. Splits never
change bits.

## Decode at small batch

- `decode_ablation_flags.txt`: the original kernel, graph-timed.
  - Batch 1 with 512 keys takes 22 µs against FlashInfer's 11.3.
  - Without QK, the cp.async wait and the barrier, a tile still costs 0.63 µs: the serial
    online-step chain of one warp per SM sub-partition.
- `decode_exchange_free*.txt`: every warp computes the full tile S, so there is no smem
  exchange.
  - Batch ≤ 8: 20–30% faster. Batch ≥ 16: 0–12% slower (4× duplicated QK).
  - End to end against the original kernel: c8 +1%, c16 −2.5%, c32 −5%.
- `decode_three_phase.txt`, `attempt_decode_three_phase.cu`: warps compute S of different
  tiles; exact prefix maxima give each tile's scale; then a serial d/o chain. 82 KB smem,
  2 CTAs/SM, so batch ≥ 16 is 10–30% slower.
- `decode_deep_ring*.txt`: the same design with all cache rows resolved up front, a
  16-slot ring and a persistent grid, used only when requests × kv heads ≤ SMs.
  - Batch 1 / 640 keys: 25.2 → 16.8 µs (FlashInfer 11.4). Batch 8 / 640 keys:
    34.3 → 24.2 µs.
  - End to end: c8 −11.4% → −9.8%, but long contexts lose 2–3 points. At long kv the items
    need several waves at 1 CTA/SM, while the original kernel overlaps ~8 CTAs per SM.
  - The right gate is the actual kv length per step, which a single CUDA graph per batch
    bucket cannot see, so this was reverted.

## GEMM, M ≤ 32

- `gemm_17b_candidates.txt`: the 1.7B down projection (N = 2048, K = 6144).
  - M = 8–32: the best canonical kernel takes 28.5 µs against cuBLAS 22–24 µs, which uses
    split-K. A reduction split is forbidden by the canonical k16 chain.
  - M = 1: the canonical stream-K ids 41/51 beat cuBLAS (22.5 vs 29.8 µs).
- `gemm_chained_streamk.txt`: cutting a tile's chain into pieces handed from CTA to CTA,
  with an L2 prefetch of each piece's weights while it waits, serialises the MMA chain:
  45.8 µs.
- `gemm_direct_a_fragments.txt`: reading the activation fragments straight from global
  memory instead of staging them is slower on every shape.
- `gemm_segsplit_deeper.txt`: deeper segment-split pipelines (BK 128 × 6, BK 64 × 12) are
  no better on the 0.6B S = 2 shapes.

## End-to-end attribution

- `e2e_component_split_c8_c16.txt`: Qwen3-1.7B, 512/256.

  | | PI attention only | PI GEMM only | both |
  |---|---|---|---|
  | c8 | −6.5% | −4.6% | −11.4% |
  | c16 | −1.8% | −2.6% | −5.3% |

- `nsys_c8_c32.txt`: per-step kernel time of the same runs.
- `e2e_quick_after_prefill.txt`: the long-context gain from the prefill change. 30000/128
  c1 goes from −20.3% to −11.7%, and TTFT from +41.8% to +18.5%.
