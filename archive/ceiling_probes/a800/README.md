# A800 ceiling probes (sm_80)

Hardware: A800-SXM4-80GB, 108 SMs, 1.41 GHz max SM clock; full environment in
`results/a800/ENV.txt`. Hardware performance counters are not available in the container
(ERR_NVGPUCTRPERM), so every attribution here comes from timing ablation builds.

The engine commit that ships is 38c32da (091612d, the planner change, the GEMM changes and the decode grid order below). Every attempt below that is not in it was
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

- `prefill_wide_warps.txt`, `attempt_prefill_wide_warps.diff`, `probe_wide.py`: 32-row
  warps on the same 64-row tiles, as FlashInfer's long-prefill kernel. Each V fragment
  then feeds two MMAs, so ldmatrix per MMA falls from 1/2 to 3/8.
  - Q and the fold totals sit in smem (80 KB). Two CTAs per SM, as occupancy confirms,
    give 4 warps per SM.
  - Bits are identical (0 invariance failures), but it runs 1.7–2.4× the shipped
    kernel's time. Likely cause (no counters on this machine): each 32-key online step
    is a serial chain of shuffles and exp2 between the QK and PV MMAs, and with one warp
    per scheduler and two independent row blocks per warp, nothing hides it.
  - 8 warps per SM would need the fold totals in per-CTA global scratch; even linear
    scaling from 4 to 8 warps would leave it at ~1.15× the shipped kernel.

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
- `decode_device_gate_engine.txt`, `attempt_decode_device_gate.diff`: the gate moved to
  the device. Both kernels are captured in each graph; each counts the step's key tiles
  from `kv_lens`, and only one does the work: the deep kernel when the tiles fit one wave
  of full fold groups (SMs × 16), otherwise `decode_kernel`. Bits pass the invariance
  tests with either kernel and with the gate.
  - `decode_deep_gate.txt` (`probe_gate.py`) repeats one cache that fits in L2. There the
    deep kernel looks 20–40% faster. `probe_gate_dram.txt` (`probe_gate_dram.py`) uses 28
    caches per replay, as a 28-layer step does. Batch 8 with 640 keys: 38.5 → 31.9 µs.
  - In the engine (nsys, Qwen3-1.7B, 8 × 512-token prompts) the deep kernel saves only
    ~2.8 µs per layer. The second launch costs ~2.4 µs per layer, and step wall time
    goes from 4593 to 4646 µs. Not shipped.
- `decode_grid_order.txt`, `probe_ring.py`, `probe_ring_old.py`: load depth and grid
  order of the shipped kernel, all in the engine regime (28 caches per graph replay).
  - Ring of K/V slots (`attempt_decode_ring.diff`, not shipped):
    - Each tile is issued RING − 1 tiles ahead, against one tile today.
    - At a capture grid for 4096 keys, 3 slots cut batch 8 / 1024 keys from 45.6 to 37.0 µs.
    - 4 slots (67 KB, 2 CTAs/SM) lose at long kv.
    - With the engine's capture grids (8192–32768 keys, most CTAs returning at once),
      the gain vanishes. The deeper ring's smem allows fewer CTAs per SM, and the
      empty CTAs then drain more slowly.
  - Shipped (engine 38c32da): the grid order. With the group as the slowest grid
    index, every working CTA is dispatched before the empty ones, and the kv heads of a
    token, adjacent in the cache, are read together.
    - Batch 8 / 1024 keys at a 32768-key grid: 42.2 → 35.0 µs. Batch 16 / 512 keys:
      40.2 → 33.7 µs.
    - Single-request launches measured faster with the group fastest (batch 1 / 512 keys at
      a 32768-key grid: 25.1 vs 28.0 µs), so `decode_grid` keeps that order for them.
    - Engine (nsys, Qwen3-1.7B c8): decode 1001 → 842 µs per step.
    - End to end (quick_e2e, 1.7B): c8 −9.5% → −6.6%, c16 −6.6% → −3.5%, c32 → −0.7%;
      0.6B c8 +2.7% → +6.6%.

## GEMM, M ≤ 32

Shipped (engine 42a5beb):
- `gemm_streamk_segments.txt`: the warp stream-K kernel (ids 40–43, 51, 53) now also
  runs S = 2 shapes. Each (n8 tile, segment) chain is cut between workers as before. A
  finished chain's partial is folded as in `gemm_balanced_kernel`: an arrival count per
  tile, and the last to arrive folds in ascending segment order.
  - 0.6B o at M = 1: 8.41 → 7.77 µs; 0.6B down: 11.67 → 9.07 µs (cuBLAS 6.22 / 8.06).
- `gemm_streamk_pairs.txt`, `probe_tn.py`: ids 60–62, chains of two adjacent n8 tiles
  that share each X fragment. At M = 8 each worker otherwise re-reads as many X bytes
  from L2 as weight bytes.
  - 1.7B down at M = 8: 27.97 → 24.59 µs (cuBLAS 21.2); 0.6B down: 11.68 → 11.01 µs.
  - One-warp workers and 16-row X variants lose, so they are not shipped.

Earlier probes:

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
- `gemm_segment_chain.txt`, `attempt_gemm_segment_chain.diff`, `probe_chain.py`: 0.6B o
  and down at M = 1 run 1.3–1.5× cuBLAS's GEMV on this GPU. That accounts for the whole
  0.6B batch-1 loss (GEMM-only ablation −8.4%).
  - The attempt splits each segment's chain into 256- or 128-column pieces, one warp per
    piece. Each warp loads its fragments straight into registers, so the whole tile is in
    flight at once, and the accumulator passes from warp to warp in k order.
  - Bit-identical, but no faster: 0.6B o 10.38 µs against 9.56 for the best shipped
    kernel, and down 14.32 against 11.69. One warp's instruction stream is therefore not
    what bounds these kernels.
  - The segment rule itself is fixed across GPUs. Changing it for the A800 would give up
    the cross-GPU identity in `results/a800/xgpu/`.

## End-to-end attribution

- `e2e_component_split_c8_c16.txt`: Qwen3-1.7B, 512/256.

  | | PI attention only | PI GEMM only | both |
  |---|---|---|---|
  | c8 | −6.5% | −4.6% | −11.4% |
  | c16 | −1.8% | −2.6% | −5.3% |

- `nsys_c8_c32.txt`: per-step kernel time of the same runs.
- `e2e_quick_after_prefill.txt`: the long-context gain from the prefill change. 30000/128
  c1 goes from −20.3% to −11.7%, and TTFT from +41.8% to +18.5%.
- `nsys_c8_38c32da.txt`: the same breakdown for engine 38c32da (Qwen3-1.7B, 8 × 512-token
  prompts; `scripts/nsys_decode.sh`, `scripts/nsys_step.py`, idle gaps from `nsys_gaps.py`).
  - Decode attention: 884 µs per step against FlashInfer's 548 + 105 (merge).
  - o and down projections: about 100 µs per step above cuBLAS with split-K. The other
    kernels are the engine's own in both modes.
  - Idle time between kernels: 228–282 µs per step against FlashInfer's 320–344.

## Where the A800 stops

At engine 38c32da the 1.7B c8 kernel gap is about 330 µs per step. It splits into two parts.

- Decode attention, about 8 µs per layer. Each working CTA runs group 0's 16 tiles in
  order at about 1.1–1.4 µs per tile, plus the fold of two groups. That costs the same at
  batch 1, where bandwidth is not under pressure.
  - Under the contract the floor is about 26 µs per layer; the shipped kernel takes 31.5.
  - Deeper loads (the ring), other decode designs (three-phase, deep ring, warp pair) and
    the device gate were measured above and do not move it in the engine.
- o and down projections, about 3.5 µs per layer. This is cuBLAS's split-K, which the
  per-element chain forbids.

Fable's third review checked the scheduling-only levers that remain, against the traces
and the code, and found none with a credible engine-level gain:
- the prefill grid and wave quantisation;
- the GEMM grids and the merge kernels.
