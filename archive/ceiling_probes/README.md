# Ceiling probes: is a remaining gap a kernel problem or a canonical-program problem?

All numbers: RTX 4090 D, M=1 unless stated, weights cold (each launch in a burst of 32
streams a different copy from a >= 600 MB pool).

1. `bench_bw.py` / `bw_probe.cu`: pure weight streaming, no MMA. A linear ldg.128 stream
   reaches 946-994 GB/s; the PI access pattern (CTA owns contiguous n8 tiles, K in cp.async
   stages) reaches 0.97-1.02x cuBLAS on every shape except those whose n8 tile count
   divides badly over 114 SMs (8B down: 512 tiles -> 5 vs 4.49 per SM, 1.10x).
   `out.txt`.
2. `bench_pi.py`: the engine's kernels in the same harness. PI / cuBLAS at M=1:
   1.02-1.04 everywhere, 1.13 on 8B down. `out_pi.txt`.
3. `seg_sweep.py`: generalised shape-only K-segment rule ("fewest segments giving >= T
   (tile, segment) items", T in {256 (= the current rule on Qwen3 shapes), 1024, 2048}),
   best kernel per M. Only 8B down gains at decode (118.5 -> 112.1 us, M=1); every newly
   segmented shape loses 1-5% at prefill M (8B o 949.6 -> 993.0 us, 4B down 1473 -> 1528 us
   at M=4096) and the others are neutral at decode. Because the canonical split must be
   the same for every M, this trade cannot be avoided per batch size; the engine keeps the
   current rule. `out_seg.txt`.

## Prefill probes (same session)

4. 8-warp (128-row) prefill CTAs, K/V tiles shared by twice the rows, one CTA per SM
   (kernel templated on warps; canonical sequence unchanged, test_invariance passed).
   Slower on every shape in `kernels/bench_prefill.py` (e.g. hq32 b4 2K+2K 3603 -> 4170 us,
   hq32 q128/16K 391 -> 450 us): with one CTA per SM nothing hides the per-tile barriers.
   Reverted.
5. `prof_pf.py`: hq32 q128 over a 16K prefix spends 321 us in prefill_kernel (FlashInfer:
   311 us in total) and 61 us in the merge, which reads the per-group fp32 partials that
   the canonical fold needs (69 MB). The b4 2K+2K gap (1.10) is the attention kernel alone
   (16-row warps; 32-row warps would need the second online state outside registers,
   measured 8-10% slower earlier).

## Stream-K and mid-M GEMM probes

6. `streamk_probe.cu` / `bench_streamk.py`: deterministic stream-K for M <= 8 (accumulator
   handoff between warps, bitwise the uncut chain; 0 bitwise failures). 8B down at M=1:
   118.5 -> 113.2 us (cuBLAS 108.1); 8B down M=8 beats cuBLAS. Shipped as kernel ids
   40..43. `out_streamk.txt`.
7. `cutlass_probe.cu` / `bench_cutlass.py`: smaller CUTLASS tiles for M=64..512 on narrow-N
   projections (configs with 16-wide warp N tiles do not compile: epilogue thread map).
   32x64x64 (4 warps of 16x32) and 64x64x64 fix M=64..128 (1.7B o M=128: 2.11 -> 1.01x
   cuBLAS) but not M >= 256, where 64x64 tiles fill 114 SMs in 1.12 waves. `out_cutlass.txt`.
8. Stream-K over CUTLASS tiles (engine ids 44..47; `gemm_cutlass.cuh` StreamKKernel): when
   tiles >= SMs every CTA range covers >= 1 tile, so the handoff never serialises a chain.
   `bench_midm.py`: old kernel set vs with ids 14/15/44..47, cold bursts. Unsegmented
   shapes go from up to 1.38x (1.7B down M=64), 1.36x (1.7B down M=256), 1.21x (4B down
   M=256), 1.18x (8B o M=128) to 0.83-1.02x at M=16..1024, except 8B down M=128 (1.06x) and
   M=512 (1.09x). At large M the stream-K ranges desynchronise the k positions of CTAs whose
   tiles share weight columns, so those columns no longer meet in L2 (8B down M=1024: k44
   2405 us vs k12 804 us); the tuner then keeps the plain tiles. Segmented shapes (0.6B
   o/down, S = 2) cannot use stream-K and stay 1.10-1.27x at M=256..1024. `out_midm.txt`.
   `occ_probe.cu` / `occ.py`: shared memory and residency of the stream-K tile configs
   (32x64x64 with 5 stages is 60 KB, one CTA per SM; 4 stages fits two).
9. `segpar/`: segment-parallel CUTLASS tiles for segmented shapes (engine ids 48, 49;
   `gemm_cutlass.cuh` SegParallelKernel). One CTA per (tile, segment); the last CTA of a
   tile to arrive folds the S fp32 fragments into +0 in ascending segment order, the
   SegmentedKernel sequence, so the bits are unchanged (test_gemm.py: 0 failures).
   `segpar_quick.py` / `out_sweep.txt`, cold bursts, x cuBLAS, old best -> new best:
   0.6B down M=32 0.94 -> 0.84, M=64 1.24 -> 0.87, M=128 1.15 -> 1.10, M=512 1.28 -> 1.19;
   0.6B o M=64 11.50 -> 8.52 us. No gain at M=256 and M=1024, where one tile per SM is
   already compute-bound (hot and cold weights time the same) and the extra CTAs add a
   second wave. A 64x128 variant never won and was dropped.
   The first version was up to 1.5x slower than this. `segpar_probe.cu` / `out_probe.txt`: its
   arrival flag was a static `__shared__ bool`, which the compiler places in front of the
   dynamic shared buffer, so CUTLASS's swizzled tiles were no longer 128-byte aligned. On a
   plain S = 1 shape the same mainloop and epilogue took 13.35 us against 8.63 us for
   CUTLASS's own kernel; with the flag broadcast by `__syncthreads_or` it takes 8.80 us.
   None of the other CUTLASS-based kernels declares static shared memory.
10. `occ2/`: 8-warp and two-CTAs-per-SM CUTLASS tiles at M=128..1024, each as a plain or
    segment-sequential tile, segment-parallel, and stream-K (`occ_probe2.cu`,
    `bench_occ2.py`, `out_occ2.txt`; 0 bitwise failures). Two configs close gaps:
    64x64x64 with 3 stages (48 KB, two CTAs per SM) as segment-parallel takes 0.6B down
    from 1.19x to 1.06x cuBLAS at M=512 and from 1.15x to 1.02x at M=1024 (o: 1.20x ->
    1.02x, 1.11x -> 1.00x), and is no slower than the 4-stage version at M=128, so it
    replaced engine id 49's config; 128x64x64 with 3 stages as stream-K takes 8B down at
    M=512 from 1.10x to 0.99x (435 -> 393 us) and became engine id 50. Unchanged: 1.7B
    down M=128 (1.09x) and 8B down M=128 (1.05x). An 8-warp 64x128x32 stream-K tile gains
    up to 5% only where the engine is already below cuBLAS and was not added.

## Second-round probes (after the full regeneration of 2026-10-22)

11. `decode_ring/`: decode attention with a K/V prefetch ring of 2, 3 or 4 (K, V) tile
    pairs loaded ahead (`decode_kernel<kStages>`; the engine's kernel holds one K and one
    V tile). Only the copy schedule changes, never the arithmetic: every ring depth is
    bitwise identical to the engine on all 66 configurations. None is faster: up to 1K keys
    at batch 1-4 every depth is within noise of the engine, and wherever the grid is larger
    (batch >= 16, or 4K+ keys at batch 4 and 16K at batch 1) deeper rings lower occupancy
    and lose, up to 1.9x hot (hq16 b128 kv64: 17.3 -> 32.3 us) and 1.26x cold. Batch-1
    decode attention is bandwidth- or latency-bound, not copy-bound: at 16K keys a layer
    reads 67 MB in 75.6 us cold (0.89 TB/s). Not adopted. `out.txt`.
12. `streamk_ring/`: stream-K for M <= 8 (ids 40..43) with one continuous cp.async stream
    through all of a worker's pieces (handed-off prefix, whole tiles, continuation) instead
    of filling and draining the ring per piece, and the A rows >= M zeroed once instead of
    zero-filled per stage (`gemm_inv.cuh`, same arithmetic; test_gemm.py passed). Slower:
    8B down M=1 113.5 -> 117.6 us and M=8 117.3 -> 120.7 us (k40); 1.7B down and 4B o lose
    their stream-K winners. Not adopted. `gemm_burst.py 1 8 16` in one session: `out_engine.txt`
    (engine kernels), `out_ring.txt` (this variant).
13. `tuner_edges/`: the tuner times kernels at one M per bucket and runs that choice for
    every M in the bucket. `gemm_grid.py` times every candidate at 42 M values in 17..2048
    on the 16 projection shapes (`grid.json`) and scores bucket policies (`out.txt`). With
    power-of-two buckets, the M that prefill graphs are captured at (192, 320, 384, 448,
    640, ...) ran a tile tuned for the next power of two: 2.7% slower than the best tile on
    average and up to 55% (0.6B down at M=192: k15 23.5 us vs k49 15.1 us). Choosing per
    bucket the tile with the best worst case over its two ends halves the worst loss but
    keeps a 2.2% mean. The engine now uses the graph capture sizes as buckets, so every
    graph runs the tile timed at its own M (loss 0 at those sizes; 1.3% mean over all grid
    M, which only eager forwards reach, against 3.2% before).
