# gemm_bw_probe: is the batch-1 projection gap a kernel problem or a canonical-program problem?

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
