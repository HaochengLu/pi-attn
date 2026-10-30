# Closing probes (A800, 2026-10-30)

Run from `kernels/` (usage line in each script). Bits of every variant were checked against the
shipped kernel before timing.

- `acc_probe.py`, `acc_probe.txt`: attention error against float64 at 4096 keys. torch SDPA on the
  tests' 3-d inputs runs its math backend (the fused backends refuse them), which keeps P in fp32,
  so its error is close to bf16 rounding of the exact result; FlashInfer's single prefill, which
  rounds P to bf16 as PI-Attn does, is the reference the tests now use (PI 8.11e-5 / 7.48e-4 mean
  error vs FlashInfer 8.32e-5 / 7.72e-4, for query scale 1 / 6).
- `probe_prefill_warps.py`, `probe_prefill_warps.txt`: prefill with 4 or 8 warps per CTA, Q in
  smem or registers, in the engine regime (CUDA graph over a per-layer KV ring). 8 warps with Q in
  registers (one CTA per SM) is 5-8% faster on long and batched prefills and up to 12% slower on
  short extends and small grids; shipped as `prefill_warps`: 8 warps when the 128-row CTAs fill two
  waves.
- `gemm_cfg.cu`, `probe_gemm_cfg.py`, `probe_gemm_cfg.txt`: ten CUTLASS configurations at
  M = 128-1024 against the best id the tuner already had. The 64-deep k-stage tiles win at
  M = 128-512 on many projections (8B down M = 128: 1.25x to 0.96x cuBLAS; 4B down M = 256: 1.32x
  to 1.04x); shipped as ids 67-70 and their stream-K forms 71/72.
