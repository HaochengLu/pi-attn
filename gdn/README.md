# Exact GDN decode: the trainer's chunk program as the decode kernel

Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) are trained with Gated DeltaNet's *chunked* forward (fla
`chunk_gated_delta_rule`, 64-token chunks) but served with a *recurrent* decode kernel. The two are different
floating-point programs, so the sampler's log-probabilities are not the trainer's. This directory makes GDN decode
execute the trainer's own program: every decode step returns, bitwise, the row that fla 0.5.0's chunked forward
produces for that token, with the trainer's kernel unchanged.

All numbers below are from the A800-80GB box (sm_80, driver 535), Triton 3.7.1, torch 2.13 (engine venv), fla 0.5.0
(`env/fla_pkgs`, see `VERSIONS.txt`); logs in `results/a800/gdn/`.

## Results

**1. The premise holds for stock fla** (`experiments/e1_premise.py`, `gdn_e1.log`; synthetic inputs, T=1024,
16 heads, K=V=128). fla's chunk program is row-causal and resumable bitwise: truncating the sequence, resuming from
the fp32 state at any 64-token boundary, replaying a partial chunk token by token (including the state at the next
boundary), and varlen packing all reproduce the full forward's rows exactly. fla's own recurrent kernel
(`fused_recurrent_gated_delta_rule`) differs on 100% of rows (relative L2 4.7e-3).

**2. On real models GDN is the whole prefill/decode gap** (`experiments/e2_qwen35.py`; `e2_*_{rowinv,default}.log`;
teacher-forced, 1024 tokens of the model README, 300-token prefill, 724 scored positions).
With every Linear, RMSNorm and full-attention layer made row-invariant (a stand-in for PI-Attn's kernels, `ROWINV=1`):

| | Qwen3.5-0.8B | Qwen3.5-4B |
|---|---|---|
| per GDN layer, recurrent decode vs chunk: rows equal | 0% (all 18 layers); mean rel. err per layer 1.8e-3..3.3e-3 | 0% (all 24 layers); 2.1e-3..5.1e-3 |
| per GDN layer, chunk replay vs chunk: rows equal | 100% (18/18 layers) | 100% (24/24 layers) |
| end to end, recurrent decode: mean KL / argmax flips vs trainer | 7.6e-4 / 1.24% | 6.6e-4 / 0.83% |
| end to end, chunk replay: logits equal to trainer | 100% (KL 0) | 100% (KL 0) |

Without the row-invariant dense ops (`*_default.log`) the replay does not help (KL 7.0e-4 / 6.1e-4): GEMM row
count and attention dominate. GDN exactness is only worth having on top of a row-invariant engine, and with one it
is the entire remaining gap. The causal conv1d needs no change: HF's update path is already bitwise to the full conv.

**3. The exact decode kernel** (`exact_decode.py`; checked by `test_exact.py`, run in `bench_decode.py`, end to end in
`experiments/e9_qwen35_exact.py`).

- Bitwise: `test_exact.py` passes for H=HV=32, H=HV=16 and H=16/HV=32 (2936 decode rows and 42 boundary states each,
  8 slots at different chunk rows, both implementations).
- End to end (`e9_qwen35_exact.log`): Qwen3.5-0.8B and 4B with row-invariant dense ops and `exact_decode` in every GDN
  layer produce logits equal to the trainer forward on all 724 scored positions.
- Cost per layer per step, CUDA graphs, Qwen3.5-4B shape (`bench_decode.log`; baseline = fla 0.5.0 `fused_recurrent`,
  state read and written once):

  | batch | 1 | 8 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | fla fused_recurrent (us) | 6.4 | 30.9 | 95.4 | 181.6 | 336.8 | 666.2 |
  | exact_decode (us) | 19.4 | 30.1 | 96.7 | 163.8 | 293.5 | 518.6 |
  | ratio | 3.02x | 0.97x | 1.01x | 0.90x | 0.87x | 0.78x |

  B=32 varies run to run between 86 and 97 us (`e8_cost4*.log`). At batch >= 8 the exact kernel is as fast or faster
  than the inexact one: per step it reads the bf16 copy of the state (32 KB/head) plus the chunk's cached rows and
  writes nothing to the state except at the fold, while the recurrence reads and writes 128 KB/head of fp32. At
  batch 1 both are latency-bound and the exact path is two dependent kernels with three barriers.
- Memory per slot per layer (H=HV=32, K=V=128): 6.25 MiB (fp32 boundary state 2, its bf16 copy 1, chunk inputs 1,
  (I+A)^-1 rows 0.75, u/w/v_new rows 1.5) versus 2 MiB for the recurrent state: 3.1x.
- How we got there (`archive/gdn_iterations/`, `e3_cost.log`, `e6_*`, `e7_*`, `e8_*`): calling fla's chunk on the
  partial chunk costs 4.5-8.4x the recurrent kernel's GPU time plus ~900 us of host prep; a stateless fused kernel
  (`exact_decode_ref.py`) 3.7x at B=256 and 10.5x at B=1 (`e6_cost_v2.log`; `e6_cost_v1.log` used a baseline
  inflated by a state copy-back); row-incremental in one kernel 1.11x at B=256 but 7.75x at B=1 (255 registers);
  the same with in-kernel l2norm and V split 1.48x at B=256, 2.74x at B=1 (`e7_cost3.log`); split into K1 (no
  state) + K2 (state) as above.

## What the canonical program is

"The trainer's program" is only a single program once these are fixed, on both sides:

- fla 0.5.0 `chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True` and fla's default `use_exp2` path; chunk 64.
- `FLA_USE_FAST_OPS` unset (it swaps `exp`/`exp2` implementations).
- fla's `l2norm_fwd_kernel` config. Its autotune key includes `NB = cdiv(tokens x heads, 65536)`, so the trainer's
  packed micro-batches and the sampler are tuned separately, and 3 of its 25 configs (BT=8 with 8 or 16 warps and
  one more config) produce different bits (`e4_autotune.log`). In 6 fresh processes the picks differed between NB=1 and
  NB=20 yet the outputs agreed (`e4b_l2pick.log`, `e10_pack.log`), so this is a latent risk, not an observed
  mismatch: pin the config. Every other autotuned kernel on the forward path is config-independent (`e4_autotune.log`).
- Triton version and GPU architecture: the kernels reproduce fla's compiled arithmetic (MMA shapes, fp-contraction);
  validated on sm_80 with Triton 3.7.1 only. Hopper's wgmma is a different instruction.

## Traps found while reproducing a Triton program bitwise

- **fp-contraction.** fla's `chunk_fwd_kernel_o` computes `b_o * scale + dot(A, v) * scale`; its PTX is
  `fma(scale, dot, b_o * scale)`. The same source in another kernel was contracted differently (4.9% of rows off by
  up to 1.2e-4). `exact_decode` writes the fma out.
- **Dot operands from registers.** An operand built in registers through a cast or a splice (`tl.where`, `tl.join`)
  is compiled to different numerics than the same bf16 values loaded from memory (a failed v5 kept the token's rows
  in registers to drop barriers: 6-20% of rows off). `exact_decode` passes such rows through memory.
- **Layout is global.** Selecting sub-block gates from a register vector instead of reloading them changed the
  reduction layout of the forward substitution elsewhere in the kernel. Edits that are "pure data movement" can
  change bits; `test_exact.py` is the arbiter.

## Prior art and what is (not) claimed

Replaying the trainer's chunk program at decode from a 64-token boundary state is published: Together AI's
zero-mismatch RL stack does it for Qwen3.5/3.6 GDN (A. Panda, 2026-08-17, https://kiddyboots216.github.io/mismatch/;
exact GDN decode is "the largest single cost": half of a 39.4% sampling slowdown), NVIDIA Megatron-Core does it for
Mamba2 (`batch_invariant_decode.py`), and vLLM RFC #55524 for Mamba2 (2-2.6x the recurrent kernel). Faster-than-
recurrent decode by checkpoint + replay with low-precision state reads is also published, without training
exactness: ReplaySSM (arXiv 2610.13778, 1.43-1.64x over tuned vLLM at B=256 on H100/B300), DeltaLog (arXiv 2608.15533).

What this directory adds, on the evidence above: a training-exact GDN decode, against an unmodified stock fla trainer
kernel, whose per-layer cost is at or below fla's fp32 recurrent kernel for batch >= 8 on A800.
Not claimed: speed versus ReplaySSM-style or bf16-state kernels, or versus vLLM's packed/CUDA GDN decode kernels
(not yet measured here); anything on Hopper/Blackwell; serving-engine throughput (not integrated yet); batch 1.
