# Exact GDN decode: the trainer's chunk program as the decode kernel

Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) are trained with Gated DeltaNet's *chunked* forward (fla
`chunk_gated_delta_rule`, 64-token chunks) but served with a *recurrent* decode kernel, and speculative decoding
verifies drafts with yet another multi-token path. These are different floating-point programs, so the sampler's
log-probabilities are neither the trainer's nor independent of whether speculation is on. This directory makes GDN
decode execute the trainer's own program: every decoded token, one per step or verified in a speculative window,
gets, bitwise, the row fla 0.5.0's chunked forward produces for it, with the trainer's kernel unchanged and with
rejected drafts rolled back for free.

All numbers are from the A800-80GB box (sm_80, driver 535), Triton 3.7.1, torch 2.13 (engine venv), fla 0.5.0
(`env/fla_pkgs`, see `VERSIONS.txt`); logs in `results/a800/gdn/`.

## Results

**1. The premise holds for stock fla** (`experiments/e1_premise.py`, `gdn_e1.log`; synthetic inputs, T=1024,
16 heads, K=V=128). fla's chunk program is row-causal and resumable bitwise: truncating the sequence, resuming from
the fp32 state at any 64-token boundary, replaying a partial chunk token by token (including the state at the next
boundary), and varlen packing all reproduce the full forward's rows exactly. fla's own recurrent kernel
(`fused_recurrent_gated_delta_rule`) differs on 100% of rows (relative L2 4.7e-3).

**2. On real models GDN is the whole prefill/decode gap once dense ops are row-invariant**
(`experiments/e2_qwen35.py`; `e2_*_{rowinv,default}.log`; teacher-forced, 1024 tokens of the model README, 300-token
prefill, 724 scored positions). With every Linear, RMSNorm and full-attention layer made row-invariant (a stand-in for
PI-Attn's kernels, `ROWINV=1`):

| | Qwen3.5-0.8B | Qwen3.5-4B |
|---|---|---|
| per GDN layer, recurrent decode vs chunk: rows equal | 0% (all 18 layers); mean rel. err per layer 1.8e-3..3.3e-3 | 0% (all 24 layers); 2.1e-3..5.1e-3 |
| per GDN layer, chunk replay vs chunk: rows equal | 100% (18/18 layers) | 100% (24/24 layers) |
| end to end, recurrent decode: mean KL / argmax flips vs trainer | 7.6e-4 / 1.24% | 6.6e-4 / 0.83% |
| end to end, chunk replay: logits equal to trainer | 100% (KL 0) | 100% (KL 0) |

Without row-invariant dense ops (`*_default.log`) the replay does not reduce the KL (7.0e-4 / 6.1e-4): GEMM row count
and attention dominate. GDN exactness is worth having on top of a row-invariant engine, where it is the entire
remaining gap. The causal conv1d needs no change: HF's update path is already bitwise to the full conv.

**3. The exact decode kernel** (`exact_decode.py`).

- Bitwise (`test_exact.py`, `test_exact.log`): every case passes, 2936 decode rows and 42 boundary states each:
  kernel configs 2:4:64 and 2:4:128 (num_warps of K1:K2, V block), head layouts H=HV=32, H=HV=16, H=16/HV=32, K=V=64,
  fp32 beta, and the stateless reference implementation `exact_decode_ref.py`.
- End to end (`experiments/e9_qwen35_exact.py`, `e9_qwen35_exact.log`): Qwen3.5-0.8B and 4B with row-invariant dense
  ops and `exact_decode` in every GDN layer produce logits equal to the trainer forward on all 724 scored positions.
- Cost per layer per step, CUDA graphs, Qwen3.5-4B shape, config 2:4:128 (`bench_decode.py`, `bench_decode.log`;
  baseline: fla 0.5.0 `fused_recurrent_gated_delta_rule`, fla's generic recurrent kernel, state read and written once):

  | batch | 1 | 8 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | fla fused_recurrent (us) | 6.3 | 30.2 | 95.4 | 172.3 | 337.1 | 666.1 |
  | exact_decode (us) | 20.7 | 30.0 | 98.7 | 140.4 | 282.1 | 534.0 |
  | ratio | 3.29x | 0.99x | 1.03x | 0.81x | 0.84x | 0.80x |

  Within noise at batch 8-32 (B=32 ranged 86-99 us across runs, `e8_cost4*.log`), 16-20% faster at 64-256.
  Against the production kernel, vLLM 0.17.1's GDN decode as its Qwen3.5 model calls it (fused_recurrent with the
  state updated in place in its pool; `bench_vllm_gdn.py`, `bench_vllm_gdn.log`, run in `env/venv-vllm17`): 11.3,
  22.7, 96.9, 175.5, 333.1, 653.8 us, so exact decode is 1.83x, 1.32x, 1.02x, 0.80x, 0.85x, 0.82x of it. Per step
  the exact kernel reads the bf16 copy of the state (32 KB/head) plus the chunk's cached rows and writes the state
  once per 64 tokens; the recurrence reads and writes 128 KB/head of fp32 every token. At batch 1 both are
  latency-bound and the exact path is two dependent kernels with barriers.
- Memory per slot per layer (H=HV=32, K=V=128): 6.75 MiB (fp32 boundary state 2, its bf16 copy 1, chunk inputs and
  l2norm(q) rows 1.5, (I+A)^-1 rows 0.75, u/w/v_new rows 1.5) versus 2 MiB for the recurrent state: 3.4x. Against a
  long context's KV cache this is small (Qwen3.5-4B: 4 KV heads x 256 dims x 32K tokens, K and V in bf16, is 128 MiB
  per attention layer); it
  matters for short-context, high-batch serving.
- How we got there (`archive/gdn_iterations/`, `e3_cost.log`, `e6_*`, `e7_*`, `e8_*`): calling fla's chunk on the
  partial chunk costs 4.5-8.4x the recurrent kernel's GPU time, plus host preparation (~900 us of wall time at
  B <= 32); a stateless fused kernel (`exact_decode_ref.py`) 3.7x at B=256 and 10.5x at B=1 (`e6_cost_v2.log`;
  `e6_cost_v1.log` used a baseline inflated by a state copy-back); row-incremental in one kernel 1.11x at B=256 but
  7.75x at B=1; the same with in-kernel l2norm and a V split 1.48x at B=256, 2.74x at B=1 (`e7_cost3.log`); split
  into a kernel without the state and one with it as above.

**4. Speculative verification is invisible** (`exact_decode.step` / `commit`).

A step can verify a window of up to 16 tokens per slot (the current token plus drafts), never past the slot's next
64-token boundary. Its rows are the trainer's rows whatever the window size, because the window just computes more rows
of the same chunk program. Rejection costs nothing: rows past the accepted ones are garbage by construction (every
consumer masks them) and are overwritten by the next window, and a completed chunk is folded into the state only after
its 64th row is accepted (lazily, by a small kernel at the start of the next step). No per-draft state checkpoints.

- Bitwise with random acceptance and wrong drafts (`test_exact.log`, "spec window"): windows of 4, 1-4 tokens
  accepted per slot per step, rejected positions fed with another sequence's tokens; 2889 accepted rows and 42
  boundary states equal to the trainer's, for H=HV=32 and H=16/HV=32.
- End to end (`experiments/e12_qwen35_spec.py`, `e12_qwen35_spec.log`): greedy generation of 300 tokens after a
  200-token prompt with prompt-lookup drafting (up to 4 drafts) gives the same tokens and bitwise the same fp32
  log-probability for every generated token as one-token decode and as the trainer's teacher-forced forward, on
  Qwen3.5-0.8B (221 of 241 drafts accepted, 3.80 tokens per target forward) and 4B (32 of 126 accepted, so mostly
  rollbacks, 1.12 tokens per forward). The status quo for contrast (`experiments/e12b_hf_default_spec.py`, same log):
  stock HF/fla re-scoring the same tokens in verification windows of 5 instead of one at a time gives bitwise-equal
  log-probabilities for 24/300 (0.8B) and 4/300 (4B) tokens, mean |dlogp| 2.9e-3 / 8.8e-3, max 0.085 / 0.127.
- Cost of a 4-token window per layer against vLLM 0.17.1's speculative path, which writes the state after every
  draft token into its own pool slot (that is its rollback) and starts from the accepted one (`bench_vllm_gdn.log`,
  `bench_decode.log`, config 2:4:128):

  | batch | 1 | 8 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | vLLM 0.17.1 verify, 4 tokens (us) | 32.7 | 69.5 | 248.0 | 457.5 | 901.7 | 1792.4 |
  | exact_decode window of 4 (us) | 55.5 | 78.8 | 211.6 | 358.2 | 706.1 | 1336.4 |
  | ratio | 1.70x | 1.13x | 0.85x | 0.78x | 0.78x | 0.75x |

  State memory per slot and layer for 4 drafts: vLLM keeps 1 + 4 fp32 states (10 MiB), exact_decode 6.75 MiB.
  Against fla's recurrent kernel over the same 4 tokens keeping only the final state (no rollback possible), the
  window is 1.59-1.88x at batch >= 8. Per verified token it costs 0.63x (B=256) to 0.66x (B=8) of exact one-token
  decode.

## What the canonical program is

"The trainer's program" is only a single program once these are fixed, on both sides:

- fla 0.5.0 `chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True` and fla's default `use_exp2` path; chunk 64.
  The gate and beta computation, conv1d and output norm around it are the caller's to match (`exact_decode` takes the
  raw log-gate g and beta as tensors, with beta in the trainer's dtype); fla's `use_gate_in_kernel` path and
  context-parallel prefill are outside the tested contract.
- `FLA_USE_FAST_OPS` unset (it swaps `exp`/`exp2` implementations).
- fla's `l2norm_fwd_kernel` config. Its autotune key includes `NB = cdiv(tokens x heads, 65536)`, so the trainer's
  packed micro-batches and the sampler are tuned separately, and 3 of its 25 configs (BT=8 with 8 or 16 warps and
  one more config) produce different bits (`e4_autotune.log`). In 6 fresh processes the picks differed between NB=1
  and NB=20 yet the outputs agreed (`e4b_l2pick.log`, `e10_pack.log`), so this is a latent risk, not an observed
  mismatch: pin the config. Every other autotuned kernel on the forward path is config-independent (`e4_autotune.log`).
- Triton version and GPU architecture: the kernels reproduce fla's compiled arithmetic (MMA shapes, fp-contraction);
  validated on sm_80 with Triton 3.7.1 only. Hopper's wgmma is a different instruction and is not covered.

## Traps found while reproducing a Triton program bitwise

- **fp-contraction.** fla's `chunk_fwd_kernel_o` computes `b_o * scale + dot(A, v) * scale`; its PTX is
  `fma(scale, dot, b_o * scale)`. The same source in another kernel was contracted differently, which changed a few
  percent of output rows by up to ~1e-4. `exact_decode` writes the fma out.
- **Restructuring that only moves data can change bits.** A variant that kept the token's rows in registers and spliced
  them into the dot tiles instead of reloading them, and one that selected the sub-block gates from a register vector,
  stopped being bitwise; Triton's layout and contraction choices are made for the whole kernel. `test_exact.py` is the
  arbiter after any edit, including ones that look arithmetic-neutral.

## Prior art and what is (not) claimed

Replaying the trainer's chunk program at decode from a 64-token boundary state is published for one token per step:
Together AI's zero-mismatch RL stack does it for Qwen3.5/3.6 GDN (A. Panda, 2026-08-17,
https://kiddyboots216.github.io/mismatch/; exact GDN decode is "the largest single cost": about half of a 39.4%
sampling slowdown), NVIDIA Megatron-Core for Mamba2 (`batch_invariant_decode.py`, which asserts one token per entry,
"no speculative decoding"), and vLLM RFC #55524 for Mamba2 (2-2.6x the recurrent kernel; speculative decoding is
rejected in that mode). Checkpoint + replay with low-precision state reads that is faster than recurrence is also
published, without training exactness: ReplaySSM (arXiv 2610.13778, 1.43-1.64x over tuned vLLM at B=256 on
H100/B300; rollback by moving a ring-buffer pointer, equal "up to floating-point error"), DeltaLog (arXiv 2608.15533).
Production engines roll speculative GDN decode back with a full state snapshot per draft token (SGLang, vLLM,
FlashInfer's MTP kernel).

What this directory adds, on the evidence above: speculative verification for a hybrid model whose accepted tokens are
bitwise the tokens and log-probabilities of one-token decode and of the trainer's stock forward, with rollback that
needs no state snapshots; on A800 its per-layer cost is 0.75-0.85x of vLLM 0.17.1's (inexact) speculative verifier
at batch >= 32, with less state memory, and plain exact decode is 0.80-0.85x of vLLM's decode kernel at batch >= 64.
Both are slower at batch 1-8 (1.1-1.8x).

Not claimed: speed versus ReplaySSM-style or bf16-state kernels, or versus the packed/CUDA GDN decode kernels of vLLM
releases after 0.17.1 (their wheels need CUDA 13, which driver 535 cannot run); anything on Hopper/Blackwell;
serving-engine throughput (not integrated into an engine yet); batch 1-8, where the exact path is slower.
