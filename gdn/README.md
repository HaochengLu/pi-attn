# Exact GDN decode: the trainer's chunk program as the decode kernel

Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) are trained with Gated DeltaNet's *chunked* forward (fla
`chunk_gated_delta_rule`, 64-token chunks) but served with a *recurrent* decode kernel, and speculative decoding
verifies drafts with yet another multi-token path. These are different floating-point programs, so the sampler's
log-probabilities are neither the trainer's nor independent of whether speculation is on. This directory makes GDN
decode execute the trainer's own program: every decoded token, one per step or verified in a speculative window,
gets, bitwise, the row fla 0.5.0's chunked forward produces for it, with the trainer's kernel unchanged and with
rejected drafts rolled back for free. In vLLM 0.17.1 (`vllm/exact_gdn.py`) this makes Qwen3.5's decode bitwise its
own prefill, with n-gram speculative decoding on or off, at 0.99x stock throughput (result 5).

Numbers are from the A800-80GB box (sm_80, driver 535), Triton 3.7.1, torch 2.13 (engine venv), fla 0.5.0
(`env/fla_pkgs`, see `VERSIONS.txt`), logs in `results/a800/gdn/`, unless marked RTX 4090 D (sm_89, same software,
logs in `results/gdn/`).

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
  fp32 beta, and the stateless reference implementation `exact_decode_ref.py`. The same suite passes on the RTX 4090 D
  (`results/gdn/test_exact.log`) and under Triton 3.6.0 in the vLLM venv (`test_exact_triton360.log`, both GPUs), the
  latter only after matching fla's version-dependent output epilogue, and with fla's odd autotune configs removed
  (`pin_fla_configs`; see "What the canonical program is").
- End to end (`experiments/e9_qwen35_exact.py`, `e9_qwen35_exact.log`): Qwen3.5-0.8B and 4B with row-invariant dense
  ops and `exact_decode` in every GDN layer produce logits equal to the trainer forward on all 724 scored positions;
  also on the RTX 4090 D (`results/gdn/e9_qwen35_exact.log`).
- Cost per layer per step, CUDA graphs, Qwen3.5-4B shape, config 2:4:128 (`bench_decode.py`, `bench_decode.log`;
  baseline: fla 0.5.0 `fused_recurrent_gated_delta_rule`, fla's generic recurrent kernel, state read and written once):

  | batch | 1 | 8 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | fla fused_recurrent (us) | 6.4 | 30.7 | 95.7 | 172.3 | 336.9 | 666.1 |
  | exact_decode (us) | 22.1 | 30.8 | 90.1 | 143.6 | 288.9 | 545.4 |
  | ratio | 3.45x | 1.00x | 0.94x | 0.83x | 0.86x | 0.82x |

  Equal at batch 8, within noise at 32 (B=32 ranged 86-99 us across runs, `e8_cost4*.log`), 14-18% faster at 64-256.
  Against the production kernel, vLLM 0.17.1's GDN decode as its Qwen3.5 model calls it (fused_recurrent with the
  state updated in place in its pool; `bench_vllm_gdn.py`, `bench_vllm_gdn.log`, run in `env/venv-vllm17`): 11.3,
  22.7, 96.9, 175.5, 333.1, 653.8 us, so exact decode is 1.96x, 1.36x, 0.93x, 0.82x, 0.87x, 0.83x of it. On the
  RTX 4090 D (`results/gdn/bench_decode.log`) fla's recurrent kernel scales worse and exact decode is 0.32-0.62x of it
  at batch >= 32. Per step
  the exact kernel reads the bf16 copy of the state (32 KB/head) plus the chunk's cached rows and writes the state
  once per 64 tokens; the recurrence reads and writes 128 KB/head of fp32 every token. At batch 1 both are
  latency-bound and the exact path is two dependent kernels with barriers.
- Memory per slot per layer (H=HV=32, K=V=128): 6.75 MiB (fp32 boundary state 2, its bf16 copy 1, chunk inputs and
  l2norm(q) rows 1.5, (I+A)^-1 rows 0.75, u/w/v_new rows 1.5) versus 2 MiB for the recurrent state: 3.4x. Against a
  long context's KV cache this is small (Qwen3.5-4B: 4 KV heads x 256 dims x 32K tokens, K and V in bf16, is 128 MiB
  per attention layer). For Qwen3.5-4B (24 GDN layers) at 256 concurrent requests the GDN state is 41 GiB instead of
  12 GiB on an 80 GB card; its 8 full-attention layers hold 32 KB of KV cache per token, so the GDN state stops
  dominating only beyond ~3.6K tokens per request. At shorter contexts memory, not kernel time, is the cost of
  exactness.
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

- Bitwise with random acceptance and wrong drafts (`test_exact.log`, "spec window"): windows of 4 (H=HV=32 and
  H=16/HV=32) and of 16 (two sub-blocks per tile; H=HV=32 and H=16/HV=32), 1..window tokens accepted per slot per
  step, one random slot idle (nothing to verify) per step, rejected positions fed with another sequence's tokens:
  2933, 2933, 3173 and 3173 accepted rows and 44, 44, 48 and 48 boundary states equal to the trainer's.
- End to end (`experiments/e12_qwen35_spec.py`, `results/gdn/e12_qwen35_spec.log`, RTX 4090 D): greedy generation of
  300 tokens after a 200-token prompt with prompt-lookup drafting (up to 4 drafts) gives the same tokens and bitwise
  the same fp32 log-probability for every generated token as one-token decode (`exact_decode.decode`) and as the
  teacher-forced forward (HF with the row-invariant stand-in and fla's chunk: the reference program, not a training
  framework), on Qwen3.5-0.8B (209 of 270 drafts accepted, 3.30 tokens per target forward) and 4B (58 of 125
  accepted, so mostly rollbacks, 1.24 tokens per forward). An earlier A800 run (`results/a800/gdn/e12_qwen35_spec.log`,
  plain path through one-token windows instead of `decode`) also matched bitwise; its texts differ from the 4090's
  because the stand-in dense ops are not cross-GPU invariant. The status quo for contrast
  (`experiments/e12b_hf_default_spec.py`, same log): stock HF/fla re-scoring the same tokens in verification windows
  of 5 instead of one at a time gives bitwise-equal log-probabilities for 17/300 (0.8B) and 2/300 (4B) tokens on the
  4090, mean |dlogp| 2.7e-3 / 8.4e-3, max 0.108 / 0.096 (A800: 24/300 and 4/300). The gap includes cuBLAS and
  attention batch-shape effects, not only GDN.
- Cost of a 4-token window per layer against vLLM 0.17.1's speculative path, which writes the state after every
  draft token into its own pool slot (that is its rollback) and starts from the accepted one (`bench_vllm_gdn.log`,
  `bench_decode.log`, config 2:4:128):

  | batch | 1 | 8 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | vLLM 0.17.1 verify, 4 tokens (us) | 32.7 | 69.5 | 248.0 | 457.5 | 901.7 | 1792.4 |
  | exact_decode window of 4 (us) | 53.7 | 77.2 | 207.2 | 348.9 | 689.1 | 1303.6 |
  | ratio | 1.64x | 1.11x | 0.84x | 0.76x | 0.76x | 0.73x |

  State memory per slot and layer for 4 drafts: vLLM keeps 1 + 4 fp32 states (10 MiB), exact_decode 6.75 MiB.
  Against fla's recurrent kernel over the same 4 tokens keeping only the final state (no rollback possible), the
  window is 1.54-1.83x at batch >= 8. If all 4 tokens are accepted a window costs 0.60x (B=256) to 0.63x (B=8) of
  exact one-token decode per token; at the 4B run's acceptance (1.24 tokens per window) it is about 2x per token.

**5. In a serving engine: vLLM 0.17.1** (`vllm/exact_gdn.py`, installed by monkeypatching before the engine is built;
`env/venv-vllm17`: vLLM 0.17.1, torch 2.10, Triton 3.6.0; logs in `a800/gdn/vllm/`, and `gdn/vllm/` for the RTX 4090 D).

Each GDN layer keeps exact_decode's caches in the request's slot of vLLM's own state pool (so vLLM sizes and schedules
them), prefills with fla 0.5.0's chunk program up to the last 64-token boundary and replays the rest through
exact_decode (whose rows are fla's), decodes with exact_decode, and verifies n-gram drafts in exact_decode windows,
committing each window's acceptance at the start of the next step. Projections, conv1d, gating and norms are vLLM's.

- Self-check (`vllm/selfcheck.py`): 16 prompts of 100-655 tokens, 128 greedy tokens each; then every prompt plus its
  generated tokens is scored in one prefill. Counted: generated tokens whose decode-time fp32 log-probability equals
  the prefill-time one bitwise; also a hash of all tokens and log-probability bits. A800 (`selfcheck.log`):

  | | stock GDN | exact GDN | stock GDN, batch-invariant dense | exact GDN, batch-invariant dense |
  |---|---|---|---|---|
  | Qwen3.5-0.8B | 274/2048 | 264/2048 | 311/2048 | 2048/2048 |
  | Qwen3.5-4B | 120/2048 | 131/2048 | 181/2048 | 2048/2048 |

  Batch-invariant dense ops (`VLLM_BATCH_INVARIANT=1`, FLASH_ATTN) alone leave GDN's recurrent-vs-chunk gap; exact GDN
  alone leaves the dense ops' batch-shape effects; together decode is the engine's prefill. With both on, exact GDN is
  2048/2048 in every further configuration, and its hash is the same with n-gram speculation (4 drafts) as without:
  whole prefill (0.8B `715ac27bc918d55b`, 4B `c591b7426314faf2`), prefill chunked into 128-token steps so prompts
  continue from a previous chunk's state (`586e6a6d6104a293`, `959d80f8ca51e8e8`), and prompts of 10-55 tokens on
  slots that held finished requests (`dd565b7655ed0b75`, `d149909481210805`). Stock GDN with speculation stays at
  311/2048 and 178/2048 and its hash changes (`0982c0620e6f8a64` vs `94d404da35349e42` for 0.8B).
  On the RTX 4090 D (`gdn/vllm/selfcheck.log`) every exact-GDN run with batch-invariant dense ops is 2048/2048 too,
  with the same hash with and without speculation, and every 0.8B hash with batch-invariant dense ops (stock or exact)
  equals the A800's: for 0.8B the engine's outputs are cross-GPU bitwise. For 4B only the chunked hash matches across
  the GPUs, and stock 4B already differs (`f6ffbb278027c1a6` vs `9ae6d50e92bcb74b`): vLLM's batch-invariant dense ops
  are not cross-GPU invariant at 4B's shapes. Changing the prefill chunk size changes the hash for a pure-attention
  model too (Qwen3-0.6B, `selfcheck_qwen3_attention_only.log`), where speculation does not: the chunk schedule is an
  input of vLLM's attention bits, not of exact GDN's.
- Throughput (`vllm/throughput.py`, `throughput.log`; Qwen3.5-4B, 256 requests of 200 prompt tokens, 256 new tokens
  each, submitted at once, after an identical warm-up):

  | | stock | exact | exact / stock | KV-cache tokens (stock / exact) |
  |---|---|---|---|---|
  | default | 6431 tok/s | 6363 tok/s | 0.99x | 415,832 / 129,328 |
  | batch-invariant dense ops | 5220 tok/s | 5170 tok/s | 0.99x | 415,360 / 129,328 |
  | n-gram speculation (piecewise graphs) | 4444 tok/s | 3872 tok/s | 0.87x | 401,672 / 127,440 |

  The cost of exactness is memory: the pool holds 3.2x fewer tokens (274 instead of 881 requests of this length).
  With speculation vLLM keeps 1 + 4 states per request and exact GDN one, so the gap narrows (270 vs 213 requests).
  On this text n-gram drafting slows stock vLLM too.
- vLLM 0.17.1 issues hit on the way (found by failing self-checks or hash mismatches; the one-token case by reading
  vLLM's GDN metadata builder): a batch mixing decode requests with and
  without drafts kills the engine on an assertion (stock too; `fix_vllm_mixed_spec_decodes`); in a step where no
  request has a draft, vLLM sends no speculative metadata and runs its plain conv update on decode requests, which
  takes the previous window's rejected drafts as conv history (exact_gdn rolls the conv state back by the acceptance
  it derives from positions); with speculation the mamba block table has 1 + drafts columns, so a request's slot
  column is a strided view; a one-token prompt runs as a decode from whatever state its slot holds (refused); and
  vLLM reuses torch.compile artifacts from its cache when the config hash matches, and an earlier artifact can be a
  different program for the dense ops (4B, same config: two hashes depending on the cache), so the self-check
  compiles afresh.
- Limits: speculation needs piecewise CUDA graphs (stock vLLM 0.17.1 fails to capture full graphs for Qwen3.5 with
  n-gram drafts); prefix caching and tensor parallelism are not handled (TP is asserted); drafters other than n-gram
  would need their drafts cut at the 64-token boundary as the n-gram proposer's are.

## What the canonical program is

"The trainer's program" is only a single program once these are fixed, on both sides:

- fla 0.5.0 `chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True` and fla's default `use_exp2` path; chunk 64.
  The gate and beta computation, conv1d and output norm around it are the caller's to match (`exact_decode` takes the
  raw log-gate g and beta as tensors, with beta in the trainer's dtype); fla's `use_gate_in_kernel` path and
  context-parallel prefill are outside the tested contract.
- `FLA_USE_FAST_OPS` unset (it swaps `exp`/`exp2` implementations).
- fla's autotune picks. fla tunes per shape, and some kernels' configs are different programs, so one forward can run
  two of them. `l2norm_fwd_kernel`: its key includes `NB = cdiv(tokens x heads, 65536)`, so a trainer's packed
  micro-batch and a short sequence are tuned separately, and 3 of its 25 configs (BT=8 with 8 or 16 warps, BT=16 with
  16 warps) give different bits from the other 22 on both GPUs and both Triton versions (`e4_autotune_triton*.log`).
  On the A800 the picks happened to fall in the majority group (`a800/gdn/e10_pack.log`: alone vs packed equal); on
  the RTX 4090 the packed batch picks BT=16 with 16 warps and 21-36% of a sequence's rows change when it is packed
  with others (`gdn/e10_pack.log`). `chunk_fwd_kernel_o`: on sm_89 with Triton 3.6.0 only, its BK=BV=64 config differs
  from the other two (`gdn/e4_autotune_triton3.6.0.log`), and since its varlen and fixed-length variants are tuned
  separately, fla's varlen forward differs from its fixed-length forward on the same sequence (about 220 of 16.8M
  output elements, `e14_fla_o_varlen.py`, `gdn/e14_fla_o_varlen.log`). `exact_decode.pin_fla_configs()` removes these
  configs; the trainer has to call it too. With it, varlen and fixed-length agree everywhere (e14); on the RTX 4090
  with Triton 3.6.0, test_exact passes with either of the other two output configs pinned and fails on 87-154 rows
  per case with BK=BV=64 pinned (`e14b_o_config_pins.py`, `gdn/e14b_o_config_pins.log`). Every other autotuned
  kernel on the forward path is config-independent (`e4_autotune*.log`).
- Triton version and GPU architecture. fla's own bits depend on the Triton version: its chunk_fwd_kernel_o epilogue
  `b_o * scale + dot * scale` is compiled to `fma(dot, scale, b_o * scale)` by Triton 3.7.1 and to
  `fma(b_o, scale, dot * scale)` by Triton 3.6.0 (read off fla's PTX), so `exact_decode` keys the form on the installed
  version and refuses unverified ones, and refuses GPUs it has not been tested on. Validated (`test_exact*.log`):
  sm_80 (A800) and sm_89 (RTX 4090 D), each with Triton 3.7.1 and 3.6.0. Hopper's wgmma is a different instruction
  and is not covered.
- The fla version, and with it the serving engine's own copy. vLLM 0.17.1 vendors an older fla whose
  `chunk_gated_delta_rule` differs from fla 0.5.0 on the same inputs in the same process (outputs by up to 4.9e-4,
  final states by up to 4.8e-3; `experiments/e13_vllm_vendored_fla.py`, `e13_vllm_vendored_fla.log`): a vLLM
  prefill is not a fla-0.5.0 trainer's forward unless the engine is pointed at the trainer's fla.

## Traps found while reproducing a Triton program bitwise

- **fp-contraction.** fla's `chunk_fwd_kernel_o` computes `b_o * scale + dot(A, v) * scale`; its PTX is
  `fma(scale, dot, b_o * scale)`. The same source in another kernel was contracted differently, which changed a few
  percent of output rows by up to ~1e-4, and which product gets contracted differs between Triton 3.6 and 3.7.1.
  `exact_decode` writes the fma out, per Triton version.
- **An autotuned kernel is a set of programs.** A config change is not a performance-only choice: it can change the
  tile shapes that feed an MMA or the contraction the compiler picks. Two fla kernels have configs with different bits
  (above), which made a vLLM prefill differ from decode on the RTX 4090 only, and only in a few rows per layer; found
  by bisecting fla's forward stage by stage on a dumped layer input.
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
needs no state snapshots; on A800 its per-layer cost is 0.73-0.84x of vLLM 0.17.1's (inexact) speculative verifier
at batch >= 32, with less state memory, and plain exact decode is 0.82-0.93x of vLLM's decode kernel at batch >= 32.
Both are slower at batch 1-8 (1.1-2.0x). Integrated into vLLM 0.17.1, decode is bitwise the engine's own prefill for
Qwen3.5-0.8B and 4B with n-gram speculation on or off, chunked prefill and slot reuse, at 0.99x stock throughput
without speculation and 0.87x with it.

Not claimed: speed versus ReplaySSM-style or bf16-state kernels, or versus the packed/CUDA GDN decode kernels of vLLM
releases after 0.17.1 (their wheels need CUDA 13, which driver 535 cannot run); anything on Hopper/Blackwell;
batch 1-8, where the exact path is slower; vLLM with prefix caching, tensor parallelism, drafters other than n-gram,
or full CUDA graphs with speculative decoding; bitwise equality of the vLLM engine to a training framework (its dense
ops are vLLM's batch-invariant kernels, not a trainer's).
