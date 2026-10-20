# Determinism Needs Fixed Trees, Not Fixed Loops: Bitwise-Reproducible LLM Serving with Every Optimization On

*Draft. Every number below is produced by a script under eval/ or detcheck/ and stored in results/.*

## Abstract

LLM serving engines return different tokens for the same request depending on what else
is in the batch, whether the prompt prefix is cached, where chunked prefill cuts the
prompt, and whether a token is produced by decode or by prefill. Existing
"deterministic modes" remove this by disabling or pinning the optimizations that create
it — fixed KV-split sizes, a single split, aligned chunk boundaries, no prefix cache on
some backends — at large throughput cost (1.9–3.5× on our GPU for SGLang's mode), and
they still leave prefill and decode numerically different. We show that bitwise reproducibility does not require sequential
reductions, only reduction trees that are fixed by problem shape and token position and
never by the schedule. For attention we make every key-chunk partial a pure function of
the query row and the chunk (chunk-local softmax statistics) and fold partials with one
position-determined two-level tree, so prefill, chunked prefill, prefix-cache hits and
split-KV decode with any number of splits all execute the same per-row operation sequence.
For projections we fix the per-element k order, which leaves the tile shape free, so tiles
can be autotuned with no numerical consequence. With seeded per-row sampling, an engine
built this way returns bitwise-identical outputs across concurrency, cache state, chunk
sizes, CUDA graphs vs eager execution, page sizes, batch limits and restarts, with every
optimization enabled. On Qwen3 1.7B–8B the throughput cost is at most 4% for batched
serving and 3–5% at batch 1 (up to 6% and 11% for 0.6B), and GSM8K accuracy is unchanged
while every one of 1,819 generations is identical between client concurrency 1 and 16. We
release the kernels, a Rust serving engine with a `--deterministic` mode, and detcheck, a
black-box checker that measures these properties for any OpenAI-compatible server.

## 1 Introduction

Reproducibility of LLM inference matters for evaluation, debugging, auditing and
reinforcement learning, where the rollout engine's log-probabilities must match the
trainer's. Temperature-0 decoding is expected to be reproducible but is not: floating-point
addition is not associative, and serving engines change the order of reductions whenever
they change how work is partitioned. Batch size changes GEMM tiling and split-K; the number
of KV splits in decode attention changes with load; chunked prefill changes where a
prompt's attention is cut; a prefix-cache hit turns part of a prefill into a read of
cached KV; and the token that decode produces with one attention kernel is recomputed by a
different (prefill) kernel when the conversation continues.

Recent work fixes this by making kernels *batch-invariant*: they keep one reduction order
regardless of batch size, typically by fixing the KV split size or disabling KV splitting
and by pinning GEMM configurations. This removes batch dependence, but (i) it costs
throughput, because the optimizations that adapt to load are exactly what is removed; (ii)
it does not make prefill and decode agree, because the two paths still compute attention
with different sequences of operations — the prefill kernel's online softmax rounds its
probabilities against a running maximum that depends on history the decode kernel never
sees; and (iii) features such as prefix caching are restricted on some backends.

Our observation is that what determinism needs is not a sequential loop but a *fixed
reduction tree*: the tree may be as parallel as we like as long as its shape is a function
of the problem (sequence positions, matrix shape) and never of the schedule. Two
ingredients make this cheap:

1. **Chunk-pure attention partials.** If the softmax statistics of a key chunk are
   computed from that chunk alone, the chunk's partial result is a pure function of the
   query row and the chunk's keys and values. Any kernel — prefill tile loop, chunked
   prefill over a cached prefix, or a split-KV decode CTA — computes bitwise the same
   partial. Folding partials with a tree fixed by absolute chunk positions then yields the
   same output on every path, while decode can still spread chunks over any number of CTAs.
2. **Per-element k order for GEMMs.** A row of `X·Wᵀ` is reproducible if every output
   element accumulates the k dimension in one fixed order. Tile sizes along M and N do not
   enter that order, so a GEMM can still use small tiles for decode and large tiles for
   prefill, chosen by an autotuner that cannot change results.

Contributions:
- The fixed-tree formulation of serving determinism and the chunk-pure attention
  construction, with a precise statement of the invariance it gives (Section 4).
- PI-Attn: Ampere/Ada CUDA kernels for paged prefill, split-KV decode and merging that
  share one per-row operation sequence, with GQA packing and a two-level fold that lets
  long-prefix prefill split over KV (Section 5).
- A row-invariant GEMM family (hand-written weight-streaming kernels and CUTLASS tiles)
  that is bitwise identical across tiles, and a batched sampler with per-row Philox streams.
- `--deterministic` in a Rust/libtorch serving engine with radix prefix caching, chunked
  prefill, mixed prefill/decode batches, decode and segmented-prefill CUDA graphs and
  CPU/GPU overlap all enabled (Section 6).
- detcheck, a black-box checker, and an evaluation showing bitwise identity across
  concurrency, cache state, decode vs re-prefill, seeded sampling, restarts and seven server
  configurations, at ≤4% batched-serving cost for 1.7B–8B models, against SGLang's
  deterministic mode, which on the same GPU is 1.9–3.5× slower, disables the radix cache, and
  still differs between decode and re-prefill and under seeded sampling (Section 7).

## 2 Background: where serving nondeterminism comes from

[Table: optimization → reduction whose order it changes → what existing deterministic modes do]
- Continuous batching: GEMM M changes → library heuristics pick different tiles / split-K.
- Split-KV decode (FlashDecoding): number of splits chosen from batch and length → the
  merge order of partial softmaxes changes.
- Chunked prefill: cut points depend on the token budget and concurrent requests → the
  prefill kernel's KV tiling relative to the query changes.
- Prefix cache: a hit replaces part of a prefill with cached KV → different kernel path
  (ragged vs paged), different tiling.
- Prefill vs decode: the token decoded at position p is later re-processed by prefill
  (multi-turn, re-prefill after eviction, RL scoring) → different kernels.
- CUDA graph padding, page size, batch limits: change shapes seen by kernels.
- Sampling: Philox streams indexed by batch row.

## 3 Related work (to expand)
- Thinking Machines, *Defeating Nondeterminism in LLM Inference* (2025): batch-invariant
  kernels, fixed split size for decode attention.
- SGLang deterministic inference (2025): fixed split size, num_splits=1 (FA3), chunk
  alignment to split size; radix cache disabled with the FlashInfer backend; ~34% average
  slowdown reported on H100 (1.9–3.5× measured on our sm_89 GPU, Section 7).
- vLLM batch invariance (`VLLM_BATCH_INVARIANT`): num_splits=1 attention, overridden aten
  matmul/reduction kernels; documented for compute capability ≥ 9.0 in 0.11.x.
- Vosti (arXiv 2609.38981): verified deterministic inference with a sequential running-max
  tile loop; prefill and decode agree but decode gets no KV-split parallelism.
- ReproFuse (arXiv 2609.25624): GEMM split-K fixed by K; TBIK (arXiv 2511.17826): TP-invariant
  reductions; LLM-42 (arXiv 2601.17768): verification and rollback; CoRun (arXiv 2608.14376):
  padding to fixed shapes; DeepSeek-V4 deterministic decode kernels.
- Reproducible summation (Demmel–Nguyen), FlashAttention/FlashDecoding, FlashInfer.

## 4 Fixed reduction trees

**Model of computation.** We assume (A1) a tensor-core MMA instruction computes each output
element as a deterministic function of its row of A, column of B and accumulator input;
(A2) `ex2.approx` is a deterministic function of its input; (A3) the compiled binary does
not reassociate the arithmetic we spell out with round-to-nearest intrinsics. A1–A2 hold per
GPU model; claims are scoped to the same GPU model and build.

**Attention.** Let keys be cut into chunks of C positions at absolute boundaries, and fold
groups of G chunks. For query row i at position p and chunk c, define
- s = (q_i · k_j) · scale·log₂e for keys j in c (fixed MMA order over head_dim), masked j ≤ p;
- m_c = max_j s_j (exact, order-free); P_j = bf16(2^(s_j − m_c)); l_c = Σ_j P_j in a fixed order;
- o_c = Σ_j P_j v_j accumulated over 16-key MMA steps in ascending order from zero.
(m_c, l_c, o_c) is a function of (q_i, K_c, V_c, mask) only. Fold(M, L, O; m, l, o) with
M′ = max(M, m), α = 2^(M−M′), β = 2^(m−M′), L′ = fma(L, α, l·β), O′ = fma(O, α, o·β).
The output is O/L after folding chunk partials sequentially within each fold group and the
group partials sequentially across groups. **Lemma 1.** For every kernel that (a) computes
chunk partials with the per-row sequence above and (b) reduces them with this tree, the
output for row i is identical, whatever rows share a kernel, how chunks are assigned to
CTAs, where chunked prefill cuts the query range, or whether keys come from a prefix cache.
*Proof sketch:* the per-row values entering every fold step are identical, and the tree is
fixed by p.

**GEMM.** y_mn = round(Σ over 16-wide k steps in ascending order, from zero). **Lemma 2.** Any
tiling of M and N that preserves this per-element order gives identical rows.

**Composition.** Embedding, norms, RoPE, activations and the KV store are row-wise.
**Theorem.** In deterministic mode every activation, KV entry and logit computed for token t
is a function of tokens 0..t only; with per-row Philox streams indexed by (seed, position),
sampled tokens are a function of tokens 0..t and the seed only.

*Why snapping the maximum to a power of two is not needed:* it would make rescaling exact,
but invariance only needs both paths to execute the same fold; snapping forces the dominant
weight to be rounded to bf16 and doubled error for peaked attention in our reference model.

## 5 PI-Attn kernels
- Shared warp-level functions: `chunk_qk` (G row groups share each K fragment), `scores_to_p`,
  `chunk_pv` (accumulators for NB n-tiles, folded immediately), `fold_*`.
- Prefill: rows are (token, head) pairs of one KV head's GQA group; K/V double-buffered via
  cp.async with zero-filled invalid positions; two-level fold in registers; optional split
  over fold groups with group partials and a merge kernel; wave-model planner.
- Decode: one warp per chunk computes the GQA group's partials into a compact workspace
  indexed by per-request chunk offsets; merge kernel realises the tree; split count free.
- Signed zeros canonicalised (masked 0·v products), NaN-safe masking, page-table addressing
  by shifts.
- Inherent cost: per-row chunk partials make split prefill expensive for many rows; the
  two-level tree confines it to group partials.

## 6 Engine integration
- llm_infer_rust: Rust scheduler, radix prefix cache, chunked prefill, mixed batches, decode
  CUDA graphs, segmented prefill graphs, CPU/GPU overlap. `--deterministic` switches the
  attention backend, routes every projection through the row-invariant GEMM, and samples
  with per-row streams. Graph replay needs no host planning (page table and lengths read on
  device); one workspace serves eager decode and all graphs; GEMM tiles tuned during graph
  warm-ups with a cold L2.

## 7 Evaluation

**Setup.** One RTX 4090 D (sm_89, driver 535, CUDA 12.6), Qwen3 0.6B/1.7B/4B/8B in bf16.
"Original" is llm_infer_rust with FlashInfer 0.6.18 attention, cuBLAS projections and its
default sampler; "PI" is the same binary with `--deterministic`. Both run with radix prefix
caching, chunked prefill (2048 tokens), mixed batches, CUDA graphs and overlap enabled.
Baselines on the same GPU: SGLang 0.5.5.post3 (`--enable-deterministic-inference`) and
vLLM 0.11.2 (`VLLM_BATCH_INVARIANT=1`). Exact versions are in VERSIONS.txt.

### 7.1 Determinism

detcheck sends token-id prompts (24–2600 tokens, 256 output tokens, `ignore_eos`) and
compares completion token ids. Within one server instance (Qwen3-1.7B):

| engine | batch c4/c16 | decode vs re-prefill | repeat ×200 (distinct) | seeded sampling c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|
| original (FlashInfer) | 1/1 of 16 | 1/24 | 14 | 6/7 of 16 | 0/0 of 8 |
| **PI (`--deterministic`)** | **0/0** | **0/24** | **1** | **0/0** | **0/0** |
| SGLang | 1/2 | 0/24 | 7 | 7/6 | 0/0 |
| SGLang deterministic | 0/0 | 1/24 | 1 | 2/1 | 0/0 † |
| vLLM | 2/1 | 0/24 | 14 | 6/6 | 1/0 |
| vLLM batch-invariant ‡ | 2/3 | 1/24 | 9 | 5/6 | 0/0 |

† SGLang disables the radix cache in deterministic mode with the FlashInfer backend (the
only one available on sm_89), so the prefix check exercises no cache hits.
‡ vLLM 0.11.x documents batch invariance for compute capability ≥ 9.0; on sm_89 it engages
(attention forced to FLASH_ATTN, aten kernels overridden) but is not batch-invariant.

SGLang's deterministic mode is the closest competitor. It removes batch dependence, but the
token decoded at position p and the same token re-processed by prefill still differ (as the
fixed-split argument of Section 2 predicts), and seeded sampling still changes with
concurrency.

**Across restarts and configurations** (Qwen3-0.6B, 16 greedy + 16 seeded-sampled requests;
references saved from a base server, compared against seven others): PI is identical in
all eight — chunk size 512, chunked prefill off, CUDA graphs off (eager), at most 4 running
requests, overlap off, page size 32, and a plain restart — while the original engine fails
11–14 comparisons in each, including the plain restart (12).

**Evaluation reproducibility.** GSM8K 5-shot through lm-eval at client concurrency 1 and 16
(`eval/accuracy.sh`, `eval/compare_accuracy.py`):

| model | mode | strict acc c=1 | c=16 | generations differing c=1 vs c=16 |
|---|---|---|---|---|
| Qwen3-1.7B (1319) | original | 69.22 | 68.92 | 893 |
| Qwen3-1.7B (1319) | PI | 69.45 | 69.45 | **0** |
| Qwen3-8B (500) | original | 90.00 | 90.00 | 327 |
| Qwen3-8B (500) | PI | 90.20 | 90.20 | **0** |

With the original engine two-thirds of the generations change with client concurrency
and the score moves by 0.3 points; with PI the evaluation is a pure function of the model.

### 7.2 Accuracy

Top-1 agreement with an fp32 HF reference over teacher-forced positions: original 99.22%,
PI 99.39% (Qwen3-0.6B). GSM8K above: PI is within noise of (and slightly above) the original.

### 7.3 Cost

Output throughput, PI relative to the original engine (`eval/e2e_matrix.sh`, median of 3,
32 requests per point; greedy, sampled within 2 points of greedy):

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | −10.9% | −2.0% | −2.1% | −4.0% | −5.9% |
| 1.7B | −4.3% | +1.3% | −1.5% | −3.0% | −2.7% |
| 4B | −2.8% | +1.0% | −0.6% | −2.9% | −3.5% |
| 8B | −4.0% | −2.3% | −1.4% | −3.9% | −3.3% |

Same-GPU baselines, greedy output tok/s on Qwen3-1.7B (`eval/baselines.sh`):

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 234 | 1394 | 3522 | 1282 | 379 |
| **PI** | **223** | **1427** | **3537** | **1242** | **370** |
| SGLang | 228 | 1373 | 2492 | 1282 | 380 |
| SGLang deterministic | 65 | 502 | 959 | 692 | 174 |
| vLLM | 198 | 1223 | 3303 | 1203 | 363 |
| vLLM batch-invariant ‡ | 198 | 1206 | 3220 | 1190 | 362 |

PI is 1.8–3.7× faster than SGLang's deterministic mode while also covering the prefix cache,
decode/re-prefill and seeded sampling, and it is within 5% of every non-deterministic engine
measured.

**Kernels** (results/PHASE1-2_kernels.md): decode attention is 0.58–1.20× the time of
nondeterministic FlashInfer and up to 3× faster than FlashInfer with a fixed split size at
batch 1 and long context; prefill is 0.69–1.5×. The remaining batch-1 cost comes from the
projections: without split-K, narrow (N=1024) GEMVs keep too few bytes in flight; deep
cp.async pipelines reduce it to 1.1× of cuBLAS at M=1 and below 1× at M≥8.

## 8 Limitations
Same GPU model and build only (A1–A3); no tensor parallelism in the engine; head_dim 128 and
GQA group ≤ 16; batch-1 projections still pay for the lack of split-K on narrow weights (a
K-segmented canonical order would recover it at a prefill cost); FP8 is not covered. The
baselines run on sm_89, where vLLM's batch invariance is outside its documented support and
SGLang can only use its FlashInfer path; Hopper/Ampere-datacenter numbers are future work.
