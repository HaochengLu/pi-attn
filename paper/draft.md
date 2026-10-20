# Determinism Needs Fixed Trees, Not Fixed Loops: Bitwise-Reproducible LLM Serving with Every Optimization On

*Draft — numbers marked [TBD] are filled from results/ after the evaluation runs.*

## Abstract

LLM serving engines return different tokens for the same request depending on what else
is in the batch, whether the prompt prefix is cached, where chunked prefill cuts the
prompt, and whether a token is produced by decode or by prefill. Existing
"deterministic modes" remove this by disabling or pinning the optimizations that create
it — fixed KV-split sizes, a single split, aligned chunk boundaries, no prefix cache on
some backends — at 20–60% throughput cost, and they still leave prefill and decode
numerically different. We show that bitwise reproducibility does not require sequential
reductions, only reduction trees that are fixed by problem shape and token position and
never by the schedule. For attention we make every key-chunk partial a pure function of
the query row and the chunk (chunk-local softmax statistics) and fold partials with one
position-determined two-level tree, so prefill, chunked prefill, prefix-cache hits and
split-KV decode with any number of splits all execute the same per-row operation sequence.
For projections we fix the per-element k order, which leaves the tile shape free, so tiles
can be autotuned with no numerical consequence. With seeded per-row sampling, an engine
built this way returns bitwise-identical outputs across concurrency, cache state, chunk
sizes, CUDA graphs vs eager execution, page sizes, batch limits and restarts, with every
optimization enabled, at a throughput cost of 0.6–6% for batched serving [TBD final]. We
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
  configurations, at [TBD] cost (Section 7).

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
  alignment to split size; radix cache support depends on backend; ~34% average slowdown.
- vLLM batch invariance (`VLLM_BATCH_INVARIANT`): num_splits=1 attention.
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

## 7 Evaluation [TBD]
- Setup: RTX 4090 D (sm_89), Qwen3 0.6B/1.7B/4B/8B, FlashInfer 0.6.18, CUDA 12.6.
- Kernel microbenchmarks (results/PHASE1-2_kernels.md, bench_attention_v6): decode 0.58–1.20×
  of nondeterministic FlashInfer and up to 3× faster than FlashInfer's fixed-split
  deterministic decode at batch 1 with long contexts; prefill 0.65–1.5×.
- Determinism (detcheck): within-instance, cross-restart and cross-config tables.
- End-to-end cost: results/e2e_matrix (greedy and sampled).
- Accuracy: agreement with fp32 reference; GSM8K.
- Baselines on the same GPU: SGLang deterministic, vLLM batch-invariant [if runnable on driver 535].

## 8 Limitations
Same GPU model and build only; no tensor parallelism in the engine; head_dim 128, GQA ≤ 16;
batch-1 decode GEMMs pay for the lack of split-K on narrow projections (a K-segmented
canonical order would recover it); FP8 not covered (sm_80/89 target).
