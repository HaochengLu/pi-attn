# Fix the Program, Not the Schedule: Bitwise-Reproducible LLM Serving with Every Optimization On

*Draft. Every number below is produced by a script under eval/ or detcheck/ and stored in results/.*

## Abstract

LLM serving engines return different tokens for the same request depending on what else
is in the batch, whether the prompt prefix is cached, where chunked prefill cuts the
prompt, and whether a token is produced by decode or by prefill. Existing
"deterministic modes" remove this by pinning or disabling the optimizations that create
it: fixed KV-split sizes, a single split, aligned chunk boundaries, and on some backends
no prefix cache. On our GPU this costs 1.9–3.6× throughput for SGLang's mode, and it
still leaves prefill and decode numerically different. We observe that bitwise
reproducibility needs one thing only: every query row and every output element must
execute one *canonical operation sequence* that depends on positions and shapes, never on
the schedule. The schedule may still partition work freely at the sequence's
associativity points. The design problem is therefore to choose a canonical sequence
whose natural partitions are the ones fast kernels want. For attention we use an online
softmax over 32-key tiles inside absolute 512-key fold groups, followed by a
fixed-order fold of the groups. Prefill kernels then have FlashAttention-2's structure,
decode kernels parallelise over groups like FlashDecoding, and prefill, chunked prefill,
prefix-cache hits and decode are bitwise identical by construction. For projections we
fix each output element's k order, cut into a number of segments fixed by the weight's
shape, which leaves the tile free to be autotuned. On an RTX 4090 D our decode attention
takes 0.65–1.02× the time of FlashInfer's faster decode and our prefill 0.54–1.09× on all but one
long-prefix extend shape. Our Rust serving engine with `--deterministic` is bitwise reproducible
across concurrency, cache state, shared-prefix batches, decode vs re-prefill, seeded
sampling, chunk sizes, CUDA graphs, page sizes, batch limits and restarts. On Qwen3
0.6B–8B its end-to-end throughput is −2.3% to +6.9% relative to the non-deterministic
engine, and 1.9–3.6× that of SGLang's deterministic mode. The only cost, 1.0–2.3% at
batch 1, is the price of M-invariance on projections whose output tiles divide unevenly
over the SMs. GSM8K accuracy is unchanged, and every generation is identical between
client concurrency 1 and 16.

## 1 Introduction

Reproducibility of LLM inference matters for evaluation, debugging, auditing and
reinforcement learning, where the rollout engine's log-probabilities must match the
trainer's. Temperature-0 decoding is expected to be reproducible but is not.
Floating-point addition is not associative, and serving engines change the order of
reductions whenever they change how work is partitioned:
- batch size changes GEMM tiling and split-K;
- the number of KV splits in decode attention changes with load;
- chunked prefill changes where a prompt's attention is cut;
- a prefix-cache hit turns part of a prefill into a read of cached KV;
- the token that decode produces with one attention kernel is recomputed by a different
  (prefill) kernel when the conversation continues.

Recent work fixes this by making kernels *batch-invariant*. They keep one reduction order
regardless of batch size, typically by fixing the KV split size or disabling KV splitting,
and by pinning GEMM configurations. This removes batch dependence, but it has three
limits:
1. **Throughput.** It costs throughput, because the optimizations that adapt to load are
   exactly what is removed.
2. **Prefill vs decode.** It does not make prefill and decode agree. The two paths still
   run different operation sequences: the prefill kernel's online softmax rounds its
   probabilities against a running maximum over the whole history, while the decode
   kernel's split rounds against the maximum of its split.
3. **Prefix caching.** Features such as prefix caching are restricted on some backends.

Our starting point is that determinism is a property of a *program*, not of a schedule.
Fix, for each query row, one canonical sequence of floating-point operations that depends
only on the row's position and the keys it sees. Any kernel that executes exactly that
sequence for every row produces the same bits, however rows are batched and whichever
kernel runs them. The sequence still leaves the schedule free wherever the sequence
itself has independent parts. A kernel may split work at those points, and only at
those points. The choice of canonical sequence therefore decides two things: which
partitions are free, and how fast the deterministic kernels can be.

This turns serving determinism from a constraint on kernels into a design problem. Pick
the canonical sequence whose free partitions are the ones fast kernels already use.

**Attention.** Fast prefill kernels (FlashAttention-2, FlashInfer) stream a query tile over
key tiles with an online softmax. Fast decode kernels split long contexts over CTAs and
merge partial softmax states. We cut keys at *absolute* positions into fold groups of 512
keys. Within a group the canonical sequence is an online softmax over 32-key tiles in
ascending order, starting from the empty state at the group start. Across groups it is
an ascending, fixed-form fold. As a result:
- a prefill kernel can be a FlashAttention-2-style kernel that restarts its running state
  every 512 keys;
- a decode kernel can give each group of each request its own CTA;
- chunked prefill, prefix-cache hits and re-prefill of a decoded token all execute the
  identical per-row sequence.

**Projections.** A row of `X·Wᵀ` is reproducible if every output element accumulates the k
dimension in one fixed order. Tile sizes along M and N do not enter that order, so a GEMM
can use small balanced tiles for decode and large tiles for prefill, chosen by an
autotuner that cannot change results.

Contributions:
- **Formulation (Section 4).** Serving determinism as a canonical per-row program with
  free partitions at its associativity points. We give a precise invariance statement
  for attention, GEMM and their composition.
- **PI-Attn (Section 5).** Ampere/Ada CUDA kernels for paged prefill/extend, decode and
  merging that share one canonical sequence.
  - Prefill uses a FlashAttention-2 scaffold with the cross-group total kept in
    registers.
  - Decode runs one CTA per (request, KV head, group), with the 32-key tile split across
    warps without changing any element's sequence.
  - On the same GPU both are at or below FlashInfer's latency on nearly every shape we
    measure (decode 0.65–1.02×).
- **GEMM and sampling.** A row-invariant GEMM family: hand-written small-M and balanced
  weight-streaming kernels plus CUTLASS tiles, bitwise identical across all of them.
  Sampling uses a batched sampler with per-row Philox streams.
- **Engine (Section 6).** `--deterministic` in a Rust/libtorch serving engine with radix
  prefix caching, chunked prefill, mixed prefill/decode batches, decode and
  segmented-prefill CUDA graphs and CPU/GPU overlap, all enabled.
- **Evaluation (Section 7).** detcheck, a black-box checker for any OpenAI-compatible
  server. With it we show bitwise identity across concurrency, cache state, shared-prefix
  batches, decode vs re-prefill, seeded sampling, restarts and seven server
  configurations. The measured costs and same-GPU comparisons against SGLang's and vLLM's
  deterministic modes are in Section 7.

## 2 Background: where serving nondeterminism comes from

A serving engine changes how work is partitioned from one step to the next, and
floating-point reductions inherit that partition. Each optimization below changes the
order of some sum:

| optimization | reduction whose order changes | typical deterministic-mode remedy |
|---|---|---|
| continuous batching | GEMM tile and split-K choice follow M (batch tokens) | pin one GEMM config, no split-K |
| split-KV decode | number of KV splits follows batch size and length | fixed split size, or one split |
| chunked prefill | where a prompt's attention is cut relative to its keys | align chunks to the split size |
| prefix cache | cached prefix is read by a different kernel / tiling | restrict or disable the cache |
| prefill vs decode | token p is decoded by one kernel and re-processed by another | none (paths stay different) |
| CUDA-graph padding, page size, batch limit | shapes seen by kernels | pin them |
| sampling | random stream indexed by batch row or request start | per-request generator |

The prefill/decode row is the subtle one:
- An online-softmax prefill kernel rounds each probability P = bf16(exp(s − m_run))
  against a running maximum m_run that depends on every earlier key tile of the query.
- A decode kernel with a fixed split size rounds against the maximum of its split.

The two produce different bf16 values of P for the same score, so fixing the split size
cannot make them agree. Re-processing a decoded token by prefill happens whenever a
conversation continues, a preempted request is recomputed, or an RL trainer scores a
rollout.

## 3 Related work

*Batch-invariant kernels.* Thinking Machines [TML25] identified batch-size dependence as
the dominant source of serving nondeterminism. They proposed batch-invariant RMSNorm,
matmul, and attention with a fixed split size for decode.
- **SGLang's deterministic inference** builds on that work: a fixed split size for
  FlashInfer/Triton, one split for FA3, chunked-prefill alignment, and the `pytorch`
  sampling backend. It reports a ~34% average slowdown on H100. On our GPU its FlashInfer
  path disables the radix cache.
- **vLLM's `VLLM_BATCH_INVARIANT`** overrides aten matmul and reductions and forces
  one-split attention. Its 0.11.x documentation requires compute capability ≥ 9.0.

These systems remove batch dependence by pinning the partition. We keep the partition
free at the canonical program's associativity points, and we also make prefill and
decode agree.

*Deterministic attention.*
- **Vosti** [2609.38981] obtains prefill/decode agreement with a single sequential
  running-max loop per query. That is our canonical sequence with one group of unbounded
  length, so decode cannot split long contexts.
- **FlashDecoding and FlashInfer** compute split partials and merge them, but their split
  boundaries follow the schedule and their prefill uses one running maximum over the
  whole history, so neither path agrees with the other.

Our fold groups sit between these extremes. They are long enough for prefill to run an
unmodified FlashAttention-2 inner loop, and short enough (512 keys) that decode has a CTA
per group even at batch 1.

*GEMM and other reductions.* ReproFuse [2609.25624] fixes split-K by K; TBIK
[2511.17826] makes tensor-parallel reductions invariant; CoRun [2608.14376] pads to fixed
shapes; LLM-42 [2601.17768] verifies and rolls back nondeterministic steps; DeepSeek-V4
ships deterministic decode kernels. Reproducible summation [Demmel–Nguyen] makes sums
order-independent at higher cost; we instead fix the order.

## 4 Canonical programs

**Model of computation.** We assume:
- **(A1)** A tensor-core MMA instruction computes each output element as a deterministic
  function of its row of A, column of B and accumulator input.
- **(A2)** `ex2.approx` is a deterministic function of its input.
- **(A3)** The compiled binary does not reassociate the arithmetic we spell out with
  round-to-nearest intrinsics.

A1–A2 hold per GPU model, so claims are scoped to the same GPU model and build.

**Attention.** Keys are cut at absolute positions into tiles of T = 32 keys and fold groups
of 16 tiles (512 keys). For query row i at position p, the canonical program runs the
following.

*Scores.* For each tile t:

    s_j = (q_i · k_j)·scale·log₂e, by MMA over head_dim in ascending 16-wide steps from
          zero, masked to −∞ for j > p.

*Group g.* Start from the empty state (m, d, o) = (−∞, 0, 0). For each of the group's tiles
in ascending order:

    m′ = max(m, max_j s_j)                              exact, order-free
    μ  = m′ if m′ > −∞ else 0;  σ = 2^(m − μ)
    P_j = bf16(2^(s_j − μ))
    d  = fma(d, σ, Σ_j P_j)                              Σ in one fixed order
    o  = fma(o, σ, +0), then o += P·V by MMA over the tile's two 16-key steps, ascending
    m  = m′

*Across groups.* Fold the group partials (m_g, d_g, o_g) in ascending g into a total
(M, L, O), starting from (−∞, 0, 0):

    M′ = max(M, m_g),  α = 2^(M−M′),  β = 2^(m_g−M′)
    L  = fma(L, α, d_g·β),  O = fma(O, α, o_g·β)

A partial with m_g = −∞ is the identity. The output is O/L.

Two details make the program well defined on every path.

1. **Masked steps are no-ops.** A prefill row may meet a tile entirely after its
   position, which decode never processes. For that step, m′ = m and σ = 2^0 = 1, so d is
   unchanged. All P are 0, and o + 0 = o unless o = −0. Rescaling by fma(o, σ, +0) maps
   −0 to +0, and an MMA result cannot be −0 when its accumulator is not −0, so o never
   holds −0. Fully masked steps therefore leave the state bit-identical, as if skipped.
2. **The first fold is exact.** Folding into the identity returns the partial exactly:
   α = 0, β = 1 and fma(0, 0, x) = x. So it does not matter whether a kernel holds the
   first group in registers or reads it from a workspace.

**Lemma 1 (attention).** Let a kernel execute the program above for every row it owns, in
the stated order within each group and across groups. Then the output for row i is
identical, whatever:
- rows share a kernel launch;
- how groups are assigned to CTAs, as long as a CTA runs a whole group in order;
- where chunked prefill cuts the query range;
- whether keys come from a prefix cache or were just written;
- whether the row is a decode row or a prefill row.

*Proof sketch.* Every floating-point operation applied to row i, and every operand it
reads, is fixed by p and the keys and values at positions ≤ p. A kernel's freedom lies in
which rows, groups and tile slices it co-schedules. Splitting a tile across warps
partitions S by key columns and O by output columns; it does not change any element's
MMA chain or the exact max. The row sum Σ_j P_j is recomputed in the canonical order.

**Choice of T and G.** The tile T is what prefill streams through shared memory per online
step. The group size trades decode parallelism (more groups, more CTAs at batch 1)
against prefill work: one fold per group, and on long-prefix extends, the split
granularity. On the chunk-pure predecessor of these kernels we measured groups of 512,
1024 and 2048 keys. 512 was fastest for decode at batch ≤ 8, for shared prefixes and for
short extends, with no measurable cost to long prefill. At 512, decode at batch 1 has
1–64 CTAs per KV head for 0.5K–32K contexts.

**GEMM.** K is cut into S = S(N, K) equal segments, a function of the weight shape alone.
Segment s accumulates its 16-wide k steps in ascending order by MMA from zero into p_s, and
y_mn = round((((+0 + p_0) + p_1) + …) + p_{S−1}). With S = 1 this is y = round(p_0). We use
S = 2 for N ≤ 1024 and K ≥ 2048, and S = 1 otherwise. **Lemma 2.** Any tiling of M and N,
any assignment of (output tile, segment) pairs to CTAs or warps, any choice of which
CTA folds a tile's segments, and any cut of a segment's chain between workers that hands
the fp32 accumulator from one to the next give identical rows.

Segments are deterministic split-K. Split-K is nondeterministic in existing engines for
two reasons: the split count follows M and the SM count, and the partials are combined
by atomics or in arrival order. Fixing S by the weight's shape, and the fold by segment
index, removes both. A segment is worth its cost only where split-K is worth it, on
narrow projections with long K. There, decode has too few n8 output tiles to occupy
every SM, so the rule spends segments only on those shapes.

**Composition.** The remaining operators are row-wise or elementwise, but two of them are
row-wise only under conditions worth stating.
- **Row-wise reductions.** RMSNorm (with residual add) reduces within one row in a fixed
  order, and q/k-norm runs one warp per head.
- **Elementwise operators.** RoPE, SiLU-and-mul, the embedding gather and the KV store
  are elementwise.
- **Sampler softmax.** It reduces each vocabulary row in one block whose access pattern
  depends on the row's alignment. Rows are 16-byte aligned because 4·|V| is a multiple of
  16 (|V| = 151,936), so every row is reduced the same way.
- **Greedy argmax.** Greedy decoding uses argmax over bf16 logits, where exact ties are
  common. Its cross-block combine is batch-dependent, but it selects the first maximal
  index, a total order, so the result is not.

**Theorem.** In deterministic mode, every activation, KV entry and logit computed for token
t is a function of tokens 0..t only. With per-row Philox streams indexed by
(seed, position), sampled tokens are a function of tokens 0..t and the seed only.

## 5 PI-Attn kernels

All kernels target sm_80/89 with `mma.sync` m16n8k16 (bf16 in, fp32 accumulate). One
header (`pi_attn.cuh`) holds the canonical pieces:
- `online_step`: one tile's max, rounding, row sum and rescale;
- the fold functions, built from explicit `__fmaf_rn` / `__fmul_rn` and `ex2.approx`;
- `finish`.

Kernels are free to choose how rows, groups and tile slices map to warps and CTAs. They
are not free to change any element's sequence.

*Prefill and extend.* The design follows FlashInfer's scaffold on Ada:
- **CTA layout.** A CTA has 4 warps × 16 query rows. Its rows are the (token, head) pairs
  of one KV head's GQA group, so each K/V tile is loaded once for the whole group.
- **Shared memory.** Q, K and V live in 128-byte-swizzled tiles with no padding: a 64-row Q
  tile plus single 32-key K and V tiles, 32 KB in total. Two CTAs share an SM.
- **Pipeline.** K for tile t+1 loads during tile t's P·V, and V for t+1 loads during the
  next S. Page-table reads run one tile ahead, off the critical path.
- **Masking.** Positions past the KV length are zero-filled without a global read, so
  stale cache contents never reach the arithmetic.
- **Group folds.** Each warp folds finished groups into a second register state. With 16
  rows per warp, both states fit in registers.

The register choice is deliberate. We first used 32 rows per warp, FlashInfer's
choice, with the cross-group total kept in an fp32 scratch in L2. That version was
8–10% slower on long prompts, because the total's read-modify-write every 512 keys costs
more L2 bandwidth than the halved fragment reuse of 16-row warps.

For long prefixes with few query rows, a wave-model planner splits a row tile's groups
across CTAs. Each CTA writes group partials, and a merge kernel folds them. When the row
tiles fill a few waves, causal prefill has a tail: the last row tiles run the most key
tiles. For that case the planner simulates the block dispatch for 1, 2, 4 and 8 splits
and memoises the choice per shape. The simulation was fitted to measured split sweeps,
and it cuts a 2048-token chunk's attention by 8%. Split mode's
workspace is capped at 256 MiB, the size of a typical FlashInfer workspace.

*Decode.* One CTA of 4 warps runs one (request, KV head, group). Per tile:
- each warp computes one 8-key slice of S;
- the tile max is exchanged through shared memory;
- each warp rounds its slice of P;
- warp 0 recomputes Σ P in the canonical order;
- each warp accumulates four of the sixteen output n-tiles.

Decode is a single kernel. A request with one group (≤ 512 keys) finishes in place, since
the first fold is exact (Section 4). Otherwise each warp writes its 32-column slice of the
group partial together with its own copy of (m, d). It then counts its arrival on a
per-(request, KV head, warp) counter. The warp that arrives last resets the counter and
folds that slice of every head over the groups, in ascending order. Which CTA folds is a
scheduling accident, but the folded values and their order are not.

No CTA-wide barrier may follow the arrival. Our first version parked the whole CTA while
thread 0's atomic round-tripped through L2, and that doubled kernel time when L2 was busy
(shared-prefix batch 32: 248 µs instead of 129 µs). The workspace is batch × groups
rows, and the grid depends only on the batch and the page-table width, so CUDA-graph
replay needs no host planning.

An earlier design computed independent 64-key chunk partials in any number of splits, a
FlashDecoding-style free split. It had three costs:
- it wrote 6–13% extra traffic;
- its merge serialised hundreds of folds at long context;
- under prefix sharing it overflowed a pool-sized workspace and fell back to a
  sequential path.

Group-level work removes all three.

*Row-invariant GEMM.* Every variant realises the per-shape segmented program of Section 4,
so all are bitwise identical (Lemma 2). An autotuner picks the fastest per (N, K,
M-bucket), and that choice has no numerical effect. The variants are:
- hand-written small-M kernels with deep `cp.async` pipelines (128–256 k per stage),
  which fold segments in registers;
- *balanced* kernels for M ≤ 64, described next;
- CUTLASS 2.x sm80 tensor-op tiles for large M. On segmented shapes they run a kernel
  derived from CUTLASS's own, which calls the mainloop once per segment from a zero
  accumulator and folds into a register total before the stock epilogue.

The balanced kernels split (n8 output tile, segment) work items contiguously over one CTA
per SM, and the warps share each X stage. With S > 1 a warp writes its fp32 partial, and
the last warp to arrive on a tile folds it, as in decode attention. These kernels exist
because decode GEMMs stream weights. With fixed 32-column tiles, a 4096-wide projection
gives 128 CTAs on 114 SMs, which leaves some SMs with twice the bytes of others. With
whole-K tiles, a 1024-wide projection has 128 work items for 114 SMs, which is why such
shapes get segments.

*Deterministic stream-K.* Some projections have too few n8 tiles to divide evenly over
the SMs; 8B's down projection has 512 tiles for 114 SMs. Segments would fix this, but they
change the program for every M, and they cost prefill (Section 7.4). Instead, a stream-K
kernel for M ≤ 8 splits the (tile, k-step) pairs of all tiles evenly over one worker warp
per (SM, warp). A tile cut between workers w and w+1 stays one chain: w runs the first k
steps from zero, stores the fp32 accumulator and raises a flag, and w+1 continues the same
MMA chain from it. The result is bit-identical to the uncut chain (Lemma 2), which
`test_gemm.py` checks for every shape and M ≤ 8.

Two choices make the waits short and rule out deadlock. Each worker runs its handed-off
prefix first and its continuation last. And a worker only ever waits on the next-lower
worker's prefix, which itself never waits. All CTAs are resident, because the grid is at
most one CTA per SM. This is split-K without a reduction, so it needs neither workspace
nor fold, and it has no numerical consequence.

Timing a single microsecond kernel after an L2 flush misranks candidates. The tuner
therefore times bursts of back-to-back launches, each streaming a different weight-sized
slice of a scratch buffer larger than L2, as consecutive projections do in a forward.

## 6 Engine integration

We implement the mode in llm_infer_rust, a Rust/libtorch serving engine with a radix
prefix cache, chunked prefill, mixed prefill/decode batches, decode CUDA graphs, segmented
prefill graphs and CPU/GPU overlap. `--deterministic` changes three things and disables
nothing:
1. **Attention** uses PI-Attn for prefill, extend and decode.
2. **Projections.** Every projection, including the LM head, goes through the
   row-invariant GEMM.
3. **Sampling** uses a batched fork of FlashInfer's top-k/top-p kernel. A row's Philox
   stream is keyed by (request seed, position) rather than by batch row. Requests without
   a seed get a random one, echoed in the response, so any output can be reproduced.

Sampling parameters are packed into two pinned, non-blocking uploads per step. After graph
capture has run every projection, the engine tunes each projection shape for every
power-of-two M bucket up to the largest forward (chunk size plus running requests). That
takes 0.5–2 s at startup. With lazy tuning, the first mixed batches of each new size
stalled inside measured requests, and time to first token at concurrency 32 rose by 17%.
A tuning failure falls back to a heuristic choice, which is numerically identical.

## 7 Evaluation

**Setup.** One RTX 4090 D (sm_89, 114 SMs, 72 MB L2, driver 535, CUDA 12.6), Qwen3
0.6B/1.7B/4B/8B in bf16. "Original" is llm_infer_rust with FlashInfer 0.6.18 attention,
cuBLAS projections and its default sampler; "PI" is the same binary with
`--deterministic`. The original engine (and the kernel benchmarks' FlashInfer baseline)
uses FlashInfer's CUDA-core paged decode (`use_tensor_cores=False`, the wrapper's
default); Section 7.3 also compares against its tensor-core decode. Both run with radix
prefix caching, chunked prefill (2048 tokens),
mixed batches, CUDA graphs and overlap enabled. Baselines on the same GPU: SGLang
0.5.5.post3 (`--enable-deterministic-inference`) and vLLM 0.11.2
(`VLLM_BATCH_INVARIANT=1`). Exact versions are in VERSIONS.txt. Every number in this
section comes from one regeneration run (`eval/run_all.sh`, engine commit 83a90ec).

### 7.1 Determinism

detcheck sends token-id prompts (24–2600 tokens, 256 output tokens, `ignore_eos`) and
compares completion token ids. Within one server instance (Qwen3-1.7B), the table counts
mismatching requests:

| engine | batch c4/c16 | shared prefix ×32 | decode vs re-prefill | seeded re-prefill | repeat ×200 (distinct) | seeded sampling c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| original (FlashInfer) | 1/1 of 16 | 11/32 | 1/24 | 5/24 | 11 | 7/6 of 16 | 0/0 of 8 |
| **PI (`--deterministic`)** | **0/0** | **0/32** | **0/24** | **0/24** | **1** | **0/0** | **0/0** |
| SGLang | 2/2 | 11/32 | 0/24 | 8/24 | 7 | 6/6 | 0/0 |
| SGLang deterministic (FlashInfer) | 0/0 | 14/32 | 1/24 | 8/24 | 1 | 1/1 | 0/0 † |
| SGLang deterministic (Triton) | server stalled (request timed out after 1200 s) | | | | | | |
| vLLM | 3/3 | 14/32 | 0/24 | 8/24 | 11 | 5/6 | 1/1 |
| vLLM batch-invariant ‡ | 2/3 | 14/32 | 1/24 | 8/24 | 12 | 5/6 | 0/0 |

† SGLang disables the radix cache in deterministic mode with the FlashInfer backend, so the
prefix check exercises no cache hits.
‡ vLLM 0.11.x documents batch invariance for compute capability ≥ 9.0. On sm_89 it engages
(attention forced to FLASH_ATTN, aten kernels overridden) but is not batch-invariant.

SGLang's deterministic mode is the closest competitor. It removes plain batch dependence,
but 32 requests that share a prefix still diverge in 14 of 32, a token decoded at
position p and the same token re-processed by prefill still differ, and seeded sampling
still changes with concurrency.

The same checks pass on every model: for Qwen3 0.6B, 1.7B, 4B and 8B, 200 repeats of one
prompt give one distinct completion, and seeded sampling at concurrency 4 and 16 matches
the sequential reference in all 16 requests.

**Across restarts and configurations** (Qwen3-0.6B, 16 greedy + 16 seeded-sampled requests;
references saved from a base server, compared against seven others). PI is identical in
all eight configurations: chunk size 512, chunked prefill off, CUDA graphs off (eager), at
most 4 running requests, overlap off, page size 32, and a plain restart. The original
engine fails 11–15 of the 32 comparisons in each, including the plain restart (12).

**Evaluation reproducibility.** GSM8K 5-shot through lm-eval at client concurrency 1 and 16
(`eval/accuracy.sh`, `eval/compare_accuracy.py`):

| model | mode | strict acc c=1 | c=16 | generations differing c=1 vs c=16 |
|---|---|---|---|---|
| Qwen3-1.7B (1319) | original | 69.22 | 69.22 | 886 |
| Qwen3-1.7B (1319) | PI | 69.07 | 69.07 | **0** |
| Qwen3-8B (500) | original | 90.00 | 90.40 | 324 |
| Qwen3-8B (500) | PI | 90.20 | 90.20 | **0** |

With the original engine about two-thirds of the generations change with client
concurrency. The 8B score moves by 0.40 points, twice the gap between the two modes; the
1.7B score happens not to move in this run, although 886 of its 1319 generations change
(an earlier run of the same check moved it by 0.83 points). With PI the evaluation is a pure
function of the model.

### 7.2 Accuracy

Top-1 agreement with an fp32 HF reference over 1,152 teacher-forced positions per model:

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| original | 99.22% | 99.48% | 99.31% | 99.39% |
| PI | 99.74% | 99.22% | 98.96% | 99.48% |

The two modes differ by at most 6 disagreements per model, in either direction. Every disagreement is at a
position whose fp32 top-2 margin is below 0.17 logits. On GSM8K (above) PI is within noise
of the original engine.

### 7.3 Cost

**End to end.** Output throughput, PI relative to the original engine
(`eval/e2e_matrix.sh`, median of 3, 32 requests per point, greedy; sampled at T = 0.8 is
within 1.2 points of greedy everywhere):

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | −1.1% | +6.9% | +1.9% | +6.3% | +5.0% |
| 1.7B | −1.1% | +5.1% | +2.3% | +3.1% | +3.6% |
| 4B | −1.0% | +3.8% | +1.4% | +1.7% | +1.7% |
| 8B | −1.5% | −0.3% | +0.6% | −0.3% | +0.2% |

On batched workloads PI is faster than the non-deterministic engine it replaces, because
its decode attention is faster than FlashInfer's (below). The only cost is at batch 1,
1.0–1.5% greedy (up to 2.3% sampled), largest for 8B.

Long-context and shared-prefix workloads (`eval/e2e_extended.sh`, 3 interleaved launches
per mode, median):

| model | workload | Δ throughput | Δ TTFT |
|---|---|---|---|
| 1.7B | 4096-token shared prefix, 128 out, c8 | +3.8% | −3.9% |
| 1.7B | 8000-token shared prefix, 64 out, c4 | +5.9% | −1.1% |
| 1.7B | 16384 in / 256 out, c2 | +1.8% | +0.7% |
| 1.7B | 30000 in / 128 out, c1 | +0.4% | −0.8% |
| 8B | 4096-token shared prefix, c8 | +1.1% | +3.4% |
| 8B | 8000-token shared prefix, c4 | +1.4% | +1.8% |
| 8B | 16384-token prompt (prefill only §) | | +1.7% |

§ 16384 + 256 exceeds the 8B server's context limit, so each request produces one token
and only TTFT is meaningful.

**Same-GPU baselines,** greedy output tok/s on Qwen3-1.7B (`eval/baselines.sh`):

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 234 | 1391 | 3515 | 1274 | 375 |
| **PI** | **231** | **1461** | **3599** | **1302** | **387** |
| SGLang | 228 | 1377 | 3713 | 1283 | 379 |
| SGLang deterministic | 65 | 503 | 1712 | 693 | 175 |
| vLLM | 198 | 1221 | 3298 | 1205 | 366 |
| vLLM batch-invariant ‡ | 198 | 1202 | 3234 | 1192 | 363 |

PI is 1.9–3.6× faster than SGLang's deterministic mode. It is also faster than SGLang
without determinism on four of the five workloads, and faster than vLLM on all five.

**Kernels** (`kernels/bench_attention.py`, `kernels/gemm_burst.py`, results/kernels/).
- **Decode attention:** 0.65–1.02× the time of the faster of FlashInfer's CUDA-core and
  tensor-core decode across batch 1–128 and contexts of 1K–32K. With FlashInfer's split size fixed at 2048 (the
  batch-invariant configuration), PI is up to 3.3× faster at batch 1 and 32K context
  (155 µs vs 513 µs).
- **Prefill and extend:** 0.54–1.09× FlashInfer, except one extend shape. A 128-query
  extend over a 16K prefix with 32 query heads takes 1.22×. The attention kernel alone
  matches FlashInfer's total there; the extra 61 µs is the merge, which reads the fp32
  group partials the canonical fold needs.
- **Projections** (cold weights, every launch streaming from DRAM, LM heads included):
  0.84–1.04× cuBLAS at M = 1, except 8B down at 1.08×; 0.60–1.01× at M = 8.

### 7.4 Where the batch-1 cost goes

Batch-1 decode is weight streaming. On this GPU:
- a plain `ldg.128` stream of a projection's weights reaches 946–994 GB/s;
- cuBLAS's GEMV reaches 890–986 GB/s;
- our kernels' access pattern alone, with no MMA, reaches 0.97–1.02× cuBLAS on every shape
  but one (`archive/ceiling_probes`).

The full kernels add 1–9% over their own access pattern: X staging, `ldmatrix` and the
per-stage barriers. That overhead is the bulk of the remaining 1.0–1.1% greedy batch-1 cost on
0.6B–4B.

The exception is 8B down. Its 512 n8 tiles leave the busiest SMs streaming 5 tiles against
an average of 4.49. Two exact remedies exist:
- **K segments** change the canonical program for every M. In our sweep they recover the
  decode time (118.5 → 112.1 µs) but cost 1–5% on every newly segmented shape at prefill M
  (8B o at M = 4096: 949.6 → 993.0 µs). Part of that cost is pipeline refill between
  segments in our CUTLASS mainloop rather than the program itself.
- **Deterministic stream-K** keeps the program and moves only the schedule. It reaches
  113.2 µs at M = 1 (cuBLAS 104.8 µs) and beats cuBLAS at M = 8 (116.8 vs 121.1 µs), so
  the engine uses it. End to end it cut 8B's batch-1 cost from 2.2% to 1.5%.

What remains on 8B down is the handoff latency and the kernel overhead above. It is a
kernel deficiency, not a property of the canonical program.

## 8 Limitations

- **Scope of the guarantee.** Bitwise claims hold for one GPU model and one build
  (A1–A3). The segment rule depends on shapes only, so it is GPU-independent, but MMA
  internals are not specified across architectures.
- **Engine features.** No tensor parallelism in the engine; head_dim 128 and GQA groups
  ≤ 16 only; FP8 is not covered.
- **Residual kernel gaps.** One long-prefix extend shape is 1.22× FlashInfer (Section
  7.3), and batch-1 decode of 8B pays 1.5–1.7%.
- **Single stream.** The segment and stream-K handoff workspaces and the decode fold
  counters are static per-device state that every launch leaves zeroed. Launches that use
  them must not run concurrently, as the engine's single forward stream guarantees. A
  launch aborted mid-kernel (e.g. by a device fault) can leave a counter set, which would
  make the next launch fold early. The process must then be restarted, which any sticky
  CUDA error forces anyway.
- **Baseline coverage.** All baselines run on sm_89, where vLLM's batch invariance is
  outside its documented support and SGLang can only use its FlashInfer path. Ampere and
  Hopper data-centre numbers are future work.
