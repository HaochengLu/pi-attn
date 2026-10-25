# Fix the Program, Not the Schedule: Bitwise-Reproducible LLM Serving with Every Optimization On

*Draft. Every number below is produced by a script under eval/ or detcheck/ and stored in results/.*

## Abstract

LLM serving engines return different tokens for the same request depending on what else
is in the batch, whether the prompt prefix is cached, where chunked prefill cuts the
prompt, and whether a token is produced by decode or by prefill. Existing
"deterministic modes" remove this by pinning or disabling the optimizations that create
it: fixed KV-split sizes, a single split, aligned chunk boundaries, and on some backends
no prefix cache. On our GPU SGLang's deterministic mode runs at 1/1.9 to 1/3.5 of SGLang's
own throughput, and it still leaves prefill and decode numerically different. Bitwise reproducibility needs
every query row and every output element to execute one *canonical operation sequence*
that depends on positions and shapes, never on the schedule; prior work states this
requirement and meets it by pinning or slowing kernels. We treat it as a design problem:
the schedule may still partition work freely at the sequence's associativity points, so
the task is to choose a canonical sequence whose natural partitions are the ones fast
kernels want. For attention we use an online
softmax over 32-key tiles inside absolute 512-key fold groups, followed by a
fixed-order fold of the groups. Prefill kernels then have FlashAttention-2's structure,
decode kernels parallelise over groups like FlashDecoding, and prefill, chunked prefill,
prefix-cache hits and decode are bitwise identical by construction. For projections we
fix each output element's k order, cut into a number of segments fixed by the weight's
shape, which leaves the tile free to be autotuned. On an RTX 4090 D our decode attention
takes 0.64–1.02× the time of FlashInfer's faster decode and our prefill 0.51–1.09× on all but one
long-prefix extend shape. Our Rust serving engine with `--deterministic` is bitwise reproducible
across concurrency, cache state, shared-prefix batches, decode vs re-prefill, seeded
sampling, chunk sizes, CUDA graphs, page sizes, batch limits and restarts. On Qwen3
0.6B–8B its end-to-end throughput is −1.8% to +8.5% relative to the non-deterministic
engine, and 1.9–3.5× that of SGLang's deterministic mode. Its decode throughput is lower
only at batch 1 on 1.7B and 8B (by 0.6–1.8%), where its weight-streaming projections run
2–4% behind cuBLAS's GEMV on the widest and longest weights, and at one 0.6B point with
128 concurrent requests (1.6%), which a component ablation places in the attention
backend although none of its kernels is slower at that run's shapes. On prefill-bound
requests its throughput is −0.3% to +2.2%, with time to first token up to 1.6% higher on
the longest 32-head prompts. GSM8K
accuracy is unchanged, and every generation is identical between client concurrency 1
and 16. On an A800 (sm_80) every PI kernel case and every GSM8K generation gives the same
bits as on the RTX 4090 D. On the A800, end-to-end throughput relative to the
non-deterministic engine is −9.1% to +14.3% over the same workloads, and 2.2–4.1× that of
SGLang's deterministic mode. Long prompts pay up to 18% in time to first token, the cost
of the canonical 32-key online step on full-rate tensor cores.

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

**Attention.** Fast prefill kernels (FlashAttention-2 [Dao23], FlashInfer [Ye25]) stream a query tile over
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
  - On the same GPU, decode takes 0.64–1.02× the time of FlashInfer's faster decode, and
    prefill 0.51–1.09× except one long-prefix extend shape (1.21×).
- **GEMM and sampling.** A row-invariant GEMM family, bitwise identical across all of
  its members:
  - hand-written small-M and balanced weight-streaming kernels;
  - CUTLASS tiles;
  - deterministic stream-K, which cuts an element's accumulation chain between workers and
    hands the fp32 accumulator on. This balances the work like split-K without changing
    the sum;
  - segment-parallel tiles, which run a tile's K segments on separate CTAs and fold them in
    the canonical order, and segment-split decode kernels, which give each segment a warp
    and fold in shared memory.

  Sampling uses a batched sampler with per-row Philox streams.
- **Engine (Section 6).** `--deterministic` in a Rust/libtorch serving engine with radix
  prefix caching, chunked prefill, mixed prefill/decode batches, decode and
  segmented-prefill CUDA graphs and CPU/GPU overlap, all enabled.
- **Evaluation (Section 7).** detcheck, a black-box checker for any OpenAI-compatible
  server. With it we show bitwise identity across concurrency, cache state, shared-prefix
  batches, decode vs re-prefill, seeded sampling, a restart and six other server
  configurations. The measured costs and same-GPU comparisons against SGLang's and vLLM's
  deterministic modes are in Section 7. All of it is repeated on an A800, whose kernel
  outputs on every tested case and GSM8K generations are bitwise identical to the RTX 4090
  D's (Section 7.5).

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
  sampling backend. Its announcement [SGL25] reports a ~34% average slowdown on H100. On our GPU its FlashInfer
  path disables the radix cache.
- **vLLM's `VLLM_BATCH_INVARIANT`** overrides aten matmul and reductions and forces
  one-split attention. Its 0.11.x documentation requires compute capability ≥ 9.0. An open
  pull request for DeepSeek-V4.1-Flash [VLLM62365] runs every causal row, prefill or
  decode, through the unsplit (one CTA per row) sparse decode kernel "so a token decodes
  to the same bits it gets as a prompt token"; it reports 0.84–0.91× of the throughput
  without batch invariance.
- **Chunk alignment.** An open SGLang pull request [SGL44076] snaps deterministic-mode
  chunk ends to an absolute 64-token grid for Gated DeltaNet layers, at no measurable cost;
  it fixes the schedule, not the attention order, and does not make prefill and decode
  agree.

- **DeepSeek-V4** [2606.19348] makes its inference kernels batch-invariant. It gives up
  split-KV and, in most scenarios, split-K. To limit the wave quantization this causes in
  decode, a second attention kernel spreads one sequence over several SMs (exchanging data
  through thread-block-cluster distributed shared memory) with its accumulation order
  designed to match the single-SM kernel.

These systems remove batch dependence by pinning the partition, or by engineering a
second kernel to reproduce the first one's order. We keep the partition free at the
canonical program's associativity points: fold groups and K segments are fixed by
positions and shapes, so splitting there needs no matching, on GPUs without clusters
too. We also make prefill and decode agree.

*Deterministic attention.*
- **Fixed absolute splits.** Thinking Machines [TML25] note that a token's reduction
  order must not depend on how much of its sequence is already in the KV cache, which is
  what prefill/decode agreement requires. Two open-source efforts apply fixed key splits
  merged in order: mlx-batch-invariant [MLXBI] (256-key splits with a per-key online
  softmax on Apple GPUs, prefill and decode matching bitwise; prefill attention runs 3.9–4.4×
  slower than MLX's stock kernel and generation at 0.47–0.85× of stock) and an unmerged
  llama.cpp draft [LCPP] (fixed split size, padded KV cache). Neither is a serving engine
  with a prefix cache, and neither keeps stock-kernel speed. Our fold groups are the same
  idea; what we add is a sequence inside the group, an online softmax over 32-key tiles,
  that a FlashAttention-2 inner loop and a group-parallel decode kernel execute as is,
  together with a projection program that leaves the GEMM tile free.
- **FoldAttention** [2609.33410] replaces the running maximum with a reference declared
  before the scan, so split contributions add without rescaling; it targets fast decode
  and deterministic backward on Hopper and does not address prefill/decode identity.
- **Vosti** [2609.38981] specifies and verifies deterministic inference. Its attention
  runs one ordered online-softmax loop over key tiles per query, so prefill and decode
  agree. That is our canonical sequence with one group of unbounded length, so decode
  cannot split long contexts; the authors report prefill throughput below vLLM's and
  SGLang's deterministic modes.
- **FlashDecoding [FD23] and FlashInfer [Ye25]** compute split partials and merge them, but their split
  boundaries follow the schedule and their prefill uses one running maximum over the
  whole history, so neither path agrees with the other.

Our fold groups sit between these extremes. They are long enough for prefill to run an
unmodified FlashAttention-2 inner loop, and short enough (512 keys) that decode has a CTA
per group even at batch 1.

*GEMM and other reductions.* Cooper et al. [2609.25624] make the GEMM reduction order a
pure function of the problem shape, with fixed-configuration kernels that upcast weights
to fp32 in registers and accumulate with IEEE-754 fp32 arithmetic, which keeps linear
layers bitwise identical across Ampere, Ada and Hopper; TBIK
[2511.17826] makes tensor-parallel reductions invariant; CoRun [2608.14376] pads to fixed
shapes; LLM-42 [2601.17768] verifies and rolls back nondeterministic steps. A recent
stream-K for convolution weight gradients [ROCM13151] makes the fix-up deterministic by
folding published partial tiles in a fixed order; our stream-K hands on the accumulator
instead, so a cut chain equals the uncut one and no fold exists. Reproducible
summation [DN13] makes sums order-independent at higher cost; we instead fix the order.

## 4 Canonical programs

**Guarantee and scope.** For a fixed GPU model, engine binary (with its CUDA, driver and
library versions) and weights, every response is a bitwise function of the request's
token ids, sampling parameters and seed. The claim does not extend to other GPU models,
other builds, tensor parallelism or FP8 (Section 8). We test it on one physical RTX 4090 D;
that two boards of the same model agree follows from A1–A4 and is not tested.

**Model of computation.** We assume:
- **(A1)** A tensor-core MMA instruction computes each output element as a deterministic
  function of its row of A, column of B and accumulator input.
- **(A2)** `ex2.approx` is a deterministic function of its input.
- **(A3)** The compiled binary does not reassociate the arithmetic we spell out with
  round-to-nearest intrinsics.
- **(A4)** An MMA whose accumulator input is not −0 does not return −0 (as IEEE
  round-to-nearest addition would not; the PTX specification does not state it).

A1, A2 and A4 hold per GPU model, so claims are scoped to the same GPU model and build.

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
   −0 to +0, and an MMA result is not −0 when its accumulator is not −0 (A4), so o never
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
S = 2 for N ≤ 1024, K ≥ 2048 and K divisible by 512, and S = 1 otherwise. **Lemma 2.** Any tiling of M and N,
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

*Prefill and extend.* The design follows FlashInfer's scaffold:
- **CTA layout.** A CTA has 4 warps × 16 query rows. Its rows are the (token, head) pairs
  of one KV head's GQA group, so each K/V tile is loaded once for the whole group.
- **Shared memory.** Q, K and V live in 128-byte-swizzled tiles with no padding: a 64-row Q
  tile plus two 32-key K slots and two V slots, 48 KB in total. Two CTAs share an SM.
- **Software pipeline.** Iteration t computes S(t+1) = Q·K(t+1)ᵀ while it runs tile t's
  online step and P·V(t). S(t+1) depends on nothing of tile t, and every row still runs
  its online steps in tile order. K(t+2) and V(t+1) load a full iteration ahead behind
  one barrier per tile. Each thread copies four 16-byte chunks of one key, so it reads
  one page-table entry and computes one address per tile.
- **Masking.** Positions past the KV length are zero-filled without a global read, so
  stale cache contents never reach the arithmetic.
- **Group folds.** Each warp folds finished groups into a second register state. With 16
  rows per warp, both states fit in registers.

On sm_80 a second variant keeps the Q fragments in registers and moves the fp32 group
totals, touched once per 512 keys, to shared memory (80 KB, still two CTAs on the A100/A800's
164 KB). It cuts the per-tile shared-memory traffic of the Q operand. On sm_89 only the
48 KB variant fits two CTAs in 100 KB. Both variants run every row's identical sequence;
the tests and the cross-GPU hashes (Section 7.5) check both on every case.

The 16-row choice is deliberate. We first used 32 rows per warp, FlashInfer's
choice, with the cross-group total kept in an fp32 scratch in L2. In an earlier kernel
version that variant measured 8–10% slower on long prompts (ceiling probe 5), because the
total's read-modify-write every 512 keys costs more L2 bandwidth than the halved fragment
reuse of 16-row warps. On the A800, 32-row warps with the totals in shared memory fit two
64-row CTAs per SM, so 4 warps per SM instead of 8, and run 1.4–1.8× the shipped kernel's
time (`archive/ceiling_probes/a800/prefill_wide_warps.txt`).

For long prefixes with few query rows, a wave-model planner splits a row tile's groups
across CTAs. Each CTA writes group partials, and a merge kernel folds them. When the row
tiles fill a few waves, causal prefill has a tail: the last row tiles run the most key
tiles. For that case the planner simulates the block dispatch for 1, 2, 4 and 8 splits
and memoises the choice per shape. The simulation was fitted to split sweeps on the RTX
4090 D, where it cut a 2048-token chunk's attention by 9% with 16 query heads (232 → 212
µs). On the A800 every split it picked was slower than one (the same chunk: 241.9 vs
205.2 µs), so sm_80 keeps the unsplit kernel in that case. Split mode's workspace is
capped at 256 MiB, the size of a typical FlashInfer workspace.

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
(shared-prefix batch 32: 248 µs instead of 129 µs; `archive/decode_fuse_probe/out.txt`). The workspace is batch × groups
rows, and the grid depends only on the batch and the page-table width, so CUDA-graph
replay needs no host planning.

The grid covers the page table's width in groups for every request, so most of its CTAs
return at once. The group is therefore the slowest grid index:
- every working CTA is dispatched before the empty ones;
- the KV heads of a token, adjacent in the cache, are read together.

Single requests keep the group as the fastest index, which measured faster for them. On
the A800 the order cut Qwen3-1.7B's decode attention at 8 requests from 1001 to 842 µs per
step (`archive/ceiling_probes/a800/decode_grid_order.txt`).

An earlier design computed independent 64-key chunk partials in any number of splits, a
FlashDecoding-style free split. It had three costs:
- it wrote a partial per 64-key chunk instead of per 512-key group;
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
- CUTLASS 2.x sm80 tensor-op tiles from 32×64 to 128×256 for mid and large M. On
  segmented shapes they run a kernel derived from CUTLASS's own, which calls the mainloop
  once per segment from a zero accumulator and folds into a register total before the
  stock epilogue;
- deterministic stream-K kernels, for M ≤ 8 and over CUTLASS tiles, and segment-parallel
  and segment-split kernels for segmented shapes, all described below.

The balanced kernels split (n8 output tile, segment) work items contiguously over one CTA
per SM, and the warps share each X stage. With S > 1 a warp writes its fp32 partial, and
the last warp to arrive on a tile folds it, as in decode attention. These kernels exist
because decode GEMMs stream weights. With fixed 32-column tiles, a 4096-wide projection
gives 128 CTAs on 114 SMs, which leaves some SMs with twice the bytes of others. With
whole-K tiles, a 1024-wide projection has 128 work items for 114 SMs, which is why such
shapes get segments.

*Deterministic stream-K.* Some projections have too few n8 tiles to divide evenly over
the SMs; 8B's down projection has 512 tiles for 114 SMs. Segments would fix this, but they
change the program for every M, and they cost prefill 1–5% (probe 3). Instead, a stream-K
kernel for M ≤ 8 splits the (tile, k-step) pairs of all tiles evenly over one worker warp
per (SM, warp). A tile cut between workers w and w+1 stays one chain: w runs the first k
steps from zero, stores the fp32 accumulator and raises a flag, and w+1 continues the same
MMA chain from it. The result is bit-identical to the uncut chain (Lemma 2).

On segmented shapes the chains are the (tile, segment) pairs. A finished segment's
partial is folded as in the balanced kernels: the last arrival on the tile adds the
partials to +0 in ascending segment order.

At M = 8 a worker re-reads as many X bytes from L2 as weight bytes, so a variant runs
chains over two adjacent n8 tiles that share each X fragment. On the A800 it cuts 1.7B's
down projection at M = 8 from 28.0 to 24.6 µs (cuBLAS 21.2).

`test_gemm.py` checks this, as for every kernel, on every output row of every shape for
14 values of M from 1 to 4096 (including 7, 9, 17, 33 and 65, which straddle tile edges).

Two choices make the waits short and rule out deadlock. Each worker runs its handed-off
prefix first and its continuation last. And a worker only ever waits on the next-lower
worker's prefix, which itself never waits. All CTAs are resident, because the grid is at
most one CTA per SM. This is split-K without a reduction: it needs one accumulator
handoff slot per worker but no partial sums and no fold, and it has no numerical
consequence.

The same scheme runs at CTA granularity over CUTLASS tiles for mid M. Without it, a 1.7B
down projection at M = 256 has 128 tiles of 64×64 for 114 SMs, and the canonical chain
forbids the split-K that cuBLAS uses there. The kernel splits the (tile, k-tile) steps
over min(tiles, c·SMs) CTAs, with c = 1 or 2 CTAs per SM; a cut tile's accumulator
fragments go from one CTA to the next and the mainloop continues from them. Because no
CTA gets less than one tile, a piece either starts a chain or ends it, so the handoffs
never serialise a chain. The launch checks that all CTAs can be resident at once and
refuses to run otherwise, which keeps the deadlock argument above valid. At large M the
scheme loses to plain tiles: CTAs whose tiles share weight columns reach a given k at
different times and no longer share those columns in L2. The tuner picks plain tiles
there.

*Segment-parallel tiles.* The CUTLASS stream-K kernels hand on one accumulator per element
and run only unsegmented shapes. On the segmented ones (0.6B's o and down projections)
the segment-sequential tile is the problem at mid M: a 0.6B down projection at M = 64
has 32 tiles of 32×64, each running two 1536-deep segments one after the other on one
SM. The segment-parallel kernel gives each (tile, segment) pair its own CTA. Each CTA
runs its segment from zero and stores its fp32 fragments. The last CTA of a tile to
arrive, counted on a per-tile counter, adds the fragments to +0 in ascending segment
order, which is the segment-sequential kernel's exact sequence. No CTA waits on another,
so there is no residency condition. Two tiles are used: 32×64 and 64×64 with three
stages, the latter small enough for two CTAs per SM.

*Segment-split kernels.* At M ≤ 8 the same two shapes are a few microseconds of weight
streaming (4–6 MB), and the balanced kernel's fold costs there: a global partial, an
arrival counter and a re-read on the critical path, plus a second pipeline pass in the CTA
whose items cross the segment boundary. In a decode graph that made them 7% slower than
cuBLAS's GEMV (probe 16). The segment-split kernel gives each n8 tile one CTA and each
segment one warp with its own `cp.async` ring; the warps fold their partials through
shared memory in ascending segment order. At M = 1 it takes 6.01 µs on 0.6B o (balanced
6.97, cuBLAS 6.57) and 8.27 µs on down (9.33, 8.75; `results/kernels/gemm_burst.txt`).

Timing a single microsecond kernel after an L2 flush misranks candidates. For M ≤ 512
the tuner therefore times bursts of back-to-back launches over a ring of weight copies of
at least twice the L2 size: the weight itself and every weight-sized slice of a scratch
buffer. Every launch streams its weights from DRAM, as consecutive layers do in a forward;
a weight larger than the scratch is its own ring. The bursts must not be launch-bound.
Launched from Python, kernels shorter than about 10 µs are, and cuBLAS's launch path is
the slower one (0.6B o at M = 1: 9.24 µs per launch eagerly, 6.49 µs in a CUDA graph), so
our benchmarks capture each burst in a graph, as the engine runs decode. The tuner
launches from C++; its choices at M = 1 are the graph-captured best or within 1% of it.

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

Sampling parameters are packed into one pinned, non-blocking upload per step. After graph
capture has run every projection, the engine tunes each projection shape for every M
bucket up to the largest forward (chunk size plus running requests) and logs the kernel
it chose for each. The buckets are the CUDA-graph capture sizes (powers of two to 128,
then steps of 64, 128 and 256), so every graph runs the kernel timed at its own M; with
power-of-two buckets the prefill graphs between powers of two ran a kernel tuned for a
larger M, 2.7% slower on average and up to 1.55× (`archive/ceiling_probes`, probe 13).
Candidates are timed on random operands: on zeros some tiles run up to 7.5% faster
and are misranked (probe 15). Tuning takes 3.7 s (0.6B), 8.2 s (1.7B), 9.9 s (4B) and 17.3 s (8B) at startup
(`results/e2e_matrix/tuning.txt`, which also lists every choice).
With lazy tuning, the first mixed batches of each new size stalled inside measured
requests, and time to first token at concurrency 32 rose by 17%.
A tuning failure falls back to a heuristic choice, which is numerically identical.

For ablation, `--attention-backend pi` runs PI attention with cuBLAS projections, and
`--row-invariant-gemm` runs the row-invariant GEMM and the deterministic sampler with
the backend chosen by `--attention-backend` (FlashInfer in Section 7.3). Neither is
deterministic on its own.

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
section comes from one regeneration run (`eval/run_all.sh`, engine commit fdc184b), except
the SGLang and vLLM rows, which run in their own environments (`results/README.md`), the
component ablation in Section 7.3 (`eval/e2e_ablation.sh`, engine commit 870c171, which
adds only the ablation flag), and results marked as probes, whose scripts and outputs are in `archive/ceiling_probes/`.

Section 7.5 repeats the whole evaluation on an A800 with engine commit 38c32da. That
commit adds the following, all from Section 5:
- the pipelined prefill kernel, with its sm_80 variant and planner rule;
- stream-K on segmented shapes and over tile pairs;
- the decode grid order.

On sm_89 the engine runs the 48 KB prefill variant, and the tuner may pick the new GEMM
kernels. Every kernel gives on the A800 the same bits as the RTX 4090 D's kernel output
hashes (Section 7.5). None of these changes has been timed on sm_89, because the RTX 4090
D was no longer available. Sections 7.1–7.4 are therefore the 4090 D with the earlier
engine.

### 7.1 Determinism

detcheck sends token-id prompts (24–2600 tokens, 256 output tokens, `ignore_eos`) and
compares completion token ids. Within one server instance (Qwen3-1.7B), the table counts
mismatching requests:

| engine | batch c4/c16 | shared prefix ×32 | decode vs re-prefill | seeded re-prefill | repeat ×200 (distinct) | seeded sampling c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| original (FlashInfer) | 3/1 of 16 | 9/32 | 1/24 | 9/24 | 14 | 6/6 of 16 | 0/0 of 8 |
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

All seven checks pass with PI on Qwen3 0.6B, 1.7B, 4B and 8B
(`results/detcheck/final_*.log`). For example, on every model 200 repeats of one prompt
give one distinct completion, and seeded sampling at concurrency 4 and 16 matches the
sequential reference in all 16 requests.

**Across restarts and configurations** (Qwen3-0.6B, 16 greedy + 16 seeded-sampled requests;
references saved from a base server, compared against seven others). PI is identical in
all eight configurations: chunk size 512, chunked prefill off, CUDA graphs off (eager), at
most 4 running requests, overlap off, page size 32, and a plain restart. The original
engine fails 13–14 of the 32 comparisons in each, including the plain restart (13).

**Evaluation reproducibility.** GSM8K 5-shot through lm-eval at client concurrency 1 and 16
(`eval/accuracy.sh`, `eval/compare_accuracy.py`):

| model | mode | strict acc c=1 | c=16 | generations differing c=1 vs c=16 |
|---|---|---|---|---|
| Qwen3-1.7B (1319) | original | 69.22 | 69.37 | 908 |
| Qwen3-1.7B (1319) | PI | 69.07 | 69.07 | **0** |
| Qwen3-8B (500) | original | 90.00 | 90.60 | 332 |
| Qwen3-8B (500) | PI | 90.20 | 90.20 | **0** |

With the original engine about two-thirds of the generations change with client
concurrency. In this run the 1.7B score moves by 0.15 points and the 8B score by 0.60;
earlier runs of the same check moved the 1.7B score by up to 0.83 points. Both are more
than the gap between the two modes. The original engine's score is a sample, not a value.
With PI the evaluation is a pure function of the model.

### 7.2 Accuracy

Top-1 agreement with an fp32 HF reference over 1,280 teacher-forced positions per model (`eval/fp32_agreement.sh`):

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| original | 99.14% | 99.53% | 99.30% | 99.45% |
| PI | 99.61% | 99.30% | 98.91% | 99.53% |

The two modes differ by at most 6 disagreements per model, in either direction. Every disagreement is at a
position whose fp32 top-2 margin is below 0.17 logits. On GSM8K (above) PI is within noise
of the original engine.

### 7.3 Cost

**End to end.** Output throughput, PI relative to the original engine
(`eval/e2e_matrix.sh`, median of 3, 32 requests per point, greedy; sampled at T = 0.8 is
within 1.4 points of greedy everywhere):

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | +2.0% | +8.5% | +4.3% | +5.6% | +5.8% |
| 1.7B | −1.4% | +5.7% | +4.9% | +3.4% | +3.9% |
| 4B | +0.1% | +5.9% | +6.1% | +3.2% | +3.0% |
| 8B | −0.6% | +1.5% | +3.3% | +0.6% | +0.9% |

On batched workloads PI is faster than the non-deterministic engine it replaces.
At batch 1 it ranges from −1.4% (1.7B) to +2.0% (0.6B) greedy and from −1.7% to +1.2%
sampled; Section 7.4 breaks the batch-1 step down.

**Components.** Each half of the mode run alone (`eval/e2e_ablation.sh`; greedy, 32
requests per point, or 256 at c128; four modes interleaved over 3 launches, median;
`results/e2e_ablation/SUMMARY.md`). Output throughput relative to the original engine:

| model | workload | PI attention, cuBLAS GEMM | row-invariant GEMM, FlashInfer | both (`--deterministic`) |
|---|---|---|---|---|
| 0.6B | 64/256 c1 | +2.0% | −0.1% | +1.9% |
| 0.6B | 512/256 c32 | +1.5% | +2.4% | +3.3% |
| 0.6B | 512/256 c128 | −1.3% | +0.1% | −1.4% |
| 1.7B | 64/256 c1 | +0.9% | −2.3% | −1.0% |
| 1.7B | 512/256 c32 | +0.8% | +4.3% | +4.9% |
| 8B | 64/256 c1 | +1.1% | −1.9% | −0.6% |
| 8B | 512/256 c32 | +0.6% | +2.6% | +3.2% |

The three launches of each cell are within 0.5% of each other at batch 1 and within
0.8% elsewhere, except GEMM-only 1.7B at c32 (1.7%). The last column is a separate
launch from the matrix above and the high-concurrency table below, and differs from them
by up to 1.0 point (0.6B c32).

PI attention alone is faster at batch 1 and at c32 on every model. The row-invariant
GEMM is what costs at batch 1 (−1.9% to −2.3% on 1.7B and 8B), and it contributes most
of the batched gain (+2.4% to +4.3% at c32), consistent with the tuned kernels' lead
over cuBLAS at M = 8–64 (Kernels, below). The two effects add to within 0.7 points of
the combined mode.

The 0.6B c128 loss comes from attention, and mostly after the first token: time per
output token rises from 13.97 to 14.16 ms (+1.4%), TTFT by 0.8%. Those steps are pure
decode steps, which run the decode kernel inside a CUDA graph, and mixed steps, in which
both backends run the decode rows (q = 1) through their prefill kernel together with new
prompts. Neither kernel is slower at that run's shapes (batch 128, 16 query and 8 KV
heads, 512–768 keys). PI decode takes 0.94–0.99× the time of FlashInfer's, with uniform or
mixed lengths and contiguous or shuffled pages, and sizing its grid for the engine's
8192-key page table, as the decode graph does, adds under 1 µs (probe 20). PI prefill on
mixed batches of up to four 512-token prompts and 124–128 decode rows takes 0.95–0.98×
(probe 21). The planner does no device synchronisation. We have not found the cause.

Long-context and shared-prefix workloads (`eval/e2e_extended.sh`, 3 interleaved launches
per mode, median):

| model | workload (in / out / concurrency) | Δ throughput | Δ TTFT |
|---|---|---|---|
| 1.7B | 64 / 256 / c1 | −1.8% | −5.8% |
| 1.7B | 512 / 256 / c32 | +4.8% | −2.8% |
| 1.7B | 2048 / 16 / c8 | +2.2% | −1.5% |
| 1.7B | 4096 / 4 / c4 | −0.2% | +0.3% |
| 1.7B | 4096 / 128 / c8, 3968-token shared prefix | +4.9% | −5.6% |
| 1.7B | 8192 / 64 / c4, 8000-token shared prefix | +6.4% | −3.7% |
| 1.7B | 16384 / 256 / c2 | +2.3% | −0.5% |
| 1.7B | 30000 / 128 / c1 | +1.0% | −1.5% |
| 8B | 64 / 256 / c1 | −0.7% | −4.6% |
| 8B | 512 / 256 / c32 | +3.1% | −3.2% |
| 8B | 2048 / 16 / c8 | +0.3% | −0.9% |
| 8B | 4096 / 4 / c4 | −0.3% | +0.3% |
| 8B | 4096 / 128 / c8, 3968-token shared prefix | +3.2% | +1.4% |
| 8B | 8192 / 64 / c4, 8000-token shared prefix | +2.8% | −0.5% |
| 8B | 16384 / 256 / c1 (prefill only §) | | +1.6% |

Throughput is output tokens/s, or requests/s for the short-output (16 and 4 token) rows.
TTFT is 0.3% higher on the prefill-bound 4096-token rows and 1.6% higher on the 8B 16K
prompt, in line with the long 32-query-head prefill shapes in the Kernels paragraph below
(1.05–1.09×). It is also 1.4% higher on the 8B 3968-token shared-prefix row (176.0 →
178.5 ms), outside the spread of either mode's three launches, although the
corresponding extend shape is faster in isolation (0.91×); we have not attributed it.

**High concurrency** (`eval/e2e_highconc.sh`, same method):

| model | workload | Δ throughput | Δ TTFT |
|---|---|---|---|
| 0.6B | 512 / 256 / c64 | +1.0% | −1.4% |
| 0.6B | 512 / 256 / c128 | −1.6% | +1.3% |
| 0.6B | 128 / 512 / c128 | −0.0% | +2.4% |
| 0.6B | 2048 / 128 / c64 | +0.9% | −1.6% |
| 1.7B | 512 / 256 / c64 | +3.1% | −6.8% |
| 1.7B | 512 / 256 / c128 | +0.5% | −1.2% |
| 1.7B | 128 / 512 / c128 | +2.3% | +0.9% |
| 1.7B | 2048 / 128 / c64 | +1.8% | −2.0% |
| 4B | 512 / 256 / c64 | +3.7% | −2.0% |
| 4B | 128 / 512 / c64 | +6.7% | −1.7% |

0.6B at 512 / 256 / c128 is the one batched point where PI is slower (−1.6%). The
component ablation above places the loss in the attention backend, not in the
projections, but no attention kernel is slower at that run's shapes, and the cause is not
identified (probes 19–21).

§ 16384 + 256 exceeds the 8B server's context limit, so each request produces one token
and only TTFT is meaningful.

**Same-GPU baselines,** greedy output tok/s on Qwen3-1.7B (`eval/baselines.sh`; its
original and PI rows are a separate launch from the matrix above, which explains their
differences of up to 0.8% from it):

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 234 | 1392 | 3507 | 1283 | 378 |
| **PI** | **230** | **1472** | **3685** | **1325** | **393** |
| SGLang | 228 | 1377 | 3713 | 1283 | 379 |
| SGLang deterministic | 65 | 503 | 1712 | 693 | 175 |
| vLLM | 198 | 1221 | 3298 | 1205 | 366 |
| vLLM batch-invariant ‡ | 198 | 1202 | 3234 | 1192 | 363 |

PI is 1.9–3.5× faster than SGLang's deterministic mode. It is also faster than SGLang
without determinism on four of the five workloads, and faster than vLLM on all five.

**Kernels** (`kernels/bench_attention.py`, `kernels/gemm_burst.py`, results/kernels/).
- **Decode attention:** 0.64–1.02× the time of the faster of FlashInfer's CUDA-core and
  tensor-core decode across batch 1–128 and contexts of 1K–32K. With FlashInfer's split size fixed at 2048 (the
  batch-invariant configuration), PI is up to 3.3× faster at batch 1 and 32K context
  (157 µs vs 522 µs).
- **Prefill and extend:** 0.51–1.09× FlashInfer, except one extend shape. A 128-query
  extend over a 16K prefix with 32 query heads takes 1.21× (387 µs vs 319 µs). Profiled
  on its own (probe 5), its attention kernel took 321 µs against FlashInfer's 311 µs in
  total, and the merge, which reads the fp32 group partials the canonical fold needs,
  61 µs.
- **Projections** (`gemm_burst.txt`: 64 back-to-back launches, each over another weight
  copy so every launch streams from DRAM; the best PI kernel for each of the 20 shapes,
  LM heads included, against cuBLAS):
  - M = 1: 0.91–1.03×;
  - M = 8–64: 0.55–1.04×;
  - M = 128–512: 0.76–1.23×;
  - M = 1024–2048: 0.90–1.01×.

  Timed one launch at a time after an L2 flush (`gemm_cold.txt`), some M = 1 kernels
  rank worse (4B and 8B gate-up 1.16×, 8B down 1.20× cuBLAS): a lone eager launch adds a
  ramp that back-to-back launches in a graph hide. The decode trace agrees with the
  burst numbers (8B down: 110.8 µs per launch traced, 110.4 µs in the burst; Section 7.4).

  Every gap above 1.05× is at M = 128–256: 0.6B's two segmented projections (o 1.19× at
  M = 128 and 256, down 1.14× and 1.23×) and 1.7B down at M = 128 (1.13×). cuBLAS runs
  split-K there. Neither smaller segment-parallel tiles nor four segments instead of two
  close the 0.6B gaps (probe 18).

### 7.4 Where the batch-1 cost goes

Batch-1 decode streams every weight once per step. On this GPU a plain `ldg.128` stream of
a 1.7B or 8B projection's weights reaches 932–992 GB/s (`archive/ceiling_probes/out.txt`),
and cuBLAS's GEMV on the same projections 862–972 GB/s in a CUDA graph
(`gemm_burst.txt`), so the projections leave little room either way. A node-level trace of greedy decode at batch 1
(`scripts/nsys_decode.sh`, `scripts/nsys_compare.py`, `results/nsys/b1_compare.txt`)
splits each step by kernel category. Tracing every graph node lengthens the gaps between
kernels, so these numbers attribute the difference; the step times are those of
Section 7.3.

| per step, PI − original (µs) | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| projections | −18 | +38 | +121 | +262 |
| attention | −35 | +1 | −93 | −108 |
| other kernels | +11 | +20 | +35 | +40 |
| between kernels | +8 | −26 | −24 | −27 |
| traced step (original) | 2129 | 4362 | 9204 | 16524 |

**Attention** is faster or equal on every model, by up to 108 µs per step.

**Projections** carry the cost. Per launch, PI's M = 1 kernels are at most 1.5% slower
than cuBLAS's GEMV on 15 of the 20 projections, and 1.9–3.8% slower on three gate-ups, the
widest weights (0.6B 14.7 → 15.1 µs, 1.7B 53.4 → 54.4 µs, 4B 103.2 → 105.6 µs), and on the
longest rows (1.7B down 29.6 → 30.2 µs, 8B down 106.7 → 110.8 µs). 0.6B's o and down
projections, whose two segments made them 6–7% slower than the GEMV in an earlier trace
(probe 16), now take 6.05 and 8.26 µs against 6.59 and 8.77 with segment-split kernels
(Section 5), and 0.6B's projections as a whole take 18 µs less per step than cuBLAS's.

8B down is the largest single gap. Our M ≤ 8 kernels give each warp or CTA whole n8 tiles
(8 weight rows) and stream K in stages. At a fixed 96 MB, the same access pattern without
any MMA loses more as rows get longer: at K = 12288 it reads in 111.8 µs with 128- to
256-column stages, against 101.4 µs for a linear stream and 105.0 µs for cuBLAS, while
1024-column stages with one tile in flight per CTA take 106.8 µs (probe 14). So the gap is
burst length on long rows, not the 512-tile/114-SM split: a balanced deterministic
stream-K and a two-segment program both remove the split and still take 112–113.5 µs.
Stream-K workers with 1024-column stages bring 8B down to 110.4 µs at M = 1 (cuBLAS
107.0) and 113.0 µs at M = 8 (cuBLAS 119.7; `gemm_burst.txt`). What remains is the
activation half of each warp's copy ring and the MMA chain; nothing in it is a property of
the canonical program, which fixes only each element's k order.

**Other kernels** are the same RMSNorm, RoPE, KV-store and activation kernels in both
modes, yet most take up to 0.25 µs longer per launch in the PI trace. In isolation a
FlashInfer norm costs the same after a PI GEMM as after cuBLAS (probe 17), so this part is
unattributed; it is 0.2–0.5% of a traced step.

### 7.5 A800 (sm_80)

**Setup.** One A800-SXM4-80GB (sm_80, 108 SMs, 40 MB L2, driver 535.129.03), the same
CUDA, PyTorch, FlashInfer, SGLang and vLLM versions as above, and engine commit 38c32da.
Every number in this section comes from one regeneration run on that commit
(`results/a800/`, environment in `results/a800/ENV.txt`), with three exceptions, listed in
`results/README.md`:
- the SGLang and vLLM rows come from the first A800 run, in their own environments;
- the batch-1 traces were retaken after the run (below);
- the cross-GPU comparisons are computed from both GPUs' files.

Probes, and the traces cited in "Why the A800 differs", are in
`archive/ceiling_probes/a800/`. The box has one GPU, so tensor parallelism was not
tested.

**Same bits on both GPUs.** The kernel-hash harness (`eval/xgpu_hashes.py`) runs 80 GEMM
cases (20 projections × M = 1, 8, 128, 1024) and five attention cases on fixed inputs.
It hashes the outputs. Compared with the RTX 4090 D (`results/a800/xgpu/compare_4090D_A800.txt`):

| | identical on both GPUs |
|---|---|
| PI GEMM | 80 of 80 |
| cuBLAS | 14 of 80 |
| PI attention | 5 of 5 |
| FlashInfer | 1 of 5 |

The 4090 D hashes were produced by the earlier engine. The kernels added since then
change the schedule, not the program, and the check runs in two steps:
- The attention hashes come from the shipped A800 kernels and match the 4090 D directly.
- The GEMM hashes come from two reference kernels (ids 1 and 13). On the A800,
  `test_gemm.py` checks every other kernel against them bit for bit, including the new
  stream-K kernels. The new GEMM kernels therefore match the 4090 D through the
  references.

The same holds end to end. GSM8K generations through lm-eval
(`results/a800/accuracy/xgpu_4090D_vs_A800.md`) are identical between the two GPUs for
PI: 0 of 1319 differ for 1.7B and 0 of 500 for 8B, at client concurrency 1 and at 16. The
original engine differs in 906–910 of 1319 and in 325–343 of 500.

**Determinism and accuracy.** Every check of Section 7.1 repeats on the A800:
- detcheck passes on all four models (`results/a800/detcheck/`).
- PI is identical in all eight server configurations. The original engine fails 9–15 of
  32 comparisons in each, including 9 after a plain restart.
- In the baseline table (`results/a800/baselines/Qwen3-1.7B/SUMMARY.md`), SGLang's
  deterministic mode again diverges on shared prefixes (15 of 32), on decode vs
  re-prefill (1 of 24) and on seeded re-prefill (7 of 24). PI has 0 everywhere.
- GSM8K strict accuracy, c = 1 / c = 16:

  | model | PI | original |
  |---|---|---|
  | 1.7B | 69.07 / 69.07, 0 generations differ | 69.37 / 68.69, 901 differ |
  | 8B | 90.20 / 90.20, 0 differ | 89.80 / 90.00, 329 differ |

  The PI scores equal the 4090 D's, as identical generations require.
- fp32 top-1 agreement:

  | | 0.6B | 1.7B | 4B | 8B |
  |---|---|---|---|---|
  | PI | 99.61% | 99.30% | 98.91% | 99.53% |
  | original | 99.53% | 99.14% | 99.30% | 99.53% |

  The PI row equals the 4090 D's. Every disagreement is at an fp32 top-2 margin below
  0.17 logits.

**End to end.** Output throughput, PI relative to the original engine on the A800 (same
method as Section 7.3; `results/a800/e2e_matrix/SUMMARY.md`; sampled within 1.6 points
of greedy):

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | −1.6% | +8.2% | +0.9% | +5.0% | +5.7% |
| 1.7B | +5.7% | −5.9% | −0.2% | −1.4% | −0.4% |
| 4B | +14.3% | +2.4% | −0.7% | −2.4% | −1.8% |
| 8B | +6.8% | +1.0% | +1.2% | −1.0% | −2.5% |

The three launches of each cell agree within 0.5 points, except 1.7B at c32 (1.5), 4B at
c1 (1.1) and 8B at c1 (0.8).

Component ablation (`results/a800/e2e_ablation/SUMMARY.md`), relative to the original
engine:

| model | workload | PI attention, cuBLAS GEMM | row-invariant GEMM, FlashInfer | both |
|---|---|---|---|---|
| 0.6B | 64/256 c1 | +3.2% | −4.7% | −1.7% |
| 0.6B | 512/256 c32 | +1.5% | −0.5% | +1.1% |
| 0.6B | 512/256 c128 | +1.3% | −2.9% | −1.5% |
| 1.7B | 64/256 c1 | +2.0% | +4.0% | +5.8% |
| 1.7B | 512/256 c32 | +1.5% | −1.8% | −0.1% |
| 8B | 64/256 c1 | +4.7% | +1.2% | +6.5% |
| 8B | 512/256 c32 | +0.5% | +0.5% | +1.2% |

On both GPUs PI attention alone is faster everywhere (here +0.5% to +4.7%), and the
row-invariant GEMM carries the batch-1 cost: 1.7B and 8B on the 4090 D, 0.6B here. What
flips is the source of the batched gain. On the 4090 D it is the GEMM; on the A800 it is
attention, and the GEMM costs up to 1.8% at c32.

Long-context and shared-prefix workloads (`results/a800/e2e_extended/SUMMARY.md`):

| model | workload | Δ throughput | Δ TTFT |
|---|---|---|---|
| 1.7B | 64 / 256 / c1 | +5.9% | +10.0% |
| 1.7B | 512 / 256 / c32 | −0.6% | +1.3% |
| 1.7B | 2048 / 16 / c8 | −3.5% | +3.9% |
| 1.7B | 4096 / 4 / c4 | −4.3% | +4.8% |
| 1.7B | 4096 / 128 / c8, 3968-token shared prefix | +1.2% | +5.3% |
| 1.7B | 8192 / 64 / c4, 8000-token shared prefix | +1.4% | +10.2% |
| 1.7B | 16384 / 256 / c2 | −4.4% | +14.4% |
| 1.7B | 30000 / 128 / c1 | −9.1% | +17.8% |
| 8B | 64 / 256 / c1 | +6.6% | +9.0% |
| 8B | 512 / 256 / c32 | +1.2% | +3.1% |
| 8B | 2048 / 16 / c8 | −4.3% | +5.1% |
| 8B | 4096 / 4 / c4 | −5.6% | +6.1% |
| 8B | 4096 / 128 / c8, 3968-token shared prefix | +7.9% | +6.4% |
| 8B | 8192 / 64 / c4, 8000-token shared prefix | +5.5% | +7.4% |
| 8B | 16384 / 256 / c1 (prefill only §) | | +14.8% |

At high concurrency (`results/a800/e2e_highconc/SUMMARY.md`) PI ranges from −3.4% to
+6.5%:

| model | workload | Δ throughput | Δ TTFT |
|---|---|---|---|
| 0.6B | 512/256 c64 | +1.5% | +0.2% |
| 0.6B | 512/256 c128 | −1.4% | +14.0% |
| 0.6B | 128/512 c128 | −0.6% | +2.8% |
| 0.6B | 2048/128 c64 | −2.6% | +5.9% |
| 1.7B | 512/256 c64 | +0.1% | +1.3% |
| 1.7B | 512/256 c128 | −2.4% | +3.0% |
| 1.7B | 128/512 c128 | −3.4% | +2.1% |
| 1.7B | 2048/128 c64 | −0.9% | +0.4% |
| 4B | 512/256 c64 | +6.4% | +4.0% |
| 4B | 128/512 c64 | +6.5% | +2.7% |

The +14.0% TTFT of 0.6B at 512/256 c128 is the largest TTFT increase outside long prompts.
Those first tokens come from batched 512-token prefills, which run at 1.30–1.47×
FlashInfer's time on this GPU (Kernels, below).

**Same-GPU baselines,** greedy output tok/s on Qwen3-1.7B:

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 283 | 1895 | 5016 | 1879 | 539 |
| **PI** | **299** | **1785** | **4978** | **1850** | **537** |
| SGLang | 253 | 1770 | 4838 | 1836 | 529 |
| SGLang deterministic | 73 | 553 | 1991 | 842 | 192 |
| vLLM | 236 | 1595 | 4141 | 1696 | 508 |
| vLLM batch-invariant ‡ | 244 | 1535 | 4059 | 1668 | 494 |

PI is 2.2–4.1× faster than SGLang's deterministic mode. It is also faster than SGLang and
vLLM without determinism on all five workloads. vLLM's batch invariance is again outside
its documented support (compute capability ≥ 9.0).

**Kernels** (`results/a800/kernels/`).
- **Decode attention:** 0.64–1.01× the time of the faster FlashInfer decode at batch
  ≤ 32. At batch 128 it is 0.97–1.06×, behind FlashInfer's tensor-core decode.
- **Prefill and extend:** 0.41–1.00× on the four short single-request shapes (512-token
  prompts and 128-token extends over 4K keys), and 1.01–1.47× on the rest, including
  batched short prompts (1.30–1.47×). On the 4090 D the same shapes were 0.51–1.09×,
  except one extend at 1.21×.
- **Projections:**
  - In a decode step's projections replayed as a graph (`gemm_step_graph.txt`, 0.6B),
    PI takes 1.06× cuBLAS's time at M = 1, 0.75× at M = 8 and 1.02× at M = 32.
  - Per shape (`gemm_burst.txt`), M = 128–512 is 1.10–1.38× on 31 of the 60 shapes,
    22 of them o and down projections, and 0.6B's segmented o and down are 1.14–1.28× at M = 1.

**Why the A800 differs.** The canonical program is the same on both GPUs. What changes is
what it costs.

- **Prefill.** An RTX 4090 D runs the fp32-accumulate MMA at half rate, so MMA time
  dominates prefill. That hides two costs the canonical program imposes:
  - the per-32-key online step, a serial chain of shuffles and exp2 between the QK and
    P·V MMAs;
  - the ldmatrix traffic of 16-row warps. Each loaded K or V fragment feeds one MMA.
    32-row warps would halve it, but with the fp32 group totals of the cross-group fold
    they fit 4 warps per SM instead of 8.

  On the A800 the MMA runs at full rate. In the kernel before pipelining, MMA plus
  ldmatrix alone already took as long as FlashInfer's whole kernel (hq = 32, q = 8192:
  3.13 vs 3.04 ms; `prefill_ablation_flags.txt`).
  - The pipelined kernel of Section 5 cut the kernel from 1.36–1.89× to 1.13–1.47×, and
    30000-token TTFT from +41.8% to +17.8%.
  - 32-row warps are bitwise identical but run 1.4–1.8× the shipped kernel's time at the
    4 warps per SM they leave room for (probe file `prefill_wide_warps.txt`).
- **Decode.** At 1.7B and 8 requests
  (`archive/ceiling_probes/a800/nsys_c8_38c32da.txt`), PI's kernels keep the GPU busy
  332 µs longer per traced step. The traced wall step is 213 µs longer, because PI spends
  less time between kernels.
  - **Attention, 222 µs.** Each working CTA runs its group's 16 tiles in order, at
    1.1–1.4 µs per tile, plus the fold. FlashInfer splits the keys more finely and merges
    the parts.
  - **Projections, 155 µs, mostly o and down.** cuBLAS runs them as split-K with a
    reduction kernel (`splitKreduce`) and as K-sliced CTAs (`sliced1x2`). Both change
    each element's k order. The deterministic stream-K kernels of Section 5 recover
    part of it (1.7B down at M = 8: 28.0 → 24.6 µs; cuBLAS 21.2).
- **Mid-size projections.** We attribute the M = 128–512 gaps to the same two cuBLAS
  techniques, but have not traced cuBLAS at those sizes on the A800. 9 of the 31 gaps are
  on qkv and gate-up projections. One case shows the mechanism: 1.7B down at M = 128 has
  32 tiles of 64 × 64 for 108 SMs. Stream-K balances that work but cannot parallelise an
  element's k chain, and smaller tiles lose more on L2 traffic than they gain.

**Probes on the A800.** Every probe below was measured on the A800 and checked bitwise.
The decode changes were kept only if they were faster in the engine. The GEMM and
planner changes were kept on kernel-level measurements (`gemm_streamk_*.txt`,
`prefill_split_planner.txt`).

Shipped (all in 38c32da):
- the pipelined prefill kernel (Section 5);
- planner rule: no tail splits on sm_80;
- stream-K over segments and over n8 tile pairs;
- the decode grid order. In the probe's end-to-end check, 1.7B c8 went from −9.5% to
  −6.6% (`decode_grid_order.txt`). The final run measures −5.9%.

Measured and reverted:
- a device-gated deep decode kernel. It is 20–40% faster when the cache fits in L2, and
  saves ~2.8 µs per layer in the engine. A second launch costs ~2.4 µs per layer, so it
  does not pay.
- a deeper K/V ring. Its gain vanishes at the engine's capture grids.
- warp-pair specialisation and 32-row prefill warps, which are slower.

Fable checked the remaining scheduling levers against the traces and found none with a
credible engine-level gain:
- the prefill grid and wave quantisation;
- the GEMM grids;
- the merge kernels.

What remains is the price of the contract, as broken down above. Lowering it would mean
changing the canonical program itself, for example the 32-key tile or the 512-key group.
That would break the identity with the 4090 D.

**Batch-1 trace** (`results/a800/nsys/b1_compare.txt`, same method as Section 7.4):

| per step, PI − original (µs) | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| projections | +49 | −139 | −13 | +97 |
| attention | −12 | −117 | −689 | −616 |
| other kernels | −9 | +5 | −186 | −18 |
| between kernels | −38 | −55 | −323 | −304 |
| traced step (original) | 2104 | 3632 | 7586 | 11398 |

At batch 1 the A800 favours PI more than the 4090 D does, mostly through attention. About
a third of the 4B and 8B difference is time between kernels, which tracing inflates
(Section 7.4).

1.7B down is faster in PI: 25.95 → 20.74 µs, with the stream-K kernel (id 43).

0.6B is the exception, and its loss is in the projections. Its two segmented projections,
o and down, take 8.05 and 9.42 µs per launch, against cuBLAS's GEMV at 6.61 and 8.50.
These shapes need two segments per output, of 1024 and 1536 elements. Their 4 and 6 MB of
weights take only 2–3 µs to stream, so both kernels are latency-bound. The gap is the
round trip of the two-segment fold or hand-off, which every scheme we measured pays
(segment chain, segment split, stream-K over segments and over pairs). The canonical
program's two segments make that round trip unavoidable.

In the trace the 0.6B PI step is still 11 µs shorter, again because of the time between
kernels. Untraced, 0.6B is 1.6% slower at batch 1, the only model that is. Tracing used
`--cuda-flush-interval=100`; without it nsys dropped the kernel records of the 0.6B PI
run, whose start-up tuning launches over 10⁵ kernels.

## 8 Limitations

- **Scope of the guarantee.** Bitwise claims hold for one GPU model and one build
  (A1–A4). Across GPUs, an RTX 4090 D (sm_89) and an A800 (sm_80) gave identical bits for
  every PI kernel case and identical GSM8K generations (Section 7.5). That is a
  measurement, not a guarantee: the segment rule depends on shapes only, but the MMA
  instruction's internal accumulation is not specified across architectures, and Hopper's
  `wgmma` is a different instruction. One route to guaranteed cross-architecture identity
  is explicit IEEE fp32 accumulation, as in [2609.25624]; we have not measured its cost
  for this program. The cross-GPU identity also rules out tuning the canonical program per
  GPU: the segment rule and the fold groups must stay the same everywhere.
- **Engine features.** No tensor parallelism in the engine; head_dim 128 and GQA groups
  ≤ 16 only; FP8 is not covered.
- **Residual gaps.** On the RTX 4090 D:
  - One long-prefix extend shape is 1.21× FlashInfer.
  - 0.6B's segmented projections are 1.14–1.23× cuBLAS at M = 128–256 (Section 7.3).
  - Batch-1 decode of 1.7B pays 1.4–1.8% and of 8B 0.6–0.7%, which the ablation assigns to
    the row-invariant GEMM.
  - 0.6B at concurrency 128 pays 1.6%, which it assigns to the attention backend,
    although no attention kernel is slower at that run's shapes.

  On the A800 (Section 7.5):
  - Long and batched prefills are 1.01–1.47× FlashInfer. Time to first token on
    16K–30K-token prompts is 14–18% higher, and throughput on a 30000-token prompt is 9.1%
    lower.
  - Prefill-bound short-output workloads lose up to 5.6% (8B 4096/4 c4).
  - 1.7B with 8 requests is 5.9% slower.
  - 0.6B at batch 1 is 1.6% slower, because its segmented o and down projections take
    1.14–1.28× cuBLAS's time at M = 1.
  - Projections at M = 128–512 are up to 1.38× cuBLAS.

  These gaps come from the per-32-key online step, from the 16-tile chain of a decode
  group, and from cuBLAS's split-K and K-slicing, which the canonical k order rules out.
  Closing them would mean changing the canonical program, and with it every bit.
- **Single stream.** The GEMM segment, stream-K and segment-parallel workspaces and
  counters and the decode fold counters are static per-device state that every launch
  leaves zeroed. Launches that use
  them must not run concurrently, as the engine's single forward stream guarantees. A
  launch aborted mid-kernel (e.g. by a device fault) can leave a counter set, which would
  make the next launch fold early. The process must then be restarted, which any sticky
  CUDA error forces anyway.
- **Baseline coverage.** Baselines run on sm_89 and sm_80. On both, vLLM's batch
  invariance is outside its documented support (compute capability ≥ 9.0), and SGLang can
  only use its FlashInfer path, because its Triton deterministic backend stalls. Hopper
  numbers, and the cost of guaranteed cross-architecture identity there, are future work.
  Each test box had one GPU, so the engine's lack of tensor parallelism went untested
  either way.

## References

- [Dao23] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning. arXiv:2307.08691, 2023.
- [FD23] T. Dao, D. Haziza, F. Massa, G. Sizov. Flash-Decoding for long-context inference.
  Blog post, 2023.
- [Ye25] Z. Ye et al. FlashInfer: Efficient and Customizable Attention Engine for LLM
  Inference Serving. arXiv:2501.01005, 2025.
- [TML25] H. He and Thinking Machines Lab. Defeating Nondeterminism in LLM Inference.
  Connectionism, 2025.
- [SGL25] The SGLang Team. Towards Deterministic Inference in SGLang and Reproducible RL
  Training. LMSYS blog, 2025.
- [DN13] J. Demmel, H. D. Nguyen. Fast Reproducible Floating-Point Summation. ARITH 2013.
- [2606.19348] DeepSeek-AI. DeepSeek-V4: Towards Highly Efficient Million-Token Context
  Intelligence. arXiv:2606.19348.
- [2609.38981] J. Qin, A. Du, D. Zhang, M. Lentz, D. Zhuo. Vosti: Specifying, Implementing,
  and Verifying Deterministic LLM Inference. arXiv:2609.38981.
- [MLXBI] waseemh. mlx-batch-invariant: batch-invariant kernels for MLX on Apple Silicon.
  github.com/waseemh/mlx-batch-invariant.
- [LCPP] Deterministic inference mode (CUDA), llama.cpp pull request 16016 (draft), 2025.
- [2609.33410] FoldAttention: Declared-Reference Softmax for Fast Decode and Deterministic
  Backward. arXiv:2609.33410.
- [2601.17768] LLM-42: Enabling Determinism in LLM Inference with Verified Speculation.
  arXiv:2601.17768.
- [2608.14376] CoRun: Padding is Simple and Efficient for Deterministic LLM Inference.
  arXiv:2608.14376.
- [2609.25624] L. Cooper, S. Jeong, H. Jeon, J. Young, H. Kim. Accelerating the Mitigation of
  LLM Inference Nondeterminism Across GPU Architectures. arXiv:2609.25624.
- [VLLM62365] vLLM pull request 62365, Bitwise batch invariance for DeepSeek-V4.1-Flash with
  CUDA graphs (open), 2026.
- [SGL44076] SGLang pull request 44076, deterministic: Align chunked prefill to the Gated
  DeltaNet block size (open), 2026.
- [ROCM13151] ROCm rocm-libraries pull request 13151, stream-K for implicit-GEMM conv wgrad,
  2026.
- [2511.17826] Z. Zhang et al. Deterministic Inference across Tensor Parallel Sizes That
  Eliminates Training-Inference Mismatch. arXiv:2511.17826.
