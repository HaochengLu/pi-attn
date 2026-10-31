# Fix the Program, Not the Schedule: Bitwise-Reproducible LLM Serving with Every Optimization On

*Draft. Every number below is produced by a script under eval/, detcheck/, kernels/ or gdn/ and stored in results/, or, for probes, in archive/.*

## Abstract

LLM serving engines return different tokens and log-probabilities for the same request
depending on what else is in the batch, whether the prompt prefix is cached, where chunked
prefill cuts the prompt, and whether a token is produced by decode or by prefill; an RL trainer
that rescores the rollout with its own forward gets yet another value. Existing
deterministic modes pin or disable the optimizations that cause this, and still leave prefill
and decode numerically different. We make every query row and every output element execute
one *canonical operation sequence* that depends on positions and shapes only, and choose that
sequence so that its natural partitions are the ones fast kernels want. For attention it is
an online softmax over 32-key tiles inside absolute 512-key fold groups, folded in a fixed
order: prefill kernels keep FlashAttention-2's structure, decode kernels parallelise over
groups, and prefill, chunked prefill, prefix-cache hits and decode are bitwise identical by
construction. For projections each output element's k order is fixed and cut into a number
of segments fixed by the weight's shape, which leaves the tile free to be autotuned;
declaring the largest tensor-parallel degree in that rule makes outputs independent of the
degree as well. In a Rust serving engine with radix caching, chunked prefill, CUDA graphs and
CPU/GPU overlap all on, `--deterministic` is bitwise reproducible, tokens and
log-probabilities, across every load, cache and configuration change we test, and a PyTorch
forward on the same kernels, as an RL trainer runs it over packed micro-batches, reproduces
the log-probability of every one of 55,593 sampled tokens on Qwen3 0.6B–8B (a standard bf16
trainer forward misses 1% importance-ratio accuracy on 10–22% of them). On an A800, end-to-end
throughput relative to the same engine without determinism is −5.9% to +14.2% on 40
workload points, within 2.8% of vLLM 0.17.1's where it is behind, and 2.0–3.7× that of
vLLM 0.17.1's batch-invariant mode, the one baseline that is also deterministic there. Prefill pays: prompts
of 16K–30K tokens take 12–18% longer to the first token, and Qwen3-0.6B serving 2K-token
prompts at 64 requests loses 8% of throughput, the cost of the canonical 32-key online step
on full-rate tensor cores. Kernel outputs and GSM8K generations are bitwise identical
between the A800 and an RTX 4090 D. The same principle makes Gated DeltaNet decode run the
trainer's chunked program, bitwise inside vLLM with speculative decoding on or off, at
0.96–1.08× of stock vLLM's throughput.

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
  - Kernel-against-kernel costs on the A800 are in Section 7.4.
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

  Sampling uses a batched sampler with per-row Philox streams, and the sampled tokens'
  log-probabilities come from a log-softmax with one fixed reduction order per row.
- **Engine (Section 6).** `--deterministic` in a Rust/libtorch serving engine with radix
  prefix caching, chunked prefill, mixed prefill/decode batches, decode and
  segmented-prefill CUDA graphs and CPU/GPU overlap, all enabled.
- **Rollout log-probabilities equal the trainer's (Section 7.3).** A PyTorch forward on the
  engine's kernels, as an RL trainer would run it over packed micro-batches, reproduces the
  engine's log-probability of every sampled token bit for bit, so importance ratios are
  exactly 1; a standard bf16 trainer forward misses by more than 1% on 10–22% of tokens.
- **Exact Gated DeltaNet decode (Section 8).** The same principle for linear attention:
  hybrid models decode with the trainer's own chunked program, with speculative
  verification and rollback for free, bitwise inside vLLM at stock throughput.
- **Evaluation (Section 7).** detcheck, a black-box checker for any OpenAI-compatible
  server, which compares tokens and their log-probabilities bit for bit. With it we show
  bitwise identity across concurrency, cache state, shared-prefix batches, decode vs
  re-prefill, seeded sampling, a restart and six other server configurations, on an A800
  with the final engine. Kernel outputs on every tested case and GSM8K generations are
  bitwise identical between the A800 and an RTX 4090 D (Section 7.2). The test suite is
  checked against deliberately broken kernels, which it must reject (Section 7.7).

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
  one-split attention. Its documentation requires compute capability ≥ 9.0; on the A800,
  0.11.2 is not batch-invariant and 0.17.1 is (Section 7.1). An open
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
other builds, tensor parallelism or FP8 (Section 10). We test it on one A800 and one RTX
4090 D; that two boards of the same model agree follows from A1–A4 and is not tested.

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
- **Token log-probability.** log softmax(z)[t] for the sampled token t, in fp32, is a new
  row reduction, so we fix it explicitly: one CTA of 1024 threads per row; thread j reads
  columns j, j + 1024, … in ascending order for the exact maximum m and then for
  Σ 2^((z − m)·log₂e); the per-thread sums are added in a fixed shared-memory tree; the result
  is (z_t − m) − log₂(Σ)·ln 2. Only explicitly rounded operations and the `ex2`/`lg2`
  approximate instructions are used (A2 extended to `lg2`), so the bits do not depend on
  the compiler's contraction choices, and a trainer linking the same header computes the
  same value.
- **Greedy argmax.** Greedy decoding uses argmax over bf16 logits, where exact ties are
  common. Its cross-block combine is batch-dependent, but it selects the first maximal
  index, a total order, so the result is not.

**Theorem.** In deterministic mode, every activation, KV entry, logit and log-probability
computed for token t is a function of tokens 0..t only. With per-row Philox streams indexed
by (seed, position), sampled tokens are a function of tokens 0..t and the seed only.

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
48 KB variant fits two CTAs in 100 KB. On sm_80 the register variant also runs with 8 warps
(128 rows) per CTA, one CTA per SM: each K/V tile is copied to shared memory once for twice
as many rows, which halves the copy traffic per row. It is 5–8% faster on long and batched
prefills and up to 12% slower on short extends, whose few CTAs then leave SMs idle, so the
launcher takes it when the 128-row CTAs fill two waves
(`archive/ceiling_probes/a800/close/probe_prefill_warps.txt`). All variants run every row's
identical sequence; the tests check every variant and split count on every case.

The 16-row choice is deliberate. We first used 32 rows per warp, FlashInfer's
choice, with the cross-group total kept in an fp32 scratch in L2. In an earlier kernel
version that variant measured 8–10% slower on long prompts (ceiling probe 5), because the
total's read-modify-write every 512 keys costs more L2 bandwidth than the halved fragment
reuse of 16-row warps. On the A800 we built 32-row warps with the totals in shared memory:
two 64-row CTAs per SM, so 4 warps per SM instead of 8, at 255 registers with 840 bytes of
spills (64 without the S pipeline); the faster, unpipelined one ran 1.4–1.8× the 4-warp kernel's time
(`archive/ceiling_probes/a800/prefill_wide_warps.txt`). That measures this build, not the
design: a spill-free 32-row step that fuses both 16-row halves was not built.

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
- CUTLASS 2.x sm80 tensor-op tiles from 32×64 to 128×256 for mid and large M, with 32- or
  64-deep k stages. On segmented shapes they run a kernel derived from CUTLASS's own,
  which calls the mainloop once per segment from a zero accumulator and folds into a
  register total before the stock epilogue. A stage's depth does not enter the program: the
  warp-level MMAs still step through k in ascending 16-wide steps. At M = 128–512 the
  64-deep tiles halve the mainloop's barriers per k and took several projections from
  1.1–1.3× cuBLAS's time to 0.85–1.05× (`archive/ceiling_probes/a800/close/probe_gemm_cfg.txt`);
- deterministic stream-K kernels, for M ≤ 8 and over CUTLASS tiles, and segment-parallel
  and segment-split kernels for segmented shapes, all described below.

The balanced kernels split (n8 output tile, segment) work items contiguously over one CTA
per SM, and the warps share each X stage. With S > 1 a warp writes its fp32 partial, and
the last warp to arrive on a tile folds it, as in decode attention. These kernels exist
because decode GEMMs stream weights. With fixed 32-column tiles, a 4096-wide projection
gives 128 CTAs on the RTX 4090 D's 114 SMs, which leaves some SMs with twice the bytes of others. With
whole-K tiles, a 1024-wide projection has 128 work items for 114 SMs, which is why such
shapes get segments.

*Deterministic stream-K.* Some projections have too few n8 tiles to divide evenly over
the SMs; 8B's down projection has 512 tiles for 114 SMs (108 on the A800). Segments would fix this, but they
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
down projection at M = 8 to 22.7 µs against cuBLAS's 21.2 (kernel 61,
`results/a800/kernels/gemm_burst.txt`).

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
shared memory in ascending segment order. On the RTX 4090 D, at M = 1 it takes 6.01 µs on 0.6B o (balanced
6.97, cuBLAS 6.57) and 8.27 µs on down (9.33, 8.75; `results/kernels/gemm_burst.txt`).

Timing a single microsecond kernel after an L2 flush misranks candidates. For M ≤ 512
the tuner therefore times bursts of back-to-back launches over a ring of weight copies of
at least twice the L2 size: the weight itself and every weight-sized slice of a scratch
buffer. Every launch streams its weights from DRAM, as consecutive layers do in a forward;
a weight larger than the scratch is its own ring. The bursts must not be launch-bound.
Launched from Python, kernels shorter than about 10 µs are, and cuBLAS's launch path is
the slower one (0.6B o at M = 1 on the RTX 4090 D: 9.24 µs per launch eagerly, 6.49 µs in a CUDA graph; probe
`eager_vs_graph.py`, `archive/ceiling_probes/README.md`), so
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

A completion request with `logprobs: 0` also receives each sampled token's log-probability
(Section 4), in either mode. The kernel runs only in steps where a row asks for it, reads
the logits the sampler already holds, and its result travels to the host in the same
asynchronous copy as the token ids, so the CPU/GPU overlap is unchanged.

Sampling parameters are packed into one pinned, non-blocking upload per step. After graph
capture has run every projection, the engine tunes each projection shape for every M
bucket up to the largest forward (chunk size plus running requests) and logs the kernel
it chose for each. The buckets are the CUDA-graph capture sizes (powers of two to 128,
then steps of 64, 128 and 256), so every graph runs the kernel timed at its own M; with
power-of-two buckets the prefill graphs between powers of two ran a kernel tuned for a
larger M, 2.7% slower on average and up to 1.55× (`archive/ceiling_probes`, probe 13).
Candidates are timed on random operands: on zeros some tiles run up to 7.5% faster
and are misranked (probe 15). Tuning takes 4.0 s (0.6B), 7.8 s (1.7B), 10.1 s (4B) and 15.5 s (8B) at startup on the A800
(`results/a800/e2e_matrix/tuning.txt`, which also lists every choice).
With lazy tuning, the first mixed batches of each new size stalled inside measured
requests, and time to first token at concurrency 32 rose by 17%.
A tuning failure falls back to a heuristic choice, which is numerically identical.
Because tuning runs after capture, a tuned kernel may need a larger CUTLASS scratch buffer than
the captured graphs were recorded with. *Scratch:* a grown buffer keeps its predecessor alive,
since freeing it left the graphs launching into freed memory, which crashed 0.6B at 128
running requests on its first request in the regeneration run.

For ablation, `--attention-backend pi` runs PI attention with cuBLAS projections, and
`--row-invariant-gemm` runs the row-invariant GEMM and the deterministic sampler with
the backend chosen by `--attention-backend` (FlashInfer in Section 7.4). Neither is
deterministic on its own.

## 7 Evaluation

**Setup.** One A800-SXM4-80GB (sm_80, 108 SMs, 40 MB L2, driver 535.129.03, CUDA 12.6,
PyTorch 2.13, FlashInfer 0.6.18), Qwen3 0.6B/1.7B/4B/8B in bf16. We did not change clocks:
under load the SM clock sat at the board's 1410 MHz boost in 78% of one-minute samples and
never fell below 1170 MHz (`results/a800/run_all/clocks.log`). "Original" is llm_infer_rust
with FlashInfer attention, cuBLAS projections and its default sampler; "PI" is the same binary
with `--deterministic`. The original engine (and the kernel benchmarks' FlashInfer baseline)
uses FlashInfer's CUDA-core paged decode (`use_tensor_cores=False`, the wrapper's default);
the kernel benchmarks also time its tensor-core decode. Both modes run with radix prefix
caching, chunked prefill (2048 tokens), mixed batches, CUDA graphs and overlap enabled.
Baselines on the same GPU: SGLang 0.5.5.post3 (`--enable-deterministic-inference`), vLLM
0.11.2 and vLLM 0.17.1 (`VLLM_BATCH_INVARIANT=1`).

Every number in this section comes from one regeneration run (`eval/run_all.sh`; outer
commit a51ba7e, engine 34203cf; `results/a800/`, environment in `results/a800/ENV.txt`). That
run exposed a crash at 128 running requests on 0.6B (Section 6, *scratch*); the stages it
touches, and every bitwise check, were rerun on the fixed engine 1ee2bc7: kernel tests,
the cross-configuration and detcheck checks, rollout log-probabilities, the
high-concurrency and ablation runs and the vLLM 0.17.1 batch-invariant baseline. The fix
changes no kernel. Probes, whose scripts and outputs are in `archive/ceiling_probes/`, are
marked as such. An RTX 4090 D (sm_89) with the earlier engine fdc184b supplies the cross-GPU
comparison (Section 7.2) and Section 7.6. The A800 has one GPU, so tensor parallelism is
evaluated with emulated ranks (Section 9).

### 7.1 Determinism

detcheck sends token-id prompts (24–2600 tokens, 256 output tokens, `ignore_eos`) and
compares, request by request, the completion's token ids and each token's fp32
log-probability bits. Within one server instance (Qwen3-1.7B,
`results/a800/baselines/Qwen3-1.7B/SUMMARY.md`), the table counts requests whose tokens or
log-probabilities differ from the reference:

| engine | batch c4/c16 | shared prefix ×32 | decode vs re-prefill | seeded re-prefill | repeat ×200 (distinct) | seeded sampling c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| original (FlashInfer) | 16/16 of 16 | 32/32 | 24/24 | 24/24 | 169 | 16/16 of 16 | 5/6 of 8 |
| **PI (`--deterministic`)** | **0/0** | **0/32** | **0/24** | **0/24** | **1** | **0/0** | **0/0** |
| SGLang | 16/16 | 32/32 | 24/24 | 24/24 | 33 | 16/16 | 8/8 |
| SGLang deterministic (FlashInfer) | 5/5 | 30/32 | 24/24 | 24/24 | 1 | 4/6 | 0/0 † |
| SGLang deterministic (Triton) | server stalled (request timed out after 1200 s) ‡ | | | | | | |
| vLLM 0.11.2 | 16/16 | 32/32 | 24/24 | 24/24 | 169 | 16/16 | 8/6 |
| vLLM 0.11.2 batch-invariant § | 16/16 | 32/32 | 24/24 | 24/24 | 168 | 16/16 | 7/2 |
| vLLM 0.17.1 | 16/16 | 32/32 | 24/24 | 24/24 | 167 | 16/16 | 8/6 |
| vLLM 0.17.1 batch-invariant | 0/0 | 0/32 | 0/24 | 8/24 ‖ | 1 | 0/0 | 0/0 |

† SGLang disables the radix cache in deterministic mode with the FlashInfer backend, so the
prefix check exercises no cache hits.
‡ From the first A800 run; the regeneration run did not repeat it.
§ vLLM documents batch invariance for compute capability ≥ 9.0. On sm_80, version 0.11.2
engages it (aten kernels overridden, attention forced to one split) but is not
batch-invariant. Version 0.17.1 is, with FlashAttention 2 named as the attention backend (it
refuses to start otherwise) and prefix caching on.
‖ The seeded-sampling variant of decode vs re-prefill holds only if the sampler keys its
random stream by (seed, position); vLLM keys it by the request's start, so a re-prefilled
request draws different numbers. The greedy variant, which compares the same
log-probabilities without a sampler, passes (0/24).

Log-probabilities are the stricter test. At token level (the earlier A800 run with engine
38c32da, `results/a800/baselines_38c32da_tokens/`), the original engine matched itself on
decode vs re-prefill (0 of 24) and on prefix-cache hits (0 of 8), and SGLang's deterministic
mode differed on 1 of 24 decode-vs-re-prefill requests. With log-probabilities every one of
the 24 differs, for both: a token decoded at position p and the same token re-processed by
prefill always get different values, even where the argmax agrees. SGLang's deterministic
mode removes most plain batch dependence of tokens, but its log-probabilities still change
with concurrency (5 of 16) and across 30 of 32 requests that share a prefix.

detcheck passes on all four models with log-probabilities compared
(`results/a800/detcheck/final_*.log`): on every model 200 repeats of one prompt give one
distinct completion, and seeded sampling at concurrency 4 and 16 matches the sequential
reference in all 16 requests.

**Across restarts and configurations** (Qwen3-0.6B, 16 greedy + 16 seeded-sampled
requests; references saved from a base server, compared against seven others;
`results/a800/crossconfig/`). PI is identical in all eight configurations: chunk size 512,
chunked prefill off, CUDA graphs off (eager), at most 4 running requests, overlap off, page
size 32, and a plain restart. The original engine fails 11–13 of the 32 comparisons in each,
including the plain restart (11).

**Evaluation reproducibility.** GSM8K 5-shot through lm-eval at client concurrency 1 and 16
(`eval/accuracy.sh`, `eval/compare_accuracy.py`, `results/a800/accuracy/`):

| model | mode | strict acc c=1 | c=16 | generations differing c=1 vs c=16 |
|---|---|---|---|---|
| Qwen3-1.7B (1319) | original | 69.37 | 68.69 | 888 |
| Qwen3-1.7B (1319) | PI | 69.07 | 69.07 | **0** |
| Qwen3-8B (500) | original | 89.80 | 89.80 | 322 |
| Qwen3-8B (500) | PI | 90.20 | 90.20 | **0** |

With the original engine two-thirds of the 1.7B generations change with client concurrency,
and its score moves by 0.68 points, more than the gap between the two modes. The original
engine's score is a sample, not a value. With PI the evaluation is a pure function of the
model.

### 7.2 Same bits on two GPUs, and accuracy

The kernel-hash harness (`eval/xgpu_hashes.py`) runs 80 GEMM cases (20 projections × M = 1,
8, 128, 1024) and five attention cases on fixed inputs and hashes the outputs. Compared with
the RTX 4090 D (`results/a800/xgpu/compare_4090D_A800.txt`):

| | identical on both GPUs |
|---|---|
| PI GEMM | 80 of 80 |
| cuBLAS | 14 of 80 |
| PI attention | 5 of 5 |
| FlashInfer | 1 of 5 |

The 4090 D hashes were produced by the earlier engine. The kernels added since then change
the schedule, not the program, and the check runs in two steps. The attention hashes come
from the shipped A800 kernels and match the 4090 D directly. The GEMM hashes come from two
reference kernels (ids 1 and 13); on the A800, `test_gemm.py` checks every other kernel,
including the 64-deep tiles and stream-K kernels added since, against them bit for bit.

The same holds end to end. GSM8K generations (`results/a800/accuracy/xgpu_4090D_vs_A800.md`)
are identical between the two GPUs for PI: 0 of 1319 differ for 1.7B and 0 of 500 for 8B,
at client concurrency 1 and at 16. The original engine differs in 906–910 of 1319 and in
325–343 of 500.

**Accuracy.** Top-1 agreement with an fp32 HF reference over 1,280 teacher-forced positions
per model (`eval/fp32_agreement.sh`):

| | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| original | 99.53% | 99.14% | 99.30% | 99.53% |
| PI | 99.61% | 99.30% | 98.91% | 99.53% |

The two modes differ by at most 5 disagreements per model, in either direction, and every
disagreement is at a position whose fp32 top-2 margin is below 0.17 logits. The PI row equals
the 4090 D's, as identical bits require. On GSM8K (above) PI is within noise of the original
engine.

Attention error is also asserted in the kernel tests, not only printed (Section 7.7): against
float64, PI-Attn's mean error is below FlashInfer's, and its maximum error no larger, on every
GQA geometry and query scale tested.

### 7.3 Rollout log-probabilities equal the trainer's

An RL trainer recomputes the log-probability of every sampled token with its own forward
and divides by the rollout engine's value to get an importance ratio. When the two
programs differ, the ratio drifts from 1 on tokens that the policy never changed, and
methods such as truncated or masked importance sampling exist to absorb that noise. Row
invariance removes the noise at its source, provided the trainer's forward runs the same
canonical program as the engine.

The engine returns each sampled token's log-probability with `logprobs: 0` on
`/v1/completions`, computed by the fixed-order log-softmax of Section 4 (`pi_logprob.cuh`).

The trainer side (`kernels/engine_forward.py`) is the engine's Qwen3 forward in PyTorch: the
PI GEMM, PI-Attn's causal prefill over a paged cache, the engine's own RMSNorm, q/k-norm,
RoPE and SiLU kernels (compiled from the engine's source with its flags), and the same
log-softmax. It is forward-only: a trainer back-propagates through its own differentiable
copy, as FlashAttention recomputes its forward.

`eval/rollout_logprobs.py` (`results/a800/rollout/`) sends 64 GSM8K questions per model (chat template, 256 tokens,
temperature 1 with a per-request seed, every fourth request greedy) to the deterministic
engine at concurrency 16, then stops the server and scores exactly those sequences with
two trainers: the PI forward, each sequence alone and in two random packings of 2–8
sequences per micro-batch, and HF transformers in bf16 with SDPA attention and an fp32
log-softmax, alone and in right-padded batches of 8.

| model | completion tokens | PI trainer, bitwise equal (alone / packing 0 / packing 1) | HF trainer, bitwise equal | HF: tokens with |ratio − 1| > 1% | HF ratio range |
|---|---|---|---|---|---|
| 0.6B | 13,063 | all / all / all | 136 (1.0%) | 22.2% | 0.59–1.52 |
| 1.7B | 14,293 | all / all / all | 5,275 (36.9%) | 11.5% | 0.49–1.64 |
| 4B | 13,975 | all / all / all | 5,698 (40.8%) | 10.4% | 0.61–2.51 |
| 8B | 14,262 | all / all / all | 5,084 (35.6%) | 11.4% | 0.61–1.44 |

(HF columns: each sequence scored alone; batches of 8 give 10.6–22.1% and ratios
0.49–2.11.) With the PI forward every one of the 55,593 completion tokens has the engine's
log-probability bit for bit, whichever packing the trainer uses, so every importance ratio
is exactly 1. The engine's tokens came from decode steps in batches of up to 16 requests
under chunked prefill; the trainer recomputes them in one prefill per micro-batch.

### 7.4 Cost

**End to end.** Output throughput, PI relative to the original engine
(`eval/e2e_matrix.sh`, `results/a800/e2e_matrix/SUMMARY.md`; median of 3 interleaved
launches, 32 requests per point or twice the concurrency where larger, greedy):

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | −3.4% | +7.1% | +0.6% | +4.5% | +4.3% |
| 1.7B | +5.7% | −5.8% | −0.4% | −0.7% | +0.2% |
| 4B | +13.8% | +2.5% | −0.6% | −2.2% | −1.5% |
| 8B | +6.2% | +1.1% | +1.2% | −1.1% | −2.5% |

Sampled at T = 0.8 every cell is within 1.7 points of greedy (overall range −5.9% to
+14.2%). The three launches of a cell agree within 1 point except at nine points, the widest
being 0.6B at batch 1 (−3.4, −6.4, −3.4) and sampled 0.6B 2048/256 c16 (+4.0, +8.6, +4.0).
At batch 1 PI is faster on 1.7B, 4B and 8B by 5.7–13.8% and slower on 0.6B; Section 7.5 breaks
the step down. Batched workloads are within ±2.5% except 1.7B at 8 requests (−5.8%) and three 0.6B
points (+4.3% to +7.1%).

**Components.** Each half of the mode run alone (`eval/e2e_ablation.sh`; greedy, 32
requests per point, or 256 at c128; four modes interleaved over 3 launches, median;
`results/a800/e2e_ablation/SUMMARY.md`). Output throughput relative to the original engine:

| model | workload (in / out / concurrency) | original tok/s | PI attention only | row-invariant GEMM only | both (`--deterministic`) |
|---|---|---|---|---|---|
| 0.6B | 64 / 256 / c1 | 513.5 | +3.3% | −6.1% | −2.9% |
| 0.6B | 512 / 256 / c32 | 7454.7 | +1.3% | −0.8% | +0.4% |
| 0.6B | 512 / 256 / c128 | 11641.0 | −0.3% | −1.4% | −1.7% |
| 1.7B | 64 / 256 / c1 | 282.7 | +1.8% | +3.9% | +5.9% |
| 1.7B | 512 / 256 / c32 | 5080.5 | +1.1% | −1.6% | −0.6% |
| 8B | 64 / 256 / c1 | 89.9 | +4.7% | +1.4% | +6.8% |
| 8B | 512 / 256 / c32 | 1813.7 | +0.5% | +0.5% | +1.1% |

PI attention alone is never slower than FlashInfer by more than 0.3% on these decode-heavy
points and gains 1.8–4.7% at batch 1, where its unsplit-or-folded decode replaces
FlashInfer's CUDA-core kernel and merge (Section 7.5). The row-invariant GEMM costs up to 1.6%
at 32 and 128 requests (8B gains 0.5%) and 6.1% on 0.6B at batch 1, whose o and down projections are its
slowest shapes (Section 7.4, *Kernels*); on 1.7B and 8B at batch 1 it gains 1.4–3.9%. The
two halves roughly add.

**Long-context and shared-prefix workloads** (`eval/e2e_extended.sh`, 3 interleaved launches
per mode, median; `results/a800/e2e_extended/SUMMARY.md`):

| model | workload (in / out / concurrency) | Δ throughput | Δ TTFT |
|---|---|---|---|
| 1.7B | 64 / 256 / c1 | +5.6% | +8.8% |
| 1.7B | 512 / 256 / c32 | −0.3% | −0.6% |
| 1.7B | 2048 / 16 / c8 | −1.7% | +0.1% |
| 1.7B | 4096 / 4 / c4 | −2.7% | +2.9% |
| 1.7B | 4096 / 128 / c8, 3968-token shared prefix | +1.6% | +5.4% |
| 1.7B | 8192 / 64 / c4, 8000-token shared prefix | +1.2% | +12.3% |
| 1.7B | 16384 / 256 / c2 | −3.7% | +12.2% |
| 1.7B | 30000 / 128 / c1 | −9.4% | +18.3% |
| 8B | 64 / 256 / c1 | +6.2% | +3.6% |
| 8B | 512 / 256 / c32 | +1.1% | +3.6% |
| 8B | 2048 / 16 / c8 | −4.4% | +5.2% |
| 8B | 4096 / 4 / c4 | −5.8% | +6.3% |
| 8B | 4096 / 128 / c8, 3968-token shared prefix | +8.1% | +3.9% |
| 8B | 8192 / 64 / c4, 8000-token shared prefix | +5.5% | +6.3% |
| 8B | 16384 / 256 / c1 (prefill only ¶) | | +13.2% |

Throughput is output tokens/s, or requests/s for the short-output (16 and 4 token) rows.
Prefill is where PI pays on this GPU: time to first token rises by up to 6.3% on 2K–4K prompts and
by 12–18% on 16K–30K prompts and on 1.7B's 8000-token shared prefix, and the 30000-token request loses 9.4% of its throughput. The
batch-1 TTFT increases (+3.6% and +8.8%) are 0.5 ms on 64-token prompts. Shared-prefix
decode gains (+1.2% to +8.1%) despite the higher TTFT.

¶ 16384 + 256 exceeds the 8B server's context limit, so each request produces one token and
only TTFT is meaningful.

**High concurrency** (`eval/e2e_highconc.sh`, same method; `results/a800/e2e_highconc/SUMMARY.md`):

| model | workload (in / out / concurrency) | Δ throughput | Δ TTFT |
|---|---|---|---|
| 0.6B | 512 / 256 / c64 | +1.2% | +6.8% |
| 0.6B | 512 / 256 / c128 | −2.0% | +18.9% |
| 0.6B | 128 / 512 / c128 | +1.4% | −2.5% |
| 0.6B | 2048 / 128 / c64 | −8.0% | +11.3% |
| 1.7B | 512 / 256 / c64 | +0.7% | +0.8% |
| 1.7B | 512 / 256 / c128 | −0.3% | +2.3% |
| 1.7B | 128 / 512 / c128 | +0.9% | −0.1% |
| 1.7B | 2048 / 128 / c64 | −2.9% | +3.4% |
| 4B | 512 / 256 / c64 | +6.5% | +2.8% |
| 4B | 128 / 512 / c64 | +7.5% | +2.8% |

Decode-heavy points at 64–128 requests are within −2.0% to +7.5%. The cost is again prefill:
0.6B's 2048-token prompts at 64 requests lose 8.0% of throughput, and 0.6B's time to first
token at 128 requests rises 18.9%, where every step is a large batched prefill whose
attention and projections are a larger share of a small model's step. On 1.7B the same points
cost 2.9% and 2.3%.

**Same-GPU baselines,** greedy output tok/s on Qwen3-1.7B (`eval/baselines.sh`; its original
and PI rows are a separate launch from the matrix above, which explains differences of up to
1.8% from it):

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 283 | 1894 | 5014 | 1878 | 539 |
| **PI** | **298** | **1790** | **4951** | **1863** | **542** |
| SGLang | 253 | 1768 | 4865 | 1837 | 530 |
| SGLang deterministic | 73 | 554 | 1993 | 843 | 192 |
| vLLM 0.11.2 | 237 | 1599 | 4149 | 1694 | 508 |
| vLLM 0.11.2 batch-invariant § | 244 | 1539 | 4083 | 1671 | 495 |
| vLLM 0.17.1 | 270 | 1842 | 4890 | 1885 | 553 |
| vLLM 0.17.1 batch-invariant | 80 | 609 | 2228 | 921 | 223 |

The relevant comparison is with engines that are not deterministic, since that is the price
of determinism. PI is within −2.8% to +10.4% of the current vLLM release without determinism
(behind it on three workloads by 1.2–2.8%) and ahead of SGLang on all five. SGLang's
deterministic mode, which disables the radix cache and fixes its split sizes, runs at 29–46%
of SGLang's own throughput here; we list it because it is the deterministic mode a user can
run on this GPU, not as the bar. vLLM 0.17.1's batch-invariant mode is the one baseline
that passes every detcheck check here except the seeded re-prefill variant, which tests its
sampler's seeding rather than its arithmetic (Section 7.1, ‖). It runs at 30–49% of vLLM 0.17.1's
own throughput, and PI at 2.0–3.7× of it. It replaces aten matmul and reductions with
batch-invariant Triton kernels and runs attention unsplit, outside its documented
compute-capability range; on Hopper, its supported target, the gap may differ.

**Kernels** (`results/a800/kernels/`). Attention is timed kernel against kernel
(`kernels/bench_attention.py`): each measurement replays a CUDA graph of at least ten calls,
each over another layer's KV cache from a ring larger than L2, as consecutive layers run in a
forward, so neither launch overhead nor a warm L2 enters. FlashInfer is planned for each exact
shape.

- **Decode attention:** at batch 8–32, 0.91–1.06× the time of the faster of FlashInfer's
  CUDA-core and tensor-core decode; with 8 or 32 requests sharing a 4096-token prefix, 0.73–0.85×;
  at batch 128, 1.05–1.10× FlashInfer's tensor-core decode. At batch 1, 1.03–1.10× at 8K–32K
  keys and 1.92× at 1024 keys (24.8 vs 12.9 µs): two groups give two CTAs per KV head, each
  running its 16 tiles in order, where FlashInfer splits the keys more finely. With
  FlashInfer's split size fixed at 2048 (the batch-invariant configuration), PI is 5.2× faster
  at batch 1 and 32K keys (95.3 vs 500.0 µs). In the engine at batch 1, with contexts below
  one group, PI's decode kernel takes 12.9 µs per layer on 8B against 30.7 µs for
  FlashInfer's decode and merge kernels (Section 7.5's trace).
- **Prefill and extend:** 1.12–1.80× FlashInfer on 17 of 18 shapes (and 0.55× on 512-token
  prompts with 16 query heads, where FlashInfer's time exceeds its own 32-head time): long
  single prompts 1.22–1.34×, batched short prompts 1.42–1.52×, extends over 4K–16K prefixes
  1.22–1.80×.
- **Projections** (`gemm_burst.txt`: 64 back-to-back launches captured in a graph, each over
  another weight copy so every launch streams from DRAM; the best PI kernel for each of the
  20 Qwen3 projections, LM heads included, against cuBLAS):
  - M = 1: 0.78–1.37×; only 0.6B o (1.37×) and down (1.21×) exceed 1.05×;
  - M = 8–64: 0.43–1.26×, above 1.05× on 17 of 80, all o and down projections;
  - M = 128–512: 0.81–1.33×, above 1.05× on 18 of 60;
  - M = 1024–2048: 0.76–1.22×, above 1.05× on 7 of 40.

  A decode step's projections replayed as one graph (`gemm_step_graph.txt`, 0.6B) take
  1.11× cuBLAS's time at M = 1, 0.77× at M = 8 and 1.03× at M = 32.

Kernel numbers for attention in earlier versions of this paper, and the RTX 4090 D numbers
quoted from them, timed one launch at a time and included host launch overhead, which hides
short kernels' differences; they are superseded by the numbers above.

### 7.5 Where the cost goes

**Batch 1.** A node-level trace of greedy decode at batch 1 (`scripts/nsys_decode.sh`,
`scripts/nsys_compare.py`, `results/a800/nsys/b1_compare.txt`) splits each step by kernel
category. Tracing every graph node lengthens the gaps between kernels, so these numbers
attribute the difference; the step times are those of Section 7.4.

| per step, PI − original (µs) | 0.6B | 1.7B | 4B | 8B |
|---|---|---|---|---|
| projections | +49 | −123 | −13 | +58 |
| attention | −86 | −100 | −679 | −640 |
| other kernels | −36 | +6 | −187 | −17 |
| between kernels | −73 | −46 | −324 | −303 |
| traced step (original) | 2316 | 3632 | 7567 | 11400 |

Attention is faster on every model, by 86–679 µs per step: PI's decode is one kernel, whose
single-group requests finish in place, where the original engine runs FlashInfer's decode and
a merge per layer. On 4B and 8B the original engine also spends 303–324 µs more per step
between kernels; tracing inflates that part. PI's projections take at most 0.6% longer per step than
cuBLAS's on 1.7B–8B; on 1.7B the stream-K down projection is faster (25.94 → 20.71 µs per launch).

0.6B is the exception, and its loss is in the projections. Its two segmented projections, o
and down, take 9.25 and 10.66 µs per launch against cuBLAS's GEMV at 7.33 and 9.28. These
shapes need two segments per output, of 1024 and 1536 elements. Their 4 and 6 MB of weights
take only 2–3 µs to stream, so both kernels are latency-bound, and the gap is the round trip
of the two-segment fold or hand-off, which every scheme we measured pays (segment chain,
segment split, stream-K over segments and over tile pairs). The canonical program's two
segments make that round trip unavoidable. In the trace the 0.6B PI step is still shorter,
because of the time between kernels; untraced, 0.6B is 3.4% slower at batch 1.

**Why this GPU charges prefill.** The canonical program is the same on every GPU; what
changes is what it costs. An RTX 4090 D runs the fp32-accumulate MMA at half rate, so MMA
time dominates prefill and hides two costs the program imposes: the per-32-key online step,
a serial chain of shuffles and exp2 between the QK and P·V MMAs, and the ldmatrix traffic of
16-row warps, where each loaded K or V fragment feeds one MMA. On the A800 the MMA runs at full
rate. In the kernel before pipelining, MMA plus ldmatrix alone already took as long as
FlashInfer's whole kernel (32 query heads, 8192 queries: 3.13 vs 3.04 ms;
`archive/ceiling_probes/a800/prefill_ablation_flags.txt`). Pipelining S(t+1) behind tile t's
online step and, for long and batched prefills, 8-warp CTAs (Section 5) brought 30000-token
TTFT from +41.8% to +18.3%. 32-row warps would halve the ldmatrix traffic, but with the fp32
group totals they fit 4 warps per SM instead of 8; built, they ran 1.4–1.8× the shipped
kernel's time (`prefill_wide_warps.txt`).

**Batched decode.** At 1.7B and 8 requests (`archive/ceiling_probes/a800/nsys_c8_38c32da.txt`,
engine 38c32da), PI's kernels keep the GPU busy 332 µs longer per traced step; the wall step
is 213 µs longer, because PI spends less time between kernels.
- **Attention, 222 µs.** Each working CTA runs its group's 16 tiles in order, at 1.1–1.4 µs
  per tile, plus the fold. FlashInfer splits the keys more finely and merges the parts.
- **Projections, 155 µs, mostly o and down.** cuBLAS runs them as split-K with a reduction
  kernel (`splitKreduce`) and as K-sliced CTAs (`sliced1x2`). Both change each element's k
  order. The deterministic stream-K kernels of Section 5 recover part of it (1.7B down at
  M = 8: 22.7 µs; cuBLAS 21.2).

We attribute the M = 128–512 projection gaps to the same two cuBLAS techniques. One case
shows the mechanism: 1.7B down at M = 128 has 64 tiles of 64 × 64 for 108 SMs. Stream-K
balances that work but cannot parallelise an element's k chain, and smaller tiles lose more
on L2 traffic than they gain.

**Measured and not shipped** (`archive/ceiling_probes/a800/`, each checked bitwise): a
device-gated deep decode kernel, 20–40% faster when the cache fits in L2 but paying a second
launch per layer (~2.4 µs) for a ~2.8 µs saving; a deeper K/V ring, whose gain vanishes at
the engine's capture grids; warp-pair specialisation and 32-row prefill warps, both slower;
tail splits of the prefill planner on sm_80, all slower than one split. What remains is the
price of the contract as broken down above. Lowering it would mean changing the canonical
program itself, for example the 32-key tile or the 512-key group, and with it every bit.

### 7.6 RTX 4090 D (earlier engine)

The 4090 D ran the whole evaluation with engine fdc184b before the board became unavailable
(`results/`, outside `results/a800/`). The kernels added since (Section 5's pipelined
prefill, 8-warp CTAs, 64-deep tiles, stream-K over segments, the decode grid order, the
log-probability kernel) were not timed on it. Its determinism results match the A800's at
token level: PI passes every detcheck check on all four models and all eight server
configurations, where the original engine fails 13–14 of 32 comparisons in each, and
SGLang's deterministic mode diverges on 14 of 32 shared-prefix requests and on 1 of 24
decode-vs-re-prefill requests. End to end, PI relative to the original engine:

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | +2.0% | +8.5% | +4.3% | +5.6% | +5.8% |
| 1.7B | −1.4% | +5.7% | +4.9% | +3.4% | +3.9% |
| 4B | +0.1% | +5.9% | +6.1% | +3.2% | +3.0% |
| 8B | −0.6% | +1.5% | +3.3% | +0.6% | +0.9% |

TTFT was within +1.6% on every long-prompt workload, including 16K–30K-token prompts. The
contrast with the A800 is the subject of Section 7.5: with half-rate fp32-accumulate MMAs,
the canonical program's online step and fragment traffic are hidden behind the MMAs. On the
4090 D the component ablation placed the batched gain in the row-invariant GEMM, which leads
cuBLAS at M = 8–64 there, and the batch-1 cost (−0.6% to −1.4% on 1.7B and 8B) in it as well.

### 7.7 Can the tests fail?

A bitwise check that never fails proves little unless it can see a violation. We checked
that ours can (`kernels/test_mutants.py`, `results/a800/run_all/kernel_tests.log`):

- **Mutated attention kernels.** Each mutant is a copy of the shipped sources with one change
  to one computation path; the shipped files are not touched. The split-prefill merge folding
  groups in descending order is caught by the split check; prefill closing a fold group every
  256 keys instead of 512 by the decode-vs-prefill, split and head-shard checks. The decode
  merge folding with two roundings instead of one fma, a one-ulp fp32 difference, reached the
  bf16 output too rarely for the decode-vs-prefill check as first written, which probed a
  dozen positions per request: it passed. The check now decodes every position of every
  request (about 4,300 positions, two random batchings) and catches it (60 failures). This
  is the one change the mutation testing forced.
- **Controls on the GEMM checks.** cuBLAS run through the M-invariance check (row 0 alone
  against row 0 inside M rows) fails on 26 of 27 projections, and the fold check rejects a
  weight computed with one segment where its canonical count is two.
- **End to end.** detcheck now compares every token's log-probability bits, not only the
  tokens; the original engine, which matched the deterministic one on the prefix-cache check
  at token level, fails it at log-probability level (Section 7.1). The tensor-parallel test
  has its own negative control (Section 9).

Accuracy is asserted, not printed:
- **Attention.** Error against float64 must be within 1.05× (mean) and 1.25× (maximum) of
  FlashInfer's single-prefill kernel on the same inputs, over every GQA geometry and query
  scale of `test_matrix.py`. FlashInfer is the reference because it rounds P to bf16 as we do;
  torch SDPA on these inputs runs its math backend, which keeps P in fp32 and so sits near
  bf16 rounding of the exact result (`archive/ceiling_probes/a800/close/acc_probe.txt`). PI's
  mean error is below FlashInfer's in every case and its maximum error no larger (for 4096 keys: mean 8.1e-5 vs 8.3e-5 at query
  scale 1, 7.5e-4 vs 7.7e-4 at scale 6).
- **GEMM.** Every fp32 output element lies within the standard fp32 summation bound
  (K + S)·2⁻²³·Σ|x_k w_k| of the float64 product, and every bf16 output is exactly the
  rounding of the fp32 one, for every projection of the four models.

## 8 The same principle for linear attention: exact Gated DeltaNet decode

Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) interleave full-attention layers with Gated
DeltaNet (GDN) layers. GDN is trained with a *chunked* forward (fla's
`chunk_gated_delta_rule`, 64-token chunks: an intra-chunk triangular solve plus a state
carried between chunks) and served with a *recurrent* kernel that updates an fp32 state per
token. These are two floating-point programs, so the sampler's log-probabilities are not the
trainer's, and when speculative verification runs a third kernel they also depend on whether
speculation is on. Section 4's principle applies unchanged: fix one canonical program per
row and make every serving path execute it. For GDN the natural canonical program is the
trainer's own chunk program, and the serving engine must decode with it. Everything in this
section is in `gdn/` and `results/a800/gdn/` (A800; RTX 4090 D logs in `results/gdn/`).

**The premise.** fla's chunk program is row-causal and resumable bitwise
(`experiments/e1_premise.py`): truncating the sequence, resuming from the fp32 state at any
64-token boundary, replaying a partial chunk token by token and varlen packing all reproduce
the full forward's rows. fla's own recurrent kernel differs on every row (relative L2
4.7e-3). On Qwen3.5 0.8B and 4B with row-invariant dense ops, a GDN-exact replay makes
decode equal to the teacher-forced forward on all 724 scored positions, while the recurrent
decode flips the argmax at 0.8–1.2% of them (`e2_*_rowinv.log`). Once dense ops are row
invariant, GDN is the entire remaining gap.

**The kernel** (`exact_decode.py`). A decode step computes the current token's row of the
chunk program from the slot's state at the last 64-token boundary and the chunk's cached
rows; once a chunk's 64th row is accepted, a fold advances the boundary state with the chunk
kernel's own arithmetic. Two kernels split the work so that the state is read as a bf16 copy;
the fp32 state is not stored beside it but recovered exactly from the bf16 copy plus its
dropped 16 bits and one rounding bit (a lossless codec checked on every fp32 bit-pattern
class). Memory per slot and layer is 4.33 MiB against the recurrent state's 2 MiB. Per
layer and step on the Qwen3.5-4B shape the kernel takes 2.01× vLLM 0.17.1's GDN decode at
batch 1, 1.40× at 8, 0.99× at 32 and 0.79–0.85× at 64–256 (`bench_vllm_gdn.log`).

**Speculation is free and invisible.** A verification window of up to 16 tokens computes
more rows of the same chunk program, so its rows are the trainer's whatever the window size.
Rejected rows are overwritten by the next window, and a chunk is folded only after its 64th
row is accepted, so rollback needs no per-draft state copies; vLLM's speculative path keeps
1 + drafts fp32 states per request (10 MiB per layer for 4 drafts) and the window of 4 runs at
0.73–0.84× of its time at batch ≥ 32.

**In vLLM 0.17.1** (`gdn/vllm/exact_gdn.py`, installed by monkeypatching, state kept in vLLM's
own pool). Prefill runs fla 0.5.0's chunk program, decode and verification run the exact
kernel, and the dense projections run PI-Attn's GEMM through vLLM's batch-invariant hook
(`kernels/pi_vllm.py`). The self-check generates 128 greedy tokens for 16 prompts and then
scores every sequence in one prefill; it counts generated tokens whose decode-time fp32
log-probability equals the prefill-time one bitwise (`results/a800/gdn/vllm/selfcheck.log`):

| | stock GDN | exact GDN | stock GDN, batch-invariant dense | exact GDN, batch-invariant dense |
|---|---|---|---|---|
| Qwen3.5-0.8B | 274/2048 | 264/2048 | 311/2048 | 2048/2048 |
| Qwen3.5-4B | 120/2048 | 131/2048 | 181/2048 | 2048/2048 |

With both on, decode equals prefill on 2048/2048 tokens, with chunked prefill, prompts
ending on a chunk boundary, slot reuse and preemption, and in each of these configurations
n-gram or MTP drafting leaves the hash unchanged. The exception is MTP drafting with chunked
prefill: the decode hash is still the unspeculated run's, but vLLM's prompt
log-probabilities themselves change (decode equals them on 848/2048 tokens on 0.8B, and with
PI-Attn's GEMM on 0/2048 and 23/2048 on 0.8B and 4B; `check_mtp_prompt_logprobs.log`).
Under seeded sampling decode still equals prefill on 2048/2048 tokens; the sampled tokens,
and so the hash, differ with speculation. Throughput on Qwen3.5-4B (256 requests unless stated, `throughput.log`): the
bitwise configuration runs at 1.01× stock vLLM on 200/256-token requests, 1.08× with MTP
drafting, 1.03× at 8 requests, 0.98× at one and 0.96× on prefill-bound work. With vLLM's own
batch-invariant matmul in place of PI-Attn's GEMM the same configuration runs at 0.27× at one
request and 0.83× at 256: the dense projections, not GDN, are where batch invariance costs
in vLLM. The price of exactness is memory per request: the pool holds 2.0× fewer 472-token
requests than stock without speculation, 0.96× as many 32K-token ones, and more than stock
with speculation.

**Scope.** "Bitwise the trainer's" holds against fla 0.5.0's chunk forward with its autotune
pinned (`pin_fla_configs`: four of fla's autotune configurations, three of `l2norm_fwd_kernel` and one of `chunk_fwd_kernel_o`, are different programs) and the
same Triton version, on sm_80 and sm_89. The end-to-end trainer used here is HF with
row-invariant dense ops, not a training framework; vLLM vendors an older fla whose chunk
forward differs from 0.5.0 (`e13_vllm_vendored_fla.log`), so a vLLM prefill equals a
fla-0.5.0 trainer only when the engine is pointed at the trainer's fla. Prefix caching and
tensor parallelism are not handled in the vLLM integration.

## 9 Tensor-parallel degree, as a canonical choice

Everything above is invariant to how one GPU partitions the work. Tensor parallelism adds one
more partition: Megatron splits the K sum of the row-parallel projections (attention output,
MLP down) over the ranks and adds the partials with an all-reduce, so the summation order
depends on the degree t. We keep every projection column-parallel instead: a rank holds the
weight rows of its output columns over the whole K, the layer input is all-gathered first and
the output after. An all-gather copies bits, so every output element runs the program it runs at
t = 1, for any t, including degrees that do not divide the head or column counts
(`kernels/tp_invariance.md`).

What this costs is the chain floor. A narrow column shard keeps the whole K on few columns, and
one MMA chain advances a 16-wide k step about every 28 ns on the A800, so a 1024-column shard of
Llama-3-70B's down projection (t = 8) is bounded by its chains, not by weight bandwidth. The
segment count is the lever, and it has to be part of the canonical program: S = S(N, K, t_max)
grows with the largest degree a deployment declares, t_max, and every launch, on any shard,
passes the whole weight's S. Bits are then the same for every t ≤ t_max and beyond; two
deployments that declare different t_max compute different bits, and t_max = 1 keeps every
tested model's S as before.

Verified on one GPU with the ranks emulated (`results/a800/kernels/`): the column shards of seven
row-parallel shapes (Qwen3 4B/8B, Qwen3.5-4B, 70B o and down) concatenate to the whole-weight
result for t = 1–8 with every kernel id that runs the shard; Qwen3-8B with real weights gives
bitwise the t = 1 logits at t = 2, 3, 4 and 8, prefill and 8 decode steps, while Megatron
row-parallel o/down at t = 2 changes 77–87% of the logits (fp32 partials summed in fp32, so only
the order differs, still 79–93%). Per-rank GEMM time of the column shard against the Megatron
row shard of the same bytes is 1.02–1.26× at t = 2 for 70B, and 1.12–2.4× at t = 8, worst on
small models' narrow shards at M ≤ 32 (`tp_invariance.md` has the tables). The column layout
moves more bytes: four all-gathers per layer instead of two all-reduces, 1.63× the bytes per
token for Llama-3-70B at t = 8 (o_proj alone: equal bytes, down: 2.25×). We had no multi-GPU
machine, so the communication cost and end-to-end TP throughput are not measured.

## 10 Limitations

- **Scope of the guarantee.** Bitwise claims hold for one GPU model and one build (A1–A4).
  Across GPUs, an RTX 4090 D (sm_89) and an A800 (sm_80) gave identical bits for every PI
  kernel case and identical GSM8K generations (Section 7.2). That is a measurement, not a
  guarantee: the segment rule depends on shapes only, but the MMA instruction's internal
  accumulation is not specified across architectures, and Hopper's `wgmma` is a different
  instruction. One route to guaranteed cross-architecture identity is explicit IEEE fp32
  accumulation, as in [2609.25624]; we have not measured its cost for this program. The
  cross-GPU identity also rules out tuning the canonical program per GPU: the segment rule
  and the fold groups must stay the same everywhere. The rollout equality of Section 7.3 is
  likewise per architecture: the log-softmax uses `ex2`/`lg2.approx`.
- **Engine features.** No tensor parallelism in the engine (Section 9 verifies the
  projection program on one GPU with emulated ranks); head_dim 128 and GQA groups ≤ 16 only;
  FP8 and MoE are not covered. Log-probabilities are returned for the sampled token only
  (`logprobs: 0`), not for top-k alternatives, and not on streaming or chat requests.
- **Trainer equality.** The trainer of Section 7.3 is the engine's forward in PyTorch,
  forward only; a training framework would back-propagate through its own differentiable copy.
  We have not run an RL training loop on it, and the GDN result of Section 8 is against HF with
  row-invariant dense ops, not a training framework.
- **Residual cost on the A800** (Section 7.4):
  - Prefill attention takes 1.12–1.80× FlashInfer's kernel time. Time to first token on
    16K–30K-token prompts is 12–18% higher, and throughput on a 30000-token prompt 9.4% lower;
    prefill-bound workloads lose up to 5.8% (8B 4096/4 c4), and 8.0% on 0.6B with 2048-token
    prompts at 64 requests, whose time to first token at 128 requests rises 18.9%.
  - Decode attention at batch 1 with 1024 keys takes 1.92× FlashInfer's kernel time; 1.7B
    with 8 requests is 5.8% slower end to end.
  - 0.6B at batch 1 is 3.4% slower, because its segmented o and down projections take
    1.21–1.37× cuBLAS's time at M = 1.
  - Projections at M = 8–512 are up to 1.33× cuBLAS, mostly on o and down shapes.

  These gaps come from the per-32-key online step, from the 16-tile chain of a decode group,
  and from cuBLAS's split-K and K-slicing, which the canonical k order rules out. Closing them
  would mean changing the canonical program, and with it every bit. On the RTX 4090 D, whose
  half-rate fp32-accumulate MMAs hide the first two, the earlier engine paid at most 1.8% of throughput end
  to end (1.7B, 64/256 c1, `results/e2e_extended/SUMMARY.md`; Section 7.6).
- **Single stream.** The GEMM segment, stream-K and segment-parallel workspaces and counters
  and the decode fold counters are static per-device state that every launch leaves zeroed.
  Launches that use them must not run concurrently, as the engine's single forward stream
  guarantees. A launch aborted mid-kernel (e.g. by a device fault) can leave a counter set,
  which would make the next launch fold early. The process must then be restarted, which any
  sticky CUDA error forces anyway. Grown CUTLASS scratch buffers are kept for the process's
  lifetime, because captured graphs still launch into them.
- **Baseline coverage.** vLLM's batch invariance is outside its documented support
  (compute capability ≥ 9.0) on both GPUs; version 0.17.1 is nonetheless deterministic on
  the A800, and its cost there need not be its cost on Hopper. SGLang can only use its
  FlashInfer path, because its Triton deterministic backend stalls. Hopper numbers, and the cost of
  guaranteed cross-architecture identity there, are future work. Each test box had one GPU,
  so tensor-parallel communication cost is not measured.

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
