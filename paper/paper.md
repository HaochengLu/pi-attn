---
title: "Fix the Program, Not the Schedule"
author: "Haocheng Lu"
date: "November 2026"
abstract: |
  Production LLM engines are not reproducible: the same request returns different tokens and log-probabilities depending on batch composition, prefix-cache state, chunking, and whether a token came from decode or prefill — and an RL trainer that rescores the rollout gets yet another answer. Today's deterministic modes buy reproducibility by switching optimizations off, and give up half their throughput or more doing so.

  We show that determinism does not have to cost throughput. Instead of constraining the *schedule*, we fix the *program*: every query row and every output element executes one canonical floating-point sequence that depends only on positions and shapes, chosen so that its independent parts are exactly the partitions fast kernels want. Attention becomes an online softmax over 32-key tiles inside absolute 512-key fold groups; projections become one fixed MMA chain per element, cut into shape-determined segments. Prefill, chunked prefill, prefix-cache hits and decode are then bitwise identical by construction, while tiling, pipelining, stream-K and even the tensor-parallel degree stay free.

  Implemented in a Rust serving engine with radix caching, chunked prefill, CUDA graphs and CPU/GPU overlap all left on, the deterministic mode is, on an A800, **2.7% faster on average** than the same engine without determinism across 40 workloads, **2.0–3.7× faster** than vLLM's batch-invariant mode — the only other engine that passes our numeric checks there — and on par with the latest vLLM release. On an H100, with Hopper kernels on `wgmma`, TMA, thread-block clusters and warp specialisation, it runs within −1.7% to +4.4% of its own non-deterministic throughput on the same-GPU baseline workloads, 1.8–3.1× faster than vLLM's batch-invariant mode and 1.5–1.9× faster than SGLang's deterministic mode. A PyTorch trainer on the same kernels reproduces all 55,593 rollout log-probabilities bit for bit, making every importance ratio exactly 1. Kernel outputs are bitwise identical across every Ampere and Ada GPU we ran — A800, A100, A10G, L4 and RTX 4090 D — and so are engine outputs on the A10G and RTX 4090 D; an H100 keeps every guarantee at its own bits, and Hopper kernels on `wgmma` and TMA reproduce those bits at near-cuBLAS GEMM speed. Exact-lattice variants of the program give one set of bits on Ampere, Ada and Hopper alike, at a measured price. The principle carries over to linear attention: Gated DeltaNet decodes with the trainer's own chunked program inside vLLM, at stock throughput.
---

![**Determinism without the tax** (A800). (a) Output throughput of three deterministic modes, each relative to its own engine without determinism, on Qwen3-1.7B: PI-Attn keeps full speed, while the batch-invariant modes of vLLM 0.17.1 and SGLang 0.5.5 keep 29–49%. (b) Rollout tokens whose trainer-side importance ratio deviates from 1 by more than 1%: none with PI-Attn's trainer forward, 10–22% with a standard bf16 trainer. (c) GSM8K generations bitwise identical across two GPU architectures.](figs/fig0_teaser.pdf){width=100%}

# Introduction

Ask a serving engine the same question twice at temperature 0 and you may get two different answers. Floating-point addition is not associative, and an engine re-partitions its reductions constantly: batch size changes GEMM tiling and split-K, load changes the number of KV splits in decode attention, chunked prefill moves where a prompt's attention is cut, a prefix-cache hit turns prefill into a cache read, and a token produced by the decode kernel is recomputed by the prefill kernel when the conversation continues. For evaluation this makes a benchmark score a sample rather than a value. For RL it means the rollout engine and the trainer disagree on every log-probability, and importance ratios drift from 1 on tokens the policy never changed.

The established fix is *batch-invariant* kernels [@tml25]: pin one reduction order regardless of batch size by fixing or removing KV splits and pinning GEMM configurations. It works, at a steep price, because the optimizations it removes are exactly the ones that adapt to load. SGLang reports a 34% slowdown on H100 [@sgl25]; on our A800 the batch-invariant modes of vLLM and SGLang keep 29–49% of their own throughput (Figure 1a). And prefill still disagrees with decode, because a prefill kernel rounds probabilities against a running maximum over the whole history while a decode kernel rounds against the maximum of its split.

**Our insight: determinism is a property of a program, not of a schedule.** Fix, for each query row and each output element, one canonical sequence of floating-point operations that depends only on positions and shapes. Any kernel that executes exactly that sequence produces the same bits, however rows are batched and whichever kernel runs them, and the sequence still leaves the schedule free wherever it has independent parts. The real question is therefore *which* sequence to fix — and that choice decides how fast deterministic kernels can be. We choose sequences whose free partitions are precisely the ones fast kernels already use:

- **Attention:** keys are cut at absolute positions into 512-key fold groups, each an online softmax over 32-key tiles, folded in a fixed order. A prefill kernel is a FlashAttention-2 kernel [@dao23] that restarts its state every 512 keys; a decode kernel gives every group its own CTA; chunked prefill, cache hits and re-prefill execute the identical per-row sequence.
- **Projections:** each output element runs one MMA chain over $K$, cut into a number of segments fixed by the weight's shape. Tile shapes never enter the chain, so an autotuner can pick any tile, pipeline or stream-K split without changing a bit, and the segments are a deterministic split-K exactly where decode needs one.

The result is a serving engine in which every optimization stays on and determinism costs no throughput. We make six contributions:

1. **Canonical programs** (§2): serving determinism as a per-row program with free partitions at its associativity points, with invariance lemmas and an explicit numerical contract stating which parameters define the bits.
2. **PI-Attn** (§3): Ampere/Ada attention kernels for prefill, extend and decode that share one program, and a family of row-invariant GEMMs — streaming, CUTLASS, deterministic stream-K and segment-parallel kernels — that are bitwise identical to each other.
3. **A deterministic serving engine** (§4, §5) that is 2.7% faster on average than its non-deterministic self, 2.0–3.7× faster than the only other engine that passes our checks on an A800 and 1.5–3.1× faster than both that pass them on an H100, and bitwise reproducible across load, cache state, configuration, restarts and GPU architectures.
4. **Rollout equals trainer** (§5.2): a PyTorch trainer forward on the same kernels reproduces every rollout log-probability bit for bit, whatever its micro-batch packing.
5. **Beyond softmax attention and one GPU** (§6): exact Gated DeltaNet decode inside vLLM at stock throughput with speculative decoding for free, and tensor-parallel degree invariance by making the declared maximum degree part of the program.
6. **Across GPU generations** (§7): calibrated tensor-core models that split GPUs into two classes, the same bits on five class-A GPUs, a Hopper port at the H100's own bits, and exact-lattice programs that are identical on every class.

# Canonical programs

**Guarantee.** For a fixed tensor-core class (§7), engine build and model weights, every response is a bitwise function of the request's tokens, sampling parameters and seed. Batch composition, concurrency, cache state, chunking, decode versus prefill, CUDA graphs, page size and restarts do not enter; with a declared $t_{\max}$, neither does the tensor-parallel degree of the projections (§6.2). We rely on four hardware properties: (A1) an MMA instruction is a deterministic function of its operands and accumulator; (A2) `ex2.approx` and `lg2.approx` are deterministic; (A3) the compiler does not reassociate arithmetic written with round-to-nearest intrinsics; (A4) an MMA with an accumulator other than $-0$ does not return $-0$.

![The canonical attention program. Keys are cut at absolute positions into 32-key tiles and 512-key fold groups. Each group runs 16 online-softmax steps from the empty state; group partials fold in ascending order. Prefill, per-group decode, and any chunked or cached split all execute this same per-row sequence.](figs/fig1_attention_program.pdf){width=100%}

## Attention

For query row $i$ at position $p$, scores $s_j = (q_i\cdot k_j)\cdot\mathrm{scale}\cdot\log_2 e$ are computed by MMA over the head dimension in ascending 16-wide steps and masked to $-\infty$ for $j>p$. Each fold group starts from the empty state $(m,d,o)=(-\infty,0,0)$ and runs, for its 16 tiles in ascending order,
$$
m' = \max\big(m,\max_j s_j\big),\quad \sigma = 2^{m-m'},\quad P_j = \mathrm{bf16}\big(2^{s_j-m'}\big),\quad d \leftarrow \mathrm{fma}\big(d,\sigma,\textstyle\sum_j P_j\big),\quad o \leftarrow \mathrm{fma}(o,\sigma,+0) + PV,
$$
with the row sum in one fixed order (and $m'$ replaced by 0 while it is $-\infty$). Group partials fold in ascending order into $(M,L,O)$: $M'=\max(M,m_g)$, $L\leftarrow\mathrm{fma}(L,2^{M-M'},d_g 2^{m_g-M'})$, likewise for $O$, and the output is $O/L$ (Figure 2). Two details make the program well defined on every path. A tile entirely past the query's position is an exact no-op (all $P=0$, $\sigma=1$, and the $+0$ rescale with A4 keeps $o$ off $-0$), so prefill rows may sweep masked tiles that decode never sees. And folding into the identity is exact, so a kernel may keep the first group in registers or read it from memory.

**Lemma 1.** If a kernel executes this program for every row it owns, the output of row $i$ is identical whatever rows share the launch, however groups map to CTAs, wherever chunked prefill cuts the query range, whether keys come from the prefix cache, and whether the row is decoded or prefilled. *Proof.* Every operation applied to row $i$, and every operand it reads, is fixed by $p$ and the keys and values at positions $\le p$. A kernel only chooses which rows, groups and tile slices it co-schedules, and splitting a tile across warps partitions $S$ and $O$ by columns without changing any element's chain. $\square$

The 512-key group is the sweet spot: long enough that prefill runs an unmodified FlashAttention-2 inner loop, short enough that decode has 1–64 CTAs per KV head even at batch 1 for 0.5K–32K contexts. Among 512, 1024 and 2048 keys it was fastest for decode at batch $\le 8$, for shared prefixes and for short extends, at no cost to long prefill.

## GEMM

![The canonical GEMM program. (a) Each output element runs one MMA chain per $K$ segment, ascending from zero, and the segments fold from $+0$ in order; $S$ depends only on the weight shape and the declared $t_{\max}$. (b) Everything else is free for the kernel and the autotuner.](figs/fig2_gemm_program.pdf){width=100%}

$K$ is cut into $S=S(N,K,t_{\max})$ equal segments; segment $s$ accumulates its 16-wide $k$ steps in ascending order by MMA from zero into $p_s$, and $y = \mathrm{round}\big((\cdots((+0+p_0)+p_1)\cdots)+p_{S-1}\big)$ (Figure 3). With $t_{\max}=1$, $S=2$ for $N\le1024$, $K\ge2048$, $K\equiv0 \pmod{512}$ — Qwen3-0.6B's o and down projections — and $S=1$ otherwise.

**Lemma 2.** Any tiling of $M$ and $N$, any assignment of (tile, segment) pairs to CTAs or warps, any choice of which CTA folds, and any cut of a segment's chain between workers that hands on the fp32 accumulator yield identical rows.

Segments are *deterministic split-K*. Split-K is nondeterministic in existing engines because the split count follows $M$ and the SM count and partials combine in arrival order. Fixing $S$ by the weight and the fold by segment index removes both, and spends segments only where decode needs them: narrow projections with long $K$.

## The numerical contract

| fixed: defines the bits | free: chosen per launch |
|---|---|
| m16n8k16 MMA semantics, bf16 in, fp32 accumulate | `mma.sync` or `wgmma`; which factor takes the 16-row operand; tile shape, warps per CTA, CTA count |
| ascending 16-wide $k$ steps per segment, from zero | $k$-stage depth, pipeline and `cp.async` ring sizes |
| $S(N,K,t_{\max})$ and the left fold from $+0$ | which CTA or warp runs or folds a segment |
| 32-key tiles, 512-key groups, the fold form | stream-K cut points (fp32 accumulator handed on) |
| explicitly rounded softmax, fold and log-softmax | rows per CTA, group-to-CTA map, split count |
| declared tensor-parallel maximum $t_{\max}$ | actual tensor-parallel degree $t$ |

: The numerical contract. The left column is part of the model's numerical definition, like its dtype; nothing in the right column can change a bit.

**$t_{\max}$ belongs to the contract.** Segments exist because narrow weights need them, and a column shard at large $t$ is as narrow as a small model's projection (§6.2), so $S$ depends on the declared maximum degree. Two deployments that declare different $t_{\max}$ compute different bits, exactly as two dtypes would; one that fixes $t_{\max}$ gets identical bits at every degree. A launch never derives $S$ from its local shard: every launch receives the whole weight's $S$. The engine fixes $t_{\max}=1$ and exposes no knob to change it.

**Every kernel runs the same instruction-level program.** In the descriptor view of Yang et al. [@taming], a GEMM's bits are set by its accumulating instruction, the order and grouping of its $k$ steps, its $K$ remainder, and the precision and order of any merge. Our kernels fix each of these by construction. Hand-written kernels issue the single PTX instruction `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, and the CUTLASS tiles are instantiated with instruction shape `GemmShape<16, 8, 16>`, which lowers to the same instruction. Every kernel refuses a launch whose segment length is not a multiple of its $k$ tile, so no remainder exists. Chains stay in fp32 registers, stream-K stores and reloads the fp32 accumulator bit-exactly, segment folds are explicit fp32 additions in ascending order, and nothing is built with fast-math. The instruction is also symmetric in its operands: $D_{ij}$ of $(A{=}X, B{=}Y, C)$ equals $D'_{ji}$ of $(A{=}Y, B{=}X, C^\top)$ bit for bit, on 1.69 × 10⁹ elements on the A800 and 6.4 × 10⁶ on the H100, special values included, as the tensor-core models of §7 predict (they group a step's products by $k$ alone). Which factor takes the 16-row operand is therefore a kernel's choice. The test suite checks every kernel against two references on every output row of every projection shape at 14 values of $M$, and mutants prove that the check bites (§5.4).

## Composition

RMSNorm, q/k-norm, RoPE, SiLU-and-mul and the KV store are row-wise or elementwise with fixed in-row orders. The sampled token's log-probability is a new row reduction, so we fix it explicitly: one 1024-thread CTA per row, strided per-thread passes for the maximum and for the sum of $2^{(z-m)\log_2 e}$, a fixed shared-memory tree, and only explicitly rounded operations, so a trainer that links the same header computes the same value. Sampling draws from per-row Philox streams keyed by (seed, position).

**Theorem.** In deterministic mode every activation, KV entry, logit and log-probability of token $t$ is a function of tokens $0..t$ only, and every sampled token a function of tokens $0..t$ and the seed.

# PI-Attn kernels

All kernels target sm_80/sm_89 and share one header holding the online step, the folds (built from `__fmaf_rn`, `__fmul_rn` and `ex2.approx`) and the finish. Kernels choose how work maps to warps and CTAs; none changes an element's sequence. Appendix B has the details.

**Prefill and extend** follow FlashInfer's scaffold [@ye25]: 4 warps × 16 query rows per CTA covering one KV head's GQA group, 128-byte-swizzled shared-memory tiles, and a software pipeline that computes $QK_{t+1}^\top$ while tile $t$ runs its online step and $PV$. Each warp folds finished groups into a second register state. On sm_80 a variant keeps Q in registers and runs 8 warps per CTA for long and batched prefills (5–8% faster), and the grid runs the longest row tiles first whenever the launch's K and V fit in L2 (up to 24% less time; Appendix B).

**Decode** has two kernels with one program. The first gives every (request, KV head, group) a 4-warp CTA: warps split each tile's scores and output columns; a request with one group finishes in place, otherwise the last warp to arrive folds every group in ascending order. The second, on sm_80, runs a whole group on a 16-warp CTA that takes an SM, because the program serialises little of a group: $S$ of a key depends on that key alone, the running maximum is a prefix maximum of exact tile maxima, and only the denominator and each output column are chains. It also swaps the MMA's operands. Decode has only a few query rows per KV head (2 or 4 on Qwen3), which the first kernel pads to the MMA's 16 rows; with the keys in the 16-row operand and up to 8 query rows in the 8 columns, $S$ and $PV$ take half the MMAs, and each element's sequence is unchanged. The engine picks the kernel per step from the longest running request, which the host knows, and captures one CUDA graph per kernel. Each grid depends only on batch size and page-table width, so replay needs no host planning.

**The row-invariant GEMM family** realises Lemma 2 in five kinds of kernel, all bitwise identical, among which an autotuner picks per $(N,K,M)$: small-M streaming kernels with deep `cp.async` pipelines; balanced kernels that split (tile, segment) work evenly over the SMs; CUTLASS tiles from 32×64 to 128×256 with 32- or 64-deep stages; segment-parallel and segment-split kernels for segmented shapes; and **deterministic stream-K**. Stream-K divides the (tile, $k$-step) pairs of all tiles evenly over the SMs. When a tile's chain is cut between workers $w$ and $w+1$, $w$ stores its fp32 accumulator and $w+1$ continues *the same chain* from it, so the result is bit-identical to the uncut chain: split-K's load balance without split-K's reduction. Each worker runs its handed-off prefix first and waits only on its lower neighbour's prefix, which never waits, so with all CTAs resident it cannot deadlock. Stream-K brings 1.7B's down projection at $M=8$ to 22.7 µs against cuBLAS's 21.2, and makes it faster than cuBLAS inside the batch-1 decode step (25.94 → 20.71 µs).

# A deterministic serving engine

We implement the mode in llm_infer_rust, a Rust/libtorch serving engine with a radix prefix cache, chunked prefill, mixed prefill/decode batches, decode and segmented-prefill CUDA graphs, and CPU/GPU overlap. `--deterministic` disables none of them. It routes attention to PI-Attn and every projection, including the LM head, to the row-invariant GEMM, and keys each sampled row's Philox stream by (request seed, position); requests without a seed get a random one, echoed back so any output can be replayed. With `logprobs: 0` a request also receives each sampled token's log-probability from the fixed-order kernel, without disturbing the overlap pipeline. At startup, after graph capture, the engine tunes every projection shape for every CUDA-graph batch bucket on random operands, with every launch streaming weights from DRAM as in a real forward (4.0–15.5 s for 0.6B–8B on the A800).

# Evaluation

**Setup.** One A800-SXM4-80GB (sm_80; CUDA 12.6, PyTorch 2.13, FlashInfer 0.6.18) at stock clocks, Qwen3 0.6B–8B in bf16. *Original* is llm_infer_rust with FlashInfer attention and cuBLAS; *PI* is the same binary with `--deterministic`; both run with radix caching, chunked prefill, mixed batches, CUDA graphs and overlap. Baselines on the same GPU: SGLang 0.5.5.post3 with `--enable-deterministic-inference`, vLLM 0.11.2, and vLLM 0.17.1 with `VLLM_BATCH_INVARIANT=1`. An RTX 4090 D (sm_89) provides the cross-GPU comparison.

## Bitwise reproducibility

We built *detcheck*, a black-box checker for any OpenAI-compatible server that compares each response's token ids and the fp32 bits of every token's log-probability across concurrency, shared-prefix batches, decode versus re-prefill, prefix-cache hits, seeded sampling and 200 repeats.

![detcheck on Qwen3-1.7B (A800): requests whose tokens or log-probabilities differ from the reference. *Decode vs re-prefill* resubmits the first $k$ generated tokens as prompt and compares the continuation; *repeat* counts distinct completions of one prompt over 200 runs. SGLang's deterministic mode disables the radix cache on this path, so it has no cache hits to compare.](figs/fig3_determinism.pdf){width=100%}

**PI-Attn passes every check** (Figure 4), on all four models. vLLM 0.17.1's batch-invariant mode also passes the numeric checks; its one failure, seeded re-prefill, comes from its sampler keying random streams by request start rather than by position. On the A800, SGLang's deterministic mode still changes log-probabilities with concurrency and across shared prefixes, and no engine besides PI and vLLM's batch-invariant mode makes decode agree with prefill; on the H100, SGLang's deterministic mode passes every check (§7). Log-probabilities are the stricter test: at token level the original engine looks consistent on decode versus re-prefill and on cache hits, but with log-probabilities all 24 re-prefilled requests differ. Across server configurations — chunk size 512, chunking off, CUDA graphs off, 4 running requests, overlap off, page size 32, and a plain restart — PI is identical every time on Qwen3-0.6B, while the original engine fails 11–13 of 32 comparisons in each, including the restart.

**Evaluation becomes a function of the model.** On GSM8K at client concurrency 1 versus 16, 888 of 1319 Qwen3-1.7B generations change with the original engine and its score moves by 0.68 points, more than the gap between the two modes; with PI, none change, on 1.7B and 8B alike. Accuracy is unchanged: top-1 agreement with an fp32 reference is 98.91–99.61% for PI and 99.14–99.53% for the original engine, differing by at most 5 positions per model, all at near-ties (Appendix A).

**Same bits on two GPU architectures.** Every PI kernel output in an 80-case GEMM and 5-case attention suite, and every PI GSM8K generation, is bitwise identical between the A800 (sm_80) and the RTX 4090 D (sm_89). cuBLAS agrees on 14 of 80 cases, FlashInfer on 1 of 5, and the original engine's generations differ on 906–910 of 1319 (Figure 1c).

## Rollout log-probabilities equal the trainer's

The trainer side is the engine's Qwen3 forward in PyTorch on the engine's own kernels: PI GEMM, PI-Attn causal prefill over a paged cache, RMSNorm, RoPE and the same log-softmax. We sample 64 GSM8K rollouts per model from the engine at concurrency 16 (temperature 1, per-request seeds) and rescore them with this trainer, each sequence alone and in two random packings of 2–8 sequences per micro-batch, and with HF transformers in bf16.

**Every one of the 55,593 rollout tokens gets exactly the engine's log-probability**, whichever packing the trainer uses, so every importance ratio is exactly 1 (Figure 1b). The engine produced those tokens by decode, in batches of up to 16 under chunked prefill; the trainer recomputes them in one prefill per micro-batch — Lemma 1 at work. The standard bf16 trainer matches on at most 41% of tokens and misses the 1% ratio band on 10–22%, with ratios from 0.49 to 2.51. Truncated and masked importance sampling exist to absorb exactly this noise; PI removes it at the source.

## Performance

![(a) End-to-end output throughput of PI relative to the original engine (greedy; median of three interleaved launches). (b) Change in time to first token with prompt length for the long-context workloads (each at its own concurrency).](figs/fig5_e2e.pdf){width=100%}

**Determinism costs no throughput end to end** (Figure 5a). Across 40 workload points — four models, five workloads, greedy and sampled — PI runs at −2.3% to +16.9% of the original engine, **+2.7% on average** (geometric mean). At batch 1 it is faster on 1.7B, 4B and 8B by 7.1–15.7% and level on 0.6B: PI's single-kernel decode replaces FlashInfer's decode-plus-merge, and stream-K beats cuBLAS on 1.7B's down projection. At 8 requests it is level on 1.7B (−0.5%, −1.4%) and 3.1–16.9% faster on the other models. The workloads at 16 and 32 requests and the 4096-token workload sit within −2.3% to +1.1%, except 0.6B's two long-prompt workloads (+4.7% to +5.2%). Ablating each half shows that the attention alone gains 4.2–7.2% at batch 1 and 0.8–3.8% at 8 requests and never costs more than 1.0%, while the GEMM ranges from −6.4% (0.6B at batch 1) to +15.2% (0.6B at 8 requests) and costs at most 1.4% at 32–128 requests (Appendix A). The small batches are where the 16-warp decode kernel runs. Before it and the longest-first prefill grid, the range was −5.9% to +14.2% and the mean +1.2%; an ablation at 8 requests on 1.7B, which our first evaluation had not run, then put 63% of that workload's 251 µs per token on attention.

**Against other engines** (Figure 1a, Appendix A), PI matches the latest vLLM release without determinism — within −2.8% to +10.4% of vLLM 0.17.1 — and beats SGLang on all five workloads. The deterministic alternatives pay heavily: vLLM 0.17.1's batch-invariant mode runs at 30–49% of vLLM's own throughput, so **PI is 2.0–3.7× faster** than the only other engine that passes our numeric checks; SGLang's deterministic mode, which also disables the radix cache, runs at 29–46% of SGLang's. vLLM documents its batch-invariant mode for compute capability ≥ 9.0, so §7 repeats the comparison on an H100. These baseline runs precede the 16-warp decode kernel, so PI's side of each ratio is conservative at 1 and 8 requests.

**Where PI pays: long prefills on full-rate tensor cores** (Figure 5b). Time to first token rises 11–18% on 16K–30K-token prompts and up to 5.6% on 2K–4K prompts, and prefill-heavy 0.6B at 64 requests loses 6.4% of throughput. The longest-first grid helps only while a launch's K and V fit in L2, about 10K keys of one request. The A800 runs fp32-accumulate MMA at full rate, which exposes the two costs the canonical program imposes on prefill: the serial per-32-key online step between the $QK^\top$ and $PV$ MMAs, and the fragment traffic of 16-row warps. On the RTX 4090 D, whose half-rate MMA hides both, the same design keeps time to first token within +1.6% on every long prompt and is faster than the original engine on 18 of 20 points (Appendix C). This is the price of one program for every GPU: retuning the tile or group size per GPU would change the bits.

**Kernels.** Timed kernel against kernel by replaying CUDA graphs over a KV ring larger than L2, PI's decode takes 0.68–1.21× the time of the faster FlashInfer decode wherever the planner takes the 16-warp kernel — 1 to 13 requests whose groups each get a CTA, where the 4-warp kernel alone took 0.86–2.10× — and beyond that 0.91–1.06× at batch 8–32, 0.74–0.85× with shared prefixes, and 1.04–1.09× at batch 1 with 8K–32K keys; against FlashInfer with its split size fixed at 2048, its batch-invariant configuration, PI is 5.2× faster at 32K keys. Prefill takes 0.49–0.93× FlashInfer's time on single requests of up to 2,048 tokens and 1.09–1.80× on the other 15 of 18 shapes. The best PI GEMM takes 0.78–1.37× cuBLAS's time at $M=1$ and 0.43–1.33× at larger $M$; the gaps are on o and down projections, where cuBLAS uses split-K, which the canonical $k$ order rules out (Appendix A).

## The tests can fail

A bitwise check that never fails proves little, so we mutate kernels and require the tests to catch every mutant: a merge folding groups in descending order, prefill closing groups every 256 keys instead of 512, and a decode merge rounding twice instead of once — a one-ulp fp32 change, which forced us to strengthen the decode-versus-prefill check to cover every position (it now fails 60 times). cuBLAS fails the $M$-invariance check on 26 of 27 projections, and the fold check rejects a wrong segment count. Accuracy is asserted rather than printed: attention error against float64 must stay within 1.05× (mean) and 1.25× (maximum) of FlashInfer's, and PI's mean error is below FlashInfer's on every geometry; every GEMM output lies within the fp32 summation bound of the float64 product.

# Beyond softmax attention and one GPU

## Exact Gated DeltaNet decode

Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) interleave attention with Gated DeltaNet (GDN) layers, which are *trained* with a chunked forward (fla's `chunk_gated_delta_rule`) and *served* with a recurrent kernel — two programs, so rollout and trainer disagree, and speculative decoding adds a third. The principle applies unchanged, with the trainer's chunk program as the canonical one. fla's chunk program is row-causal and resumable bitwise from any 64-token boundary state, so our decode kernel computes the current token's row of the chunk program from the last boundary state and the chunk's cached rows, folding the state once a chunk's 64th row is accepted. On Qwen3.5 with row-invariant dense ops, decode then equals the teacher-forced forward at all 724 scored positions, where recurrent decode flips the argmax at 0.8–1.2% of them.

**Speculation is free.** A verification window computes more rows of the same chunk program, and a chunk is folded only after its last row is accepted, so rollback needs no state copies, where vLLM keeps 1 + drafts fp32 states per request.

![Decode log-probability equals prefill log-probability, bitwise, in vLLM 0.17.1 (2048 generated tokens per model). Only exact GDN together with batch-invariant dense projections makes decode the trainer's program.](figs/fig9_gdn.pdf){width=100%}

**Inside vLLM 0.17.1**, with exact GDN and PI-Attn's GEMM as vLLM's batch-invariant dense ops, decode equals prefill on all 2048 generated tokens on Qwen3.5-0.8B and 4B (Figure 6), through chunked prefill, slot reuse, preemption and n-gram or MTP speculation, at **1.01× stock vLLM throughput** (1.08× with MTP, 0.96–1.03× elsewhere). The one exception is MTP combined with chunked prefill, where vLLM's own prompt log-probabilities change; the decode output is still the unspeculated run's. With vLLM's own batch-invariant matmul instead of PI's GEMM, the same configuration drops to 0.27× at batch 1: in vLLM, the cost of batch invariance lives in the dense projections, and PI's GEMM removes it.

## Tensor-parallel degree as a canonical parameter

Megatron-style tensor parallelism splits the $K$ sum of row-parallel projections across ranks and all-reduces the partials, so the bits depend on the degree $t$. We keep every projection column-parallel: each rank owns whole-$K$ output columns and all-gathers copy bits, so each output element runs its $t=1$ chain at any degree, including degrees that do not divide the head count. Narrow shards are bounded by chain latency rather than bandwidth, which is why $S$ grows with the declared $t_{\max}$ (§2.3). With ranks emulated on one GPU, Qwen3-8B reproduces its $t=1$ logits bitwise at $t=2,3,4,8$, while Megatron row-parallel o/down changes 77–87% of them; per-rank GEMM time is 1.02–1.26× the row shard's at $t=2$ for 70B shapes. Measuring the extra all-gather traffic (1.63× the bytes for Llama-3-70B at $t=8$) on a multi-GPU system is the next step.

# Across GPU generations

The contract fixes an instruction, `mma` m16n8k16 with fp32 accumulation, not its arithmetic. Two GPUs give the same bits exactly when their tensor cores evaluate that instruction identically, so the question is answered by bit-accurate models of MMA units [@mmasim; @hawkeye], which we re-fitted on the GPUs we use.

**Two tensor-core classes.** One fused group truncates the accumulator and every product toward zero to a multiple of $2^{E-F}$, where $E$ is the largest term exponent, sums them exactly and truncates the sum to 24 bits. *Class A* (A100 sm_80, A10G sm_86, L4 sm_89; the A800 and RTX 4090 D join it by kernel-hash equality) has $F=24$ and evaluates a $k16$ step as two chained groups of 8 products; *class H* (H100 sm_90) has $F=25$ and one group of 16. Our models reproduce 910,000 single steps per class (500,000 fitted, 410,000 fresh), NaN, infinities and subnormals included, and K=1024 chains, with no mismatch. Everything else the program uses agrees across classes: all 30 SFU, IEEE and libm operations we checked (`ex2.approx`, `lg2.approx`, `rsqrt.approx`, `div.rn`, `fmaf`, `expf`, …) return identical bits for all $2^{32}$ inputs (two-argument operations with one pseudo-random second operand per input) on A100, A10G, L4 and H100. The guarantee of §2 is therefore a guarantee per tensor-core class.

| GPU (class) | PI GEMM | PI attention | cuBLAS | FlashInfer | engine rollouts vs A800 |
|---|---|---|---|---|---|
| A100-80GB (A) | 108/108 | 5/5 | 108/108 | 5/5 | |
| A10G (A) | 108/108 | 5/5 | 52/108 | 0/5 | 64/64 sequences, 14,293/14,293 log-probs |
| L4 (A) | 108/108 | 5/5 | 39/108 | 0/5 | |
| RTX 4090 D (A)$^\dagger$ | 80/80 | 5/5 | 14/80 | 1/5 | 1319/1319 GSM8K generations |
| H100 (H) | 6/108 | 0/5 | | | 24/64 sequences, 4,377/14,293 log-probs |

: Kernel outputs bitwise equal to the A800's (108 GEMM cases: every Qwen3 0.6B–8B projection at several $M$, bf16 and fp32 output; 5 attention cases), and 64 seeded GSM8K rollouts of Qwen3-1.7B replayed by the engine. The A100, A10G, L4 and H100 pass all five kernel test suites; the A10G and H100 engines also pass detcheck. $^\dagger$Earlier 80-case suite and earlier engine; greedy GSM8K at concurrency 1 and 16 (Figure 8).

**Within a class, one program means one set of bits** (Table 2). On the A10G, the engine regenerates all 64 A800 rollouts token for token with every log-probability bitwise equal, and the trainer reproduces all 14,293 log-probabilities. Running on 99 KB-shared-memory GPUs also found a real bug: a CUTLASS tile that needs more shared memory than the device allows faulted instead of refusing, and a refused launch left a sticky CUDA error that the next launch misreported. Both are fixed: such tiles now refuse, and the tests model each tile's shared memory. On the H100 every guarantee of §2 holds within the GPU: detcheck passes and the trainer reproduces all 14,442 rollout log-probabilities. But its bits differ from class A wherever a $k16$ step truncates differently, as the models predict.

**Hopper at Hopper speed, same bits.** `wgmma` is bit-identical to `mma.sync` on the H100: 910,000 single steps and K=1024 chains, 0 mismatches. A Hopper kernel may therefore use TMA and warpgroup MMAs while executing the canonical program; the instruction moves to the free column of the contract. Feeding `wgmma` through `st.shared` raced in our probes, so operands must arrive by `cp.async` or TMA. Our sm_90 GEMM uses CUTLASS's TMA/GMMA mainloop with persistent CTAs, our own consumer loop (ascending $k16$ steps from $+0$, no split-K) and a TMA-store epilogue. Its tiles are free under the contract, so they are chosen for the H100 alone. At decode sizes, time follows the weight bytes in flight across the GPU, so narrow tiles keep deep weight stages and load exactly $M$ rows of $X$; a TMA box past the last row is slow. Between $M$ = 128 and 1,792 the tuner mostly picks 2×1 thread-block clusters, which share each weight stage between two row tiles through one TMA multicast. At $M$ = 2,304, a prefill chunk plus decode rows, 192×192 tiles fill the 132 SMs in one wave where 128×256 tiles need two. With 28 layers' weights streaming from DRAM, Qwen3-1.7B's four projections take 0.96–1.00× cuBLAS's time at $M\le64$, 1.01–1.11× at 128–512 and 1.04–1.08× at 1,024–8,192. All of these kernels reproduce the `mma.sync` kernels' H100 bits on 2,412 launches of every sm_90 tile over the 108 hash cases; the first sm_90 tiles also passed 1,516 checks over all Qwen3-0.6B and 1.7B projections, the 1.7B LM head, and 4B o and down, with $M$ from 1 to 4096, special values and $M$-invariance. Prefill runs PI-Attn's program in a warp-specialised kernel: a producer warp group feeds K and V through an mbarrier ring to two or three 64-row consumer warp groups that issue the `wgmma` chains asynchronously, over a longest-first grid. Decode adds a 16-warp kernel that runs a whole fold group per CTA, and the engine picks it, or the 4-warp kernel, per step from the longest request, with one CUDA graph per kernel. All of these are sm_90-only tuner candidates, so class-A builds are unchanged (A10G: 108/108 and 5/5 against the A800). On the H100 the suites, the 108 + 5 kernel hashes, the engine's 64 rollouts (14,442 log-probabilities) and detcheck are bitwise unchanged.

| Qwen3-1.7B, H100 | 64/32 c1 | 64/32 c8 | 64/32 c32 | 512/256 c8 | 2048/128 c8 | 1024/512 c32 |
|---|---|---|---|---|---|---|
| throughput, `mma.sync` kernels | −2.8% | −9.6% | −9.8% | −16.4% | −16.6% | −13.5% |
| throughput, first Hopper port | −1.7% | −2.2% | −4.8% | −11.8% | −7.6% | −5.3% |
| throughput, final Hopper kernels | +4.7% | +3.7% | +0.5% | −1.8% | −1.1% | −0.5% |
| TTFT, `mma.sync` kernels | +15% | +15% | −2% | +46% | +52% | +50% |
| TTFT, first Hopper port | +4% | +6% | +21% | +19% | +20% | +16% |
| TTFT, final Hopper kernels | +6% | −9% | +13% | +7% | +8% | +7% |

: PI relative to the same engine with FlashInfer and cuBLAS on an H100 (workload = input / output tokens / concurrency). Each row is from its own sandbox: FlashInfer and PI servers interleaved, a fresh server per pass, throughput is the median of 2 repeats (`mma.sync` and first-port rows) or 4 (final rows), and the final TTFT is the mean over both rounds. Across sandboxes FlashInfer mode varies by up to 4%, and 64/32 c32 is bimodal in both modes, so differences of a few points between rows, and its TTFT entries, are within noise.

**Where the H100 gap was.** The first port left PI 2–12% behind, and we had attributed that to attention. Running each half of PI alone in the engine showed otherwise: the projection GEMM caused at least half of the gap in 8 of 10 workloads by throughput (9 of 10 by time per output token). The causes were the weight bytes in flight up to $M$ = 512 and tile waves at $M$ = 2,304. A register-only probe ruled out the serial $K$ chain, which runs at 21 ns per $k16$ step. With the GEMM above and the new attention kernels, PI is within −1.8% to +4.7% of FlashInfer mode with up to 64 running requests (Table 3), except 2048/128 at 64 (−4.5%). At 128 and 256 running requests it is 4.3% and 8.0% below, and time to first token is 4–10% higher on most rows. What remains, and why:

- *Prefill attention* takes 1.14–1.49× FlashInfer's best kernel (FlashAttention-2 or -3) at 1,024 or more query rows, against 1.9–2.5× before; at short prompts after a long prefix it is 0.98–1.06×. Two attempts to overlap the softmax across warp groups, as FlashAttention-3 does, were slower. Most of the remainder is the contract: the online step runs once per 32 keys, so softmax and rescaling take 107 of 399 µs at 32 heads × 4,096 rows. By our estimate from instruction counts, a 128-key step would halve that, about 50 of the 79 µs gap; the other ~30 µs is unattributed. Skipping a rescale whose factor is exactly 1 would be bit-exact, because the online step's `fma(o, scale, +0)` never leaves $-0$ in $o$, but all 64 rows of a warp group must agree, which they rarely do.
- *Decode attention* at batch 1–4 and 640–2,176 keys, and batch 8 at 640, takes 1.4–1.7× FlashInfer (batch 1: 14.6–15.3 µs per layer, against 22 µs before). From batch 8 at 2,176 keys, and from batch 32 at any length, it is within 1.01–1.12×. Neither L2 residency nor deeper loads helped. A CTA's intake time grows linearly with the bytes it loads and makes up about 6 of the roughly 12.5 µs a 512-key group takes on one SM, where FlashInfer spreads the keys over many SMs. Every bit-exact attempt to spread a group lost more to hand-offs than it saved: two- and four-CTA clusters (0.45 µs per hand-off), a split into a K/softmax CTA and a V/$O$ CTA (1.3 µs fixed), and two-phase or two-launch kernels (all archived). So with this kernel the 512-key group size, a constant of the contract chosen on the A800, bounds short decode on the H100. One lever is untested there: `decode90_kernel` still pads a KV head's query rows to the MMA's 16, and on the A800 the same 16-warp program with the operands swapped (§3) took 0.69–0.84× the unswapped kernel's time, which brought short decode (320 keys and more, a CTA per group) from 1.02–1.67× FlashInfer to 0.80–1.14×.
- *GEMM* at $M$ = 192–1,024 stays at 1.06–1.11×. Tiling and clustering are free under the contract, and none of our probes changed a $K$ order; the next lever, a 3×1 cluster of 64-row tiles, is untested. This gap is implementation, not contract.
- *High concurrency* (128–256 running requests): we have no per-kernel trace on the H100 to divide the remaining 4–8% between mid-$M$ GEMM and decode attention.

**Against other engines on the H100** (Table 4), on vLLM's documented platform, the comparison of §5.3 holds with smaller factors.

- *Correctness.* SGLang 0.5.5's deterministic mode now passes every detcheck check, as does PI. vLLM 0.17.1's batch-invariant mode again fails only seeded re-prefill (8 of 24).
- *Cost of the alternatives.* vLLM's batch-invariant mode keeps 34–52% of vLLM's throughput, and SGLang's deterministic mode keeps 60–70% of SGLang's.
- *PI.* With the final Hopper kernels PI runs at 0.98–1.04× of its own engine's FlashInfer mode in the same sandbox (0.88–0.96× with the first port). It is 1.8–3.1× faster than vLLM's batch-invariant mode and 1.5–1.9× faster than SGLang's deterministic mode, and at 0.89–1.03× of vLLM 0.17.1 without determinism.

| H100, Qwen3-1.7B, greedy out tok/s | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust, FlashInfer | 443 | 3024 | 8645 | 3411 | 985 |
| PI, first Hopper port | 426 | 2661 | 8273 | 3239 | 899 |
| **PI, final Hopper kernels**$^\dagger$ | **455** | **2946** | **8624** | **3400** | **966** |
| SGLang 0.5.5 | 407 | 2758 | 7812 | 3296 | 956 |
| SGLang 0.5.5 deterministic | 243 | 1710 | 5446 | 2311 | 577 |
| vLLM 0.17.1 | 442 | 3066 | 8738 | 3648 | 1081 |
| vLLM 0.17.1 batch-invariant | 149 | 1168 | 4442 | 1889 | 467 |

: Same-GPU baselines on an H100 with `eval/baselines.sh` unchanged (median of three repeats in one launch per engine). PI, SGLang deterministic and vLLM batch-invariant all pass detcheck's numeric checks; vLLM's fails seeded re-prefill. $^\dagger$Measured in a later sandbox where FlashInfer mode ran 1–2% slower (436 / 2998 / 8537 / 3375 / 974), median of six repeats, so its ratios to the other rows are slightly conservative.

**One program for every class.** *Exact lattice.* If an MMA step's accumulator and all its products are integer multiples of one unit $u$ and $|c|+\sum|a_ib_i|\le2^{24}u$, every class returns the exact sum: for any $F\ge24$ no truncation fires, and every partial sum is representable. A program whose tensor-core steps all satisfy this, and whose roundings are explicitly ordered IEEE fp32 operations on CUDA cores, has the same bits on every class. We built four such programs and measured each on A100, L4 and H100, and all but I4 also on an A10 or A10G.

- *XC GEMM* rounds each operand per (row, $k16$ block) to a block lattice, splits it into slices and runs each slice pair as one MMA from zero, then folds the partials in fp32. The triangular variant needs 3 MMAs per $k16$ and is as accurate as the native chain (relative error 0.35–1.5×10⁻⁶ against 1.0–1.7×10⁻⁶). Its outputs are identical on all four GPUs, via `mma.sync` and `wgmma`, and equal a CPU reference.
- *I4 GEMM* writes each bf16 value as a 16-bit integer on a per-256-block exponent and multiplies it on the integer tensor cores in four int8 products, exactly. Its outputs are identical on L4, A100 and H100, and it is $M$-invariant. 0.67% of its bf16 outputs are not correctly rounded on Gaussian data (native: 0.12%), and 6.5–10% with outliers, wide exponents or cancellation (native: 0.01–0.13%). An untuned prototype takes 2.9–3.8× cuBLAS's time on the A100 and 3.9–6.4× on the H100 at $M\ge256$.
- *Portable decode GEMM.* XC-triangular with two of its three products chained inside the tensor core (still exact), slicing the weight in registers, streams each weight byte once. Its outputs are identical on L4, A10, A100 and H100 and equal a CPU reference on 11 cases × 6 program variants (adversarial data with $\pm0$, subnormals and $2^{\pm20}$ exponents included), at 11 values of $M$ from 1 to 64 and every scheduling. Its error before rounding is 0.34–1.7×10⁻⁶ against 1.3–8.1×10⁻⁶ for the native chain, lower on all but one case, and takes 1.26–1.40× the native PI kernel's time at $M\le16$ (geometric mean, weights streaming from DRAM). Rounding the weights once, offline, onto a 12-bit lattice per (column, $k16$) block — a changed checkpoint that moves bf16-output error from 1.653×10⁻³ to 1.654×10⁻³ on Gaussian data and at most from 1.659×10⁻³ to 1.685×10⁻³ — leaves two MMAs per $k16$ and brings this to 1.07–1.19×. At $M$ = 32–64 both cost 1.7–3.6×, and prefill-sized $M$ needs at least two MMAs per step and is not yet tuned.
- *XC attention* projects Q, K and V onto power-of-two lattices when they enter the cache, so every $QK^\top$ partial and every tile's $PV$ partial is exact. Its outputs are identical on all three GPUs in 14 of 14 cases, where stock PI-Attn agrees between L4 and H100 in 1 and FlashInfer in 1. Every invariance of Lemma 1 survives: 0 failures in about 8,700 checks per variant. With 32-wide QK blocks it costs 10–36% on prefill and 1–14% on decode, and raises the error against float64 from 1.95–2.03×10⁻³ to 2.19–2.29×10⁻³ on Gaussian data (bf16 floor 1.6×10⁻³).

The lattice is a declared projection: it changes the model function, slightly on realistic data, by 2–4% on adversarial rows with exponents spread over $2^{\pm10}$, and by 18–39% when 0.5% of elements are scaled by $2^{20}$. Cross-generation identity is therefore a choice in the contract, like the dtype: per-class bits at native speed, or one set of bits on both classes for a measured price. Contrary to the claim that bitwise methods cannot generalise across GPU generations [@heal], they can.

# Related work

**Batch-invariant kernels** [@tml25] underlie SGLang's deterministic inference [@sgl25] and vLLM's `VLLM_BATCH_INVARIANT`, which pin splits and GEMM configurations. DeepSeek-V4 [@dsv4] drops split-KV and most split-K and engineers a second attention kernel to match the single-SM order, and a vLLM pull request [@vllm62365] routes every row through the unsplit decode kernel at 0.84–0.91× throughput. These pin the partition or engineer one kernel to reproduce another; we leave the partition free and make prefill and decode agree by construction. **Deterministic attention** with fixed absolute splits appears in mlx-batch-invariant [@mlxbi] (prefill 3.9–4.4× slower than stock) and a llama.cpp draft [@lcpp]. Vosti [@vosti] verifies an ordered single-loop attention — our program with one unbounded group, which cannot split long decodes — and FoldAttention [@foldattn] declares the softmax reference ahead of the scan. Our contribution is the in-group program that FlashAttention-2 and group-parallel decode both execute as is, inside an engine with a prefix cache; FlashDecoding [@fd23] and FlashInfer [@ye25] merge split partials, but along schedule-dependent boundaries. **Across GPUs.** MMA-Sim [@mmasim] and Hawkeye [@hawkeye] model tensor-core arithmetic bit-exactly per generation, the latter to replay GPU results on a CPU; we use the same models to decide when two GPUs share bits and to design programs that are exact on all of them. HEAL [@heal] reduces drift with 16-bit integer QKV and error compensation, without bitwise identity, and argues that bitwise methods do not carry across generations. EigenAI [@eigenai] verifies deterministic inference by re-executing it on the same GPU model. VeXact [@vexact] uses a zero-mismatch reference to show that training–inference mismatch alone can destabilise RL, which PI's rollout-equals-trainer property removes at serving speed. **Bitwise GEMM.** Yang et al. [@taming] describe a GEMM's bitwise behaviour by a descriptor of its reduction order, show that output tiling leaves accumulation chains intact, reconstruct cuBLAS's descriptor, enforce orders in Triton and verify equivalence statically on PTX and AMDGCN. Our program is such a descriptor designed for serving: invariant to $M$ and to tensor-parallel sharding, realised by a whole kernel family, and extended to attention, which [@taming] treats only theoretically. Cooper et al. [@cooper] accumulate in explicit IEEE fp32 for cross-architecture identity; TBIK [@tbik] orders tensor-parallel all-reduces; CoRun [@corun] pads to fixed shapes; LLM-42 [@llm42] verifies and rolls back; reproducible summation [@dn13] makes sums order-free at higher cost; and a recent stream-K for convolution gradients [@rocm13151] folds partial tiles in a fixed order, where ours hands on the accumulator itself.

# Scope and conclusion

The native guarantee is per tensor-core class and build; the exact-lattice programs extend it to both measured classes at a measured cost, and are not yet in the engine. Blackwell (`tcgen05`) is untested; the lattice argument needs only exact sums for $F\ge24$. On the H100 the deterministic mode is within −1.8% to +4.7% of the engine's own throughput up to 64 running requests (except 2048/128 at 64, −4.5%) and 4–8% below at 128–256. Two of the remaining costs are the contract's own: short decode is bounded by one 512-key group's bytes per SM, and prefill by a 32-key online step. Since the H100's bits already differ from class A, a Hopper-specific group size would cost no shared bits, but it would make the program itself depend on the GPU. The engine runs on one GPU with head dimension 128; FP8, MoE kernels and a full RL training loop are natural next steps.

Bitwise-reproducible serving does not require turning optimizations off. It requires choosing the floating-point program first and letting every kernel, partition and engine feature execute it. With attention cut into absolute 32-key tiles and 512-key groups and projections into shape-determined segments, batching, chunking, caching, decode, packing and even the tensor-parallel degree become schedule choices that cannot change a bit, and a trainer running the same program sees exactly what the rollout saw. Determinism, it turns out, need not cost throughput.

# References {-}

::: {#refs}
:::

\clearpage

# Appendix A: Additional results {-}

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| original | 283 | 1894 | 5014 | 1878 | 539 |
| **PI** | **298** | **1790** | **4951** | **1863** | **542** |
| SGLang 0.5.5 | 253 | 1768 | 4865 | 1837 | 530 |
| SGLang deterministic | 73 | 554 | 1993 | 843 | 192 |
| vLLM 0.11.2 | 237 | 1599 | 4149 | 1694 | 508 |
| vLLM 0.11.2 batch-invariant | 244 | 1539 | 4083 | 1671 | 495 |
| vLLM 0.17.1 | 270 | 1842 | 4890 | 1885 | 553 |
| vLLM 0.17.1 batch-invariant | 80 | 609 | 2228 | 921 | 223 |

: Same-GPU baselines: greedy output tokens/s on Qwen3-1.7B (A800). The original and PI rows are a separate launch from Figure 5a, measured before the 16-warp decode kernel and the longest-first prefill grid; with them PI gains 2.3% at batch 1 and 5.5% at 8 requests and is within 1% elsewhere (Figure 5a's runs). vLLM 0.11.2's batch-invariant mode is not batch-invariant on sm_80.

| model | workload | original tok/s | PI attention only | row-invariant GEMM only | both | Δ time per token: attention, GEMM, both (µs) |
|---|---|---|---|---|---|---|
| 0.6B | 64 / 256 / c1 | 513.7 | +7.2% | −6.4% | +0.1% | −132, +135, −3 |
| 0.6B | 512 / 256 / c8 | 2587.6 | +1.2% | +15.2% | +16.7% | −42, −413, −453 |
| 0.6B | 512 / 256 / c32 | 7446.2 | +1.6% | −0.8% | +0.6% | −73, +33, −37 |
| 0.6B | 512 / 256 / c128 | 11607.2 | −0.6% | −1.1% | −1.4% | −92, +108, +63 |
| 1.7B | 64 / 256 / c1 | 282.7 | +4.2% | +3.8% | +8.2% | −145, −129, −272 |
| 1.7B | 512 / 256 / c8 | 1909.1 | +0.8% | −2.2% | −0.9% | −32, +95, +39 |
| 1.7B | 512 / 256 / c32 | 5072.5 | +1.3% | −1.4% | −0.6% | −93, +100, +37 |
| 1.7B | 2048 / 256 / c16 | 1901.5 | −0.4% | −0.1% | −0.6% | +66, −3, +48 |
| 1.7B | 4096 / 128 / c4 | 541.8 | −1.0% | +2.0% | +0.9% | +34, −110, −66 |
| 1.7B | 2048 / 128 / c8 | 1118.6 | −0.8% | +0.9% | −0.2% | +26, −33, −24 |
| 1.7B | 512 / 256 / c64 | 7033.7 | +1.5% | −1.0% | +0.2% | −157, +65, −76 |
| 8B | 64 / 256 / c1 | 89.9 | +5.4% | +1.4% | +7.0% | −577, −159, −735 |
| 8B | 512 / 256 / c8 | 630.9 | +3.8% | −1.0% | +3.4% | −475, +107, −449 |
| 8B | 512 / 256 / c32 | 1809.5 | +0.5% | +0.6% | +1.2% | −100, −128, −263 |

: Component ablation: output throughput relative to the original engine and the change in time per output token (workload = input / output tokens / concurrency). The two halves roughly add. Before the 16-warp decode kernel, attention alone cost 3.8% on 1.7B at 8 requests (159 µs per token) and gained 1.8% at batch 1.

| model | workload (in / out / concurrency) | Δ throughput | Δ TTFT |
|---|---|---|---|
| 1.7B | 64 / 256 / c1 | +8.3% | +6.5% |
| 1.7B | 512 / 256 / c32 | −0.2% | −1.1% |
| 1.7B | 2048 / 16 / c8 | −0.9% | −1.2% |
| 1.7B | 4096 / 4 / c4 | −2.2% | +2.5% |
| 1.7B | 4096 / 128 / c8, 3968-token shared prefix | +2.0% | +2.2% |
| 1.7B | 8192 / 64 / c4, 8000-token shared prefix | +1.7% | +12.1% |
| 1.7B | 16384 / 256 / c2 | −3.4% | +11.3% |
| 1.7B | 30000 / 128 / c1 | −9.1% | +17.7% |
| 8B | 64 / 256 / c1 | +7.1% | +3.1% |
| 8B | 512 / 256 / c32 | +1.2% | +3.3% |
| 8B | 2048 / 16 / c8 | −3.9% | +1.9% |
| 8B | 4096 / 4 / c4 | −5.2% | +5.6% |
| 8B | 4096 / 128 / c8, 3968-token shared prefix | +8.1% | +3.5% |
| 8B | 8192 / 64 / c4, 8000-token shared prefix | +5.6% | +6.4% |
| 8B | 16384 / 256 / c1 (prefill only) |  | +12.6% |
| 0.6B | 512 / 256 / c64 | +1.4% | +7.7% |
| 0.6B | 512 / 256 / c128 | −1.8% | +17.4% |
| 0.6B | 128 / 512 / c128 | +0.8% | −0.2% |
| 0.6B | 2048 / 128 / c64 | −6.4% | +8.6% |
| 1.7B | 512 / 256 / c64 | +0.8% | +1.9% |
| 1.7B | 512 / 256 / c128 | −0.1% | +1.4% |
| 1.7B | 128 / 512 / c128 | +0.8% | −2.5% |
| 1.7B | 2048 / 128 / c64 | −1.9% | +1.8% |
| 4B | 512 / 256 / c64 | +6.6% | +4.0% |
| 4B | 128 / 512 / c64 | +7.7% | +2.0% |

: Long-context, shared-prefix and high-concurrency workloads, PI vs. original (A800; median of three interleaved launches). Throughput is output tokens/s, or requests/s for the 16- and 4-token outputs. The 8B 16K row produces one token per request because 16384 + 256 exceeds the server's context limit.

| model | mode | strict acc. c=1 | c=16 | generations differing, c=1 vs c=16 | top-1 agreement with fp32 |
|---|---|---|---|---|---|
| Qwen3-1.7B (1319) | original | 69.37 | 68.69 | 888 | 99.14% |
| Qwen3-1.7B (1319) | PI | 69.07 | 69.07 | **0** | 99.30% |
| Qwen3-8B (500) | original | 89.80 | 89.80 | 322 | 99.53% |
| Qwen3-8B (500) | PI | 90.20 | 90.20 | **0** | 99.53% |

: GSM8K at two client concurrencies, and top-1 agreement with an fp32 HF reference over 1,280 teacher-forced positions (0.6B: 99.53% original, 99.61% PI; 4B: 99.30%, 98.91%). Every disagreement between the modes is at a position whose fp32 top-2 margin is below 0.17 logits.

![Rollout log-probabilities per model. (a) Tokens whose trainer log-probability equals the engine's bit for bit. (b) Tokens whose importance ratio deviates from 1 by more than 1% (HF: each sequence alone; batches of 8 give 10.6–22.1%).](figs/fig6_rollout.pdf){width=100%}

![Bitwise identity between an A800 and an RTX 4090 D: kernel cases and GSM8K generations (both client concurrencies; the original engine's range is over the two).](figs/fig8_xgpu.pdf){width=100%}

![Per-step time of PI minus the original engine at batch 1, by kernel category (node-level trace, taken with the 4-warp decode kernel). Attention is faster on every model, by 86–679 µs per step; projections cost 0.5% of a step on 8B and are faster on 1.7B and 4B. 0.6B's two segmented projections take 9.25 and 10.66 µs against cuBLAS's 7.33 and 9.28: their weights stream in 2–3 µs, so the round trip of the two-segment fold is the gap.](figs/fig7_breakdown.pdf){width=100%}

**Kernels in detail.** Decode, PI's time relative to the faster of FlashInfer's CUDA-core and tensor-core decode. Where the planner takes the 16-warp kernel (1–13 requests of 96–4,200 keys, a CTA per group or one short group more): 0.68–1.21× with 16 query heads and 0.58–1.22× with 32, against 0.86–2.10× and 0.72–2.10× for the 4-warp kernel there; at batch 1 and 1024 keys 14.7 µs against FlashInfer's 12.9, where the 4-warp kernel took 24.8. Elsewhere: 0.91–1.06× at batch 8–32; 0.74–0.85× with 8 or 32 requests sharing a 4096-token prefix; 1.06–1.10× at batch 128; 1.04–1.09× at batch 1 with 8K–32K keys. Just past one wave (one request at 7,168 keys, two at 3,584, seven or eight at 896–1,024) the 4-warp kernel still takes 1.15–1.28× FlashInfer's time. In the engine at batch 1, a node-level trace of the 4-warp kernel put PI's decode at 12.9 µs per layer on 8B against 30.7 µs for FlashInfer's decode and merge. Prefill and extend, PI's time: 0.49×, 0.88× and 0.93× FlashInfer's on single requests of 512–2,048 tokens and 1.09–1.80× on the other 15 of 18 shapes; with the ascending grid it was 1.12–1.80× on 17 of 18. Best PI GEMM against cuBLAS over the 20 Qwen3 projections, every launch streaming from DRAM: 0.78–1.37× at $M=1$ (only 0.6B o and down above 1.05×); 0.43–1.26× at $M$ = 8–64; 0.81–1.33× at $M$ = 128–512; 0.76–1.22× at $M$ = 1024–2048. A decode step's projections replayed as one graph take 1.11×, 0.77× and 1.03× cuBLAS's time at $M$ = 1, 8 and 32 on 0.6B.

# Appendix B: Kernel details {-}

**Prefill.** Q, K and V live in 128-byte-swizzled tiles — a 64-row Q tile plus two 32-key K and V slots, 48 KB, two CTAs per SM. K and V load a full iteration ahead behind one barrier per tile, and positions past the KV length are zero-filled without a global read. The sm_80 register-Q variant moves the fp32 group totals, touched once per 512 keys, to shared memory (80 KB); with 8 warps per CTA it copies each K/V tile once for twice the rows, is 5–8% faster on long and batched prefills and up to 12% slower on short extends, and the launcher takes it only when 128-row CTAs fill two waves. Sixteen-row warps are deliberate: 32-row warps need the group totals outside registers, which halves occupancy, and the built variant ran 1.4–1.8× the shipped kernel's time. Pipelining $QK_{t+1}^\top$ behind tile $t$ and 8-warp CTAs brought 30000-token TTFT from +41.8% to +18.3%. For long prefixes with few query rows, a wave-model planner splits a row tile's groups across CTAs and a merge kernel folds them. Grid order is scheduling too. A causal row tile runs as many key tiles as its position, so with row tiles ascending the last CTAs of a launch are its longest and run beside idle SMs; with the longest first, the short ones fill in behind. But longest-first separates the row tiles of a (request, KV head), which read the same K and V: once a launch's K and V exceed L2, every row-tile level streams them from DRAM again (3–9% slower), and split launches, whose CTAs all run equally long, lose up to 27%. The sm_80 launcher therefore orders longest-first only unsplit launches whose K and V fit in L2: 0.76–0.98× the ascending grid's time on 27 of the 29 shapes it reorders among 54 measured, and at most +1.6% on the other two.

**Decode.** Per tile, each warp computes an 8-key slice of $S$; the tile maximum is exchanged through shared memory; warp 0 recomputes $\sum P$ in the canonical order; each warp accumulates four of the sixteen output column tiles. In a multi-group request each warp writes its slice of the group partial and counts its arrival on a per-(request, KV head, warp) counter; the last arrival folds that slice over all groups. No CTA-wide barrier follows the arrival: an earlier version that parked the CTA on one atomic doubled kernel time under L2 pressure. The group is the slowest grid index, so working CTAs are dispatched before empty ones; on the A800 this cut Qwen3-1.7B's decode attention at 8 requests from 1001 to 842 µs per step.

**Decode on 16 warps (sm_80).** A CTA loads the group's sixteen K tiles in one `cp.async` group and V into the same slots once $S$ is done (138 KB of shared memory, one CTA per SM). Warp $t$ computes $S^\top = KQ^\top$ of tile $t$ in two MMA chains of 16 keys by 8 query rows and has the tile's exact maximum itself; one warp forms the running maxima, the scales and the denominator chain; four warps form the row sums of $P$ from shared memory in the canonical order; eight warps run the output chain, $O^\top = V^\top P^\top$, one accumulator per 16 dimensions. From 320 keys the unswapped 16-warp kernel took 0.70–0.96× the 4-warp kernel's time, and the swapped one 0.69–0.84× of the unswapped. A launch gives each (request, KV head) as many CTAs as fit one wave, and the planner takes the kernel when every group of the longest request has a CTA of its own (0.48–0.97× the 4-warp kernel's time at 96–6,656 keys) or one short group more (0.67–0.96×); with 7 or 8 requests, rows of helper CTAs on the otherwise idle SMs run those last groups. Beyond that the 16-warp kernel takes 0.97–1.28× the 4-warp kernel's time, and the 4-warp kernel keeps the launch. One trap: finding a CTA's (request, head, group) by integer division cost 0.2–0.7 µs per launch across 512 threads, so the grid is three-dimensional.

**GEMM.** Balanced kernels exist because decode GEMMs stream weights, and fixed 32-column tiles leave some SMs with twice the bytes of others. CUTLASS kernels on segmented shapes call the mainloop once per segment from a zero accumulator and fold before the stock epilogue; 64-deep stages took several $M=128$–512 projections from 1.1–1.3× cuBLAS's time to 0.85–1.05×. Stream-K runs over one worker warp per (SM, warp slot) for $M\le8$, and at CTA granularity over CUTLASS tiles for mid $M$, handing the accumulator fragments from CTA to CTA and refusing to launch unless all CTAs can be resident. The segment-parallel kernel gives each (tile, segment) its own CTA, and the last arrival adds the fragments to $+0$ in ascending order; the segment-split kernel at $M\le8$ gives each segment a warp with its own `cp.async` ring and folds through shared memory. The tuner times bursts of graph-captured launches over a ring of weight copies larger than L2, because a microsecond kernel timed after an L2 flush, or launched from Python, is misranked.

# Appendix C: RTX 4090 D {-}

The 4090 D ran the full evaluation with an earlier engine. PI passed every detcheck check, at token level, on all four models and all eight server configurations, where the original engine failed 13–14 of 32 comparisons in each. End to end, PI was faster than the original engine on 18 of 20 greedy points and kept time to first token within +1.6% on every long-prompt workload, 16K–30K tokens included.

| model | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B | +2.0% | +8.5% | +4.3% | +5.6% | +5.8% |
| 1.7B | −1.4% | +5.7% | +4.9% | +3.4% | +3.9% |
| 4B | +0.1% | +5.9% | +6.1% | +3.2% | +3.0% |
| 8B | −0.6% | +1.5% | +3.3% | +0.6% | +0.9% |

: End-to-end throughput of PI vs. original on the RTX 4090 D (earlier engine).

# Appendix D: Artifact {-}

Everything is in one repository, `github.com/HaochengLu/pi-attn`: the engine (`llm_infer_rust/`, kernels in `llm_infer_rust/src/kernels/pi/`), the kernel bindings and tests, detcheck, the evaluation scripts, Gated DeltaNet, the cross-class models and programs of §7 (`xclass/`) and all results, including the per-GPU runs of Table 2 (`results/modal/`). `eval/run_all.sh` rebuilds the engine and regenerates every result in about 8 hours on one A800, `results/README.md` maps every table and figure to its files, and `paper/figures.py` redraws every figure from them.
