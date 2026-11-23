---
title: "Fix the Program, Not the Schedule"
author: "Haocheng Lu, Haoxuan Cheng"
date: ""
abstract: |
  Production LLM serving engines are not reproducible: identical requests at temperature zero return different tokens and log-probabilities depending on batch composition, prefix-cache hits, chunked prefill boundaries, and whether a token is evaluated during decode or prefill. In reinforcement learning (RL), rescoring rollouts in a trainer produces yet another answer, corrupting importance ratios before a single gradient step is taken. Existing deterministic modes treat reproducibility as a heavy performance tax—disabling split-KV, pinning GEMM tiles, or turning off radix caching—sacrificing 30–71% of serving throughput while still failing prefill–decode equivalence.

  We show that this tradeoff is an artifact of fixing the wrong abstraction: **bitwise determinism carries zero throughput tax.** Instead of constraining the hardware schedule, **PI-Attn** fixes the *floating-point program*: every query row and projection output element executes one canonical sequence of operations determined solely by token positions and weight shapes, designed so its associative boundaries align with the exact partitions fast GPU kernels exploit. Prefill, chunked prefill, prefix-cache hits, split-KV decode, and packed trainer forwards become bitwise identical by construction, while tiling, software pipelining, thread-block clusters, stream-K, CUDA graphs, and tensor-parallel sharding remain completely free.

  Deployed in a full-featured serving engine across Ampere (A800), Ada (RTX 4090 D), and Hopper (H100) with radix caching, chunked prefill, mixed batching, and CUDA graphs all active, PI-Attn matches or beats non-deterministic serving (**+3.3% geomean speedup** on A800; **+0.7%** on H100; faster than non-deterministic vLLM and SGLang across A800 workloads) and runs **1.5–4.2× faster** than the deterministic modes of vLLM and SGLang. Sharing the canonical forward with a PyTorch trainer reproduces 100% of rollout log-probabilities bit-for-bit across arbitrary sequence packings, eliminating RL importance-ratio noise ($r_t \equiv 1$) and rollout re-scoring in one stroke. The same principle delivers zero-overhead exact Gated DeltaNet decode in vLLM, tensor-parallel degree invariance across $t \in \{1..8\}$, and bit-exact inference across GPU generations.
---

![**Bitwise determinism without the throughput tax.** (a) Deterministic serving throughput relative to each engine's own non-deterministic baseline on Qwen3-1.7B: PI-Attn runs at full speed on both A800 (1.02× geomean across the 5 standard workloads; +3.3% across all 40 multi-model points) and H100 (1.01× geomean), whereas vLLM 0.17.1 batch-invariant and SGLang 0.5.5 deterministic collapse to 29–70%. (b) Rollout tokens whose trainer-rescored importance ratio $r_t$ drifts from 1 by $>1\%$: **0%** for PI-Attn across arbitrary sequence packings versus 10.4–22.2% for a standard BF16 trainer. (c) Cross-GPU bitwise agreement relative to an A800 across Class-A GPUs (**100%** on PI-Attn vs.\ 0–36% stock).](figs/fig0_teaser.pdf){width=100%}

# Introduction

Ask a production LLM server the same question twice at temperature 0 and you routinely get different answers. Floating-point addition is non-associative, and modern serving engines continuously repartition reductions in response to load: dynamic batch sizes shift GEMM tiling and split-$K$ factors, KV occupancy alters split-KV decode boundaries, chunked prefill slices prompts at load-dependent offsets, radix-cache hits replace prefill compute with paged reads, and multi-turn continuations re-evaluate decoded tokens through prefill kernels. Consequently, evaluation benchmarks degrade into load-dependent random variables—changing client concurrency from 1 to 16 alters **67% of Qwen3-1.7B answers** on GSM8K—and RL pipelines suffer systematic rollout–trainer mismatch on policy-unchanged tokens, forcing practitioners either to burn GPU hours on redundant prefill re-scoring passes or to absorb gradient bias through heuristic ratio clipping and masking [@vexact].

Prevailing systems assume that determinism requires *schedule invariance* [@tml25; @sgl25]: force one reduction order by disabling adaptive split-KV, pinning one GEMM tile across batch sizes, or turning off radix caching. Because those exact mechanisms keep GPUs saturated under dynamic load, schedule pinning craters throughput by **51–71% on A800** and **30–66% on H100** (Figure 1a)—and still fails *prefill–decode equivalence*, because prefill rescales probabilities against a single running maximum while split decode rescales against split-local maxima.

**Our thesis is simple: fix the floating-point program, not the hardware schedule.** If every query row and projection element executes one *canonical sequence* of floating-point operations parameterized solely by token positions and weight shapes, *any* kernel evaluating that sequence produces identical bits regardless of batching, tiling, or SM placement. Crucially, fixing the program does not serialize execution: every associative boundary built into the program becomes an unconstrained degree of parallelism for the scheduler and autotuner. By aligning those canonical boundaries with the natural parallel partitions of modern Tensor Core kernels, we make determinism free:

- **Canonical Attention (§2.1):** Keys are partitioned at *absolute* sequence positions into 512-key fold groups, each executing 16 online-softmax steps over 32-key tiles from $(-\infty, 0, 0)$ before folding in ascending order. FlashAttention-2 [@dao23] and Hopper `wgmma` prefill kernels run this exact program at full speed simply by resetting state every 512 keys, while split-KV decode assigns each group to an independent CTA.
- **Canonical Projections (§2.2):** Each output element executes an ascending chain of 16-wide Tensor Core MMA steps over $K$, partitioned into shape-determined segments folded from $+0$. Because dynamic batch size $M$ and output tile shapes never enter the arithmetic chain, autotuners remain free to deploy warp-streaming kernels, multi-stage CUTLASS tiles, Hopper TMA multicast clusters, or **deterministic stream-K**—passing exact `fp32` accumulators across SMs mid-chain with zero bit drift.

We realize this design end-to-end across six GPU architectures and make five contributions:

1. **Canonical Numerical Contract (§2):** A formal specification decoupling bit-defining arithmetic from schedule-free execution, backed by invariance proofs spanning dynamic batching, prefix caching, decode-vs-prefill, stream-K, and tensor parallelism.
2. **Full-Speed Ampere, Ada, and Hopper Serving Engine (§3, §4):** Native Class-A (`sm_80`/`sm_89`) and Hopper (`sm_90` `wgmma`/TMA/cluster) kernels inside a production serving engine with radix caching, chunked prefill, mixed batching, and CUDA graphs all active—delivering a **+3.3% geomean speedup** over non-deterministic serving on A800 (**2.1–4.2×** faster than deterministic vLLM/SGLang), **18/20 wins** on RTX 4090 D, and **+0.7% geomean speed** on H100 (**1.5–3.1×** faster than deterministic vLLM/SGLang).
3. **Exact Zero-Mismatch RL Alignment (§4.2):** Sharing the canonical forward with PyTorch reproduces **55,593 / 55,593** rollout log-probabilities bit-for-bit across arbitrary sequence packings ($r_t \equiv 1$), eliminating rollout re-scoring passes and off-policy ratio corruption at full serving speed.
4. **Cross-GPU Bitwise Identity & Exact Lattices (§5.1):** Bit-exact Tensor Core models establishing native bitwise identity across five Ampere and Ada GPUs (Class A, $F{=}24$) and Hopper (Class H, $F{=}25$), paired with **Exact-Lattice** kernels that guarantee identical bits across Ampere, Ada, Hopper, and CPU at higher numerical precision than native chains.
5. **Architectural Generality: Linear Attention & Elastic TP (§5.2):** Bitwise-exact Gated DeltaNet decode in vLLM 0.17.1 at **0.98–1.08×** stock throughput with zero-copy speculative rollback, plus 100% bit-identical logits across tensor-parallel degrees $t \in \{1,2,3,4,8\}$.

# Canonical Floating-Point Programs

**System Guarantee.** PI-Attn turns LLM inference into a pure mathematical function: every generated token ID and `fp32` log-probability is determined bit-for-bit solely by the request's input tokens, sampling parameters, and seed. Batch composition, concurrency, radix-cache state, prefill chunking, decode versus prefill, CUDA graphs, KV page size, and server restarts cannot alter a single bit—nor can scaling the runtime tensor-parallel degree $t \le t_{\max}$ (§5.2). This guarantee holds natively across all GPUs within a Tensor Core equivalence class (§5.1) and extends across GPU generations and CPUs in Exact-Lattice mode. It rests on four hardware properties verified exhaustively in §5.1: **(A1)** an MMA instruction is a deterministic function of its operands and accumulator; **(A2)** `ex2.approx` and `lg2.approx` are deterministic; **(A3)** explicit IEEE round-to-nearest intrinsics (`__fmaf_rn`, `__fadd_rn`, `__fmul_rn`) preserve evaluation order; and **(A4)** an MMA whose accumulator is not $-0$ never returns $-0$.

![**The canonical attention program.** Keys are partitioned at absolute sequence indices into 32-key tiles and 512-key fold groups. Each group executes 16 online-softmax steps from $(-\infty,0,0)$, and group partials fold in ascending order. Streaming prefill, split-KV decode, radix-cache hits, and chunked extends all execute the identical per-row operation DAG.](figs/fig1_attention_program.pdf){width=100%}

## Canonical Attention

Why do existing attention kernels disagree between prefill and decode? Defining single-pass FlashAttention-2 over the entire prefix $0..p$ as the reference program carries one unbroken running maximum across all keys, making parallel split-KV decode impossible without serializing across the sequence. Conversely, reducing every 32-key tile independently in a tree would force expensive `fp32` partial merges inside prefill's tightest inner loop. A **two-level hierarchy** resolves the tension: running 16 online-softmax steps over 32-key tiles *inside* independent, fixed-boundary **512-key fold groups** gives prefill an uninterrupted 512-key FlashAttention-2 inner loop while giving decode up to 64 independent split-KV CTAs per head across 32K contexts.

Concretely, for query row $i$ at causal position $p$, scores $s_j = (q_i \cdot k_j)\cdot \mathrm{scale}\cdot \log_2 e$ are accumulated via `m16n8k16` MMAs in ascending 16-wide head-dimension steps and masked to $-\infty$ for $j > p$. Keys are cut at absolute indices $0, 512, 1024, \dots$ into 512-key fold groups. Each group $g$ starts from $(m, d, o) = (-\infty, 0, 0)$ and scans its 16 tiles of 32 keys in ascending order:
$$
m' = \max\!\big(m, \max_j s_j\big),\quad \sigma = 2^{m - m'},\quad P_j = \mathrm{bf16}\!\big(2^{s_j - m'}\big),\quad d \leftarrow \mathrm{fma}\!\big(d, \sigma, \textstyle\sum_j P_j\big),\quad o \leftarrow \mathrm{fma}(o, \sigma, +0) + P V,
$$
where $\sum_j P_j$ reduces the 32 tile keys in one fixed order (replacing $m'$ with $0$ while $m' = -\infty$). Completed group partials $(m_g, d_g, o_g)$ fold in ascending group order $g = 0, 1, \dots$ into $(M, L, O)$ via $M' = \max(M, m_g)$, $L \leftarrow \mathrm{fma}(L, 2^{M - M'}, d_g 2^{m_g - M'})$, and $O \leftarrow \mathrm{fma}(O, 2^{M - M'}, o_g 2^{m_g - M'})$, producing $\mathrm{bf16}(O / L)$ (Figure 2).

Two algebraic details keep prefill and decode bitwise identical at causal boundaries. First, any 32-key tile entirely past $p$ is an **exact no-op**: all $P_j = 0$, $\sigma = 1$, and $\mathrm{fma}(o, 1, +0)$ under Axiom A4 preserves $o$ without introducing $-0$, allowing wide prefill warps to sweep masked tail tiles that decode never visits. Second, folding group $g=0$ into $(-\infty, 0, 0)$ is exact, so single-group requests finish in registers without global memory round-trips.

**Lemma 1 (Attention Partition Invariance).** *Any kernel executing the canonical attention program per owned query row yields bitwise-identical outputs regardless of co-batched rows, group-to-CTA assignment, chunked prefill boundaries, prefix-cache hits, or prefill-versus-decode execution.* *Proof.* Every operand and operation on row $i$ is fixed by $p$ and KV entries at indices $\le p$; splitting warps across output columns or groups across CTAs leaves every element's arithmetic chain unchanged. $\square$

## Canonical Projection GEMM

Narrow projections with deep reduction dimensions (such as `o_proj` and `down_proj` during small-batch decode) require split-$K$ parallelism to saturate SMs, yet standard engines choose split counts from dynamic batch size $M$ and reduce partials in arrival order. To provide **deterministic split-$K$** only where decode needs it, we partition $K$ into $S = S(N, K, t_{\max})$ equal segments determined solely by static weight shape $(N, K)$ and the declared tensor-parallel ceiling $t_{\max}$.^[For $t_{\max}=1$, $S=2$ on narrow deep projections satisfying $N \le 1024, K \ge 2048, K \equiv 0 \pmod{512}$, and $S=1$ otherwise.] Within segment $s$, each output element accumulates ascending 16-wide $k$ steps from $+0$ into an `fp32` partial $p_s$, followed by an explicit left-fold $y = \mathrm{bf16}\big((\cdots((+0 + p_0) + p_1)\cdots) + p_{S-1}\big)$ (Figure 3a).

![**The canonical GEMM program.** (a) Each output element executes one ascending `m16n8k16` accumulation chain per shape-determined $K$ segment and left-folds segment partials in `fp32`. (b) Output tiling, cluster multicast, deterministic stream-K accumulator hand-offs, and column-parallel TP sharding leave every element's arithmetic chain untouched.](figs/fig2_gemm_program.pdf){width=100%}

**Lemma 2 (GEMM Schedule Invariance).** *Arbitrary $(M, N)$ tiling, arbitrary placement of (tile, segment) pairs across warps or CTAs, any CTA arrival order prior to the ascending segment fold, and any mid-chain worker hand-off that transfers the exact `fp32` accumulator yield bitwise-identical outputs.*

## The Numerical Contract and Composition

\begin{table}[t]
\centering
\small
\caption{The PI-Attn numerical contract. Left-column parameters define the model's bitwise function alongside its dtype; right-column choices remain completely free per kernel launch.}
\label{tab:contract}
\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}p{0.46\textwidth} >{\raggedright\arraybackslash}X @{}}
\toprule
\textbf{Fixed in Contract (Defines Output Bits)} & \textbf{Free per Launch (Autotuner \& Scheduler Choice)} \\
\midrule
\texttt{m16n8k16} MMA semantics (\texttt{bf16} in, \texttt{fp32} acc.) & Instruction lowering (\texttt{mma.sync} vs.\ \texttt{wgmma}); operand transpose$^\ast$; CTA tile $(M_t, N_t)$, warps/CTA \\
Ascending 16-wide $k$ steps per segment from $+0$ & Pipeline stage depth, \texttt{cp.async} / TMA ring sizes \\
Segment count $S(N, K, t_{\max})$ \& ascending \texttt{fp32} fold & Segment-to-CTA/warp placement and arrival order \\
32-key tiles, 512-key groups, canonical softmax/fold & Deterministic stream-K cut points (\texttt{fp32} state hand-off) \\
Explicitly rounded norms, RoPE, SiLU, fixed-tree log-softmax & Query rows per CTA, split-KV count, mixed-batch routing \\
Declared tensor-parallel ceiling $t_{\max}$ & Runtime tensor-parallel degree $t \le t_{\max}$ (column-parallel) \\
\bottomrule
\multicolumn{2}{@{}p{\textwidth}@{}}{\footnotesize $^\ast$Operand-transpose equivalence—$D_{ij}(X, Y, C) \equiv D'_{ji}(Y, X, C^\top)$ bit-for-bit—is a verified microarchitectural symmetry of Class-A and Class-H Tensor Cores arising from per-$k$ product grouping (confirmed with zero mismatches across $1.69\times 10^9$ A800 and $6.4\times 10^6$ H100 outputs, including subnormals and NaNs; §5.1), distinct from the PTX ISA specification.}
\end{tabularx}
\end{table}

Table \ref{tab:contract} separates numerical specification from scheduling. Following Yang et al. [@taming], fixing the accumulating instruction semantics, $k$-step order, remainder-free segment multiples, and `fp32` merge tree uniquely fixes every output bit. Ampere and Ada kernels emit `mma.sync.aligned.m16n8k16`, while Hopper kernels emit asynchronous warpgroup `wgmma`—which we verify is **100% bit-identical to `mma.sync` on H100** across 910,000 single steps, `K=1024` chains, and 2,412 tile launches over all 108 Qwen3 projection shapes (0 mismatches). Moreover, because both Class-A and Class-H Tensor Cores group products strictly by $k$ (§5.1), `m16n8k16` is bit-symmetric under operand transposition $(A{=}X, B{=}Y, C) \leftrightarrow (A{=}Y, B{=}X, C^\top)$, allowing decode kernels to place narrow GQA query groups into the 8-column operand instead of padding to 16 rows.

Elementwise and row-local operations (RMSNorm, QK-norm, RoPE, SiLU-and-mul, paged KV writes) use fixed in-row orders. Token log-probabilities run on a dedicated 1,024-thread CTA per row with strided passes and a fixed shared-memory reduction tree shared verbatim between server and trainer, and sampling keys stateless Philox-4×32 streams by $(\text{request seed}, \text{position})$.

**Theorem 1 (Serving Determinism).** *Under Axioms A1–A4, within a fixed Tensor Core class, every activation, KV entry, logit, and fp32 log-probability at token $t$ is a bitwise function solely of tokens $0..t$, and every sampled token is a function solely of tokens $0..t$ and the seed.*

# Kernel and Serving Engine Architecture

Our implementation shares one C++/CUDA header for the canonical online step, group fold, and log-softmax across serving and training, while tailoring scheduling to Ampere/Ada (`sm_80`/`sm_89`) and Hopper (`sm_90`). Appendix B details tile sizes and shared-memory layouts.

## Schedule-Free Attention Kernels

**Operand-Swapped Low-Batch Decode and Lock-Free Split-KV.** Standard tensor-core decode pads narrow GQA query groups (2 or 4 heads per KV head on Qwen3) to an MMA's 16 rows, wasting up to 87% of Tensor Core lanes. Exploiting operand-transpose symmetry (Table \ref{tab:contract}), our 16-warp single-SM decode kernel places keys into the 16-row factor and up to 8 GQA query rows into the 8-column factor, halving MMA instructions while assigning warp $t \in [0,15]$ its own 32-key tile of a 512-key group. Because tile maxima are independent and only $d$ and $o$ form chains, this kernel speeds up low-to-medium batch decode by **1.2–2.1×**. At higher concurrencies or deeper contexts, a 4-warp group-parallel kernel launches one CTA per (request, KV head, fold group) and folds completed groups via a lock-free arrival counter; the runtime selects the faster kernel per step from the longest active request, capturing one CUDA graph per kernel.

**Pipelined Prefill and Pure-Schedule Routing.** On Ampere/Ada, prefill pipelines $QK_{t+1}^\top$ behind tile $t$'s online softmax and $PV$ across swizzled shared-memory tiles; on Hopper (`sm_90`), a warp-specialized kernel dedicates one producer warpgroup to TMA $K/V$ ring prefetching via `mbarrier`s while two or three 64-row consumer warpgroups issue asynchronous `wgmma` chains. Because Lemma 1 makes output bits independent of launch geometry, the engine applies three pure scheduling optimizations at zero bit cost: dispatching longest causal row tiles first when $K/V$ fit in L2 (up to 24% faster), splitting multi-wave chunks at whole SM wave boundaries over long prefixes (up to 22% faster), and peeling trailing 1-token decode rows off mixed prefill batches onto the decode kernel instead of sparse 128-row prefill CTAs (cutting mixed-batch latency by up to 38%).

## Row-Invariant GEMMs, Deterministic Stream-K, and Hopper Clusters

Lemma 2 allows our startup autotuner to select freely per $(N, K, M)$ bucket among deep `cp.async` warp-streaming kernels ($M \le 8$), multi-stage CUTLASS tiles, segment-parallel kernels, native Hopper TMA kernels, and **deterministic stream-K** (Appendix B). Conventional stream-K breaks determinism by merging partial tile sums across SMs. Instead, when our deterministic stream-K splits a tile's $K$-chain between resident CTAs $w$ and $w+1$, CTA $w$ writes its exact `fp32` accumulator to workspace memory and CTA $w+1$ *continues the unbroken MMA chain* from that exact bit state (Figure 3b). Because each worker executes its incoming prefix first and waits only on $w-1$, execution is deadlock-free and bit-identical to an uncut chain—outperforming cuBLAS inside Qwen3-1.7B's batch-1 decode step (`25.9 µs` $\to$ `20.7 µs`). On Hopper (`sm_90`), persistent TMA/GMMA kernels use exact-$M$ TMA descriptors at decode sizes (matching cuBLAS within 0–4% across 28 DRAM-streaming layers), **$2\times 1$ thread-block clusters** that multicast each weight stage across paired row tiles at $M \in [128, 1792]$, and single-wave $192\times 192$ tiles at $M \ge 2048$.

## Production Serving Engine Integration

We integrate these kernels into `llm_infer_rust`, a Rust/libtorch serving engine supporting radix prefix caching, chunked prefill, continuous mixed batching, segmented-prefill and decode CUDA graphs, and asynchronous CPU/GPU overlap. Enabling `--deterministic` **turns none of these optimizations off**: it routes attention to PI-Attn, all projections and the LM head to the row-invariant GEMM family, and token log-probabilities to the canonical reduction kernel without stalling overlap, tuning GEMM buckets over DRAM-streaming weight rings in 4.0–15.5 s at startup.

# Evaluation: Reproducibility, RL Alignment, and Speed

**Experimental Setup.** We evaluate dense **Qwen3** models (**0.6B, 1.7B, 4B, 8B**) in `bf16` on an NVIDIA A800-SXM4-80GB (`sm_80`) and an NVIDIA H100-SXM5-80GB (`sm_90`), comparing **PI-Attn** (`llm_infer_rust --deterministic`) against **Stock** (`llm_infer_rust` on FlashInfer + cuBLAS with identical server flags), **vLLM 0.17.1** (stock and `VLLM_BATCH_INVARIANT=1`), and **SGLang 0.5.5** (stock and `--enable-deterministic-inference`). Cross-GPU tests add an RTX 4090 D (`sm_89`), A100-80GB, A10G, and L4.

## Bitwise Reproducibility and Benchmark Stability

We built `detcheck`, an automated black-box verifier for OpenAI-compatible servers that tests bitwise identity of generated token IDs and `fp32` log-probabilities across 10 serving perturbations: dynamic batching ($c=4, 16$), 32-way shared-prefix fan-out, decode versus multi-turn re-prefill, seeded sampling, 200-run stability, and full/partial radix-cache hits.

![**Black-box `detcheck` verification on Qwen3-1.7B (A800).** Cells count requests diverging in token IDs or `fp32` log-probabilities from isolated single-request execution (`Repeat ×200`: distinct completions observed; `1 / 200` = exact). SGLang's deterministic FlashInfer path disables radix caching (`disabled`). PI-Attn achieves zero divergences across every regime.](figs/fig3_determinism.pdf){width=100%}

**Zero Bit Drift Across Load, Cache State, and Restarts.** PI-Attn passes all 10 `detcheck` suites with **zero divergences** on all four models across both A800 and H100 (Figure 4). Stock engines fail nearly every check—and checking token IDs alone is misleading: stock `llm_infer_rust` appears token-consistent on cache hits, yet `fp32` log-probabilities diverge on all 24 re-prefilled requests. On A800, SGLang's deterministic mode disables radix caching and still leaks batch variance, while vLLM's batch-invariant mode fails seeded re-prefill (8/24) because its RNG keys streams by local request offset rather than sequence position. Across eight server configurations (varying chunk size, CUDA graphs, concurrency caps, CPU/GPU overlap, KV page size, and cold restarts), PI-Attn produces identical bits on all 32 comparisons per configuration, whereas the stock engine fails 11–13 comparisons even across a plain restart. Injecting targeted kernel mutants—descending group folds, 256-key group resets, and a 1-ULP double rounding in decode merge—confirms 100% test detection.

**Benchmarks Become True Model Constants.** On GSM8K, raising client concurrency from $c=1$ to $c=16$ causes the stock engine to alter **888 of 1,319** Qwen3-1.7B generations (and 322 of 500 on 8B), shifting accuracy by 0.68 points. With PI-Attn, **zero generations change** on both models while matching the stock engine's teacher-forced `fp32` logit agreement (>98.9%; Appendix Table \ref{tab:gsm8k}).

## Zero-Mismatch RL Rollout–Trainer Alignment

In policy-gradient RL (PPO, GRPO), tokens decoded under dynamic batching, prefix caching, and split-KV are rescored in packed micro-batches by a trainer forward. Any numerical discrepancy corrupts importance ratios $r_t = \pi_\theta(y_t \mid y_{<t}) / \pi_{\text{rollout}}(y_t \mid y_{<t})$ before $\theta$ even updates—forcing frameworks either to pay for a redundant prefill *re-score pass* over generated responses, disable serving optimizations, or rely on heuristic ratio clipping/masking [@vexact].

Linking PI-Attn's canonical header and kernels into a PyTorch Qwen3 forward, we sample 64 GSM8K rollouts per model from the serving engine ($T=1.0$, $c=16$, radix cache and chunked prefill active, **55,593 tokens** across 0.6B–8B) and rescore them both individually and in randomly packed micro-batches of 2–8 sequences. Every single token (**55,593 / 55,593**) produces the **exact bitwise `fp32` log-probability** recorded during live serving, making every on-policy importance ratio **identically $1.0$** (Figure 1b). By contrast, a standard HuggingFace `bf16` trainer matches bitwise on $\le 40.8\%$ of tokens and falls outside the $\pm 1\%$ ratio band on **10.4–22.2%** of tokens, injecting spurious ratio spikes from **$0.49\times$ to $2.51\times$** (Appendix Figure 7). Rollout fleets can therefore run with every serving optimization turned on while trainers pack variable-length trajectories freely, eliminating both rollout re-scoring passes and off-policy ratio corruption at the root.

## End-to-End Serving Speed on Ampere, Ada, and Hopper

![**End-to-end serving performance.** (a) Output throughput delta of PI-Attn vs.\ the stock non-deterministic engine across 4 models and 5 greedy workloads on A800 (40-point greedy+sampled geomean: **+3.3%**). (b) Long-context TTFT scaling (512–30K tokens) on full-rate Ampere (A800) and half-rate Ada (RTX 4090 D), with min–max error bars.](figs/fig5_e2e.pdf){width=100%}

![**Cross-engine throughput on Qwen3-1.7B** (normalized to stock `llm_infer_rust` = $1.00\times$). Solid bars denote deterministic modes; hatched bars denote non-deterministic releases. PI-Attn sustains full stock speed (+1.9% geomean on A800, +0.7% on H100) while outperforming deterministic vLLM and SGLang by **2.1–4.2× on A800** and **1.5–3.1× on H100**.](figs/fig4_baselines.pdf){width=100%}

**Faster Than Non-Deterministic Serving on Ampere and Ada.** Across all 40 workload points on A800 (four models $\times$ five workloads $\times$ greedy and $T=0.8$), PI-Attn delivers a **+3.3% geometric-mean speedup** (−1.6% to +16.9%) over its own non-deterministic baseline (Figure 5a). Low-batch decoding gains **+6.8% to +16.9%**, propelled by our single-launch 16-warp operand-swapped decode kernel and deterministic stream-K, while high-concurrency serving ($c=32$–$128$) runs at full speed (Appendix Tables \ref{tab:ablation}–\ref{tab:extended}). On an Ada RTX 4090 D (`sm_89`), PI-Attn beats the non-deterministic engine on **18 of 20 workloads** by up to **+8.5%** (Appendix B). Against external engines on A800 (Figure 6a, Appendix Table \ref{tab:base}), PI-Attn **outperforms non-deterministic vLLM 0.17.1 (+0.7% to +13.3%) and non-deterministic SGLang 0.5.5 (+3.2% to +20.9%) on every single workload**, and runs **2.1–3.8× faster** than vLLM's batch-invariant mode and **2.3–4.2× faster** than SGLang's deterministic mode.

\begin{table}[t]
\centering
\small
\caption{Hopper schedule ablation on NVIDIA H100 (Qwen3-1.7B, greedy, vs.\ stock FlashInfer + cuBLAS engine). Upgrading only the free side of the contract from portable \texttt{mma.sync} kernels to native \texttt{wgmma}/TMA/cluster kernels eliminates the portability penalty while preserving 100\% identical Class-H bits.}
\label{tab:h100abl}
\begin{tabular}{@{} l r r r r r r @{}}
\toprule
\textbf{H100 Kernel Schedule (Identical Bits)} & \textbf{64/32 c1} & \textbf{64/32 c8} & \textbf{64/32 c32} & \textbf{512/256 c8} & \textbf{2048/128 c8} & \textbf{1024/512 c32} \\
\midrule
Portable \texttt{mma.sync} ($\Delta$ throughput) & $-2.8\%$ & $-9.6\%$ & $-9.8\%$ & $-16.4\%$ & $-16.6\%$ & $-13.5\%$ \\
\textbf{Native Hopper \texttt{wgmma}/TMA ($\Delta$ throughput)} & \textbf{+4.7\%} & \textbf{+3.7\%} & \textbf{+0.5\%} & \textbf{$-$1.8\%} & \textbf{$-$1.1\%} & \textbf{$-$0.5\%} \\
\midrule
Portable \texttt{mma.sync} ($\Delta$ TTFT) & $+15\%$ & $+15\%$ & $-2\%$ & $+46\%$ & $+52\%$ & $+50\%$ \\
\textbf{Native Hopper \texttt{wgmma}/TMA ($\Delta$ TTFT)} & \textbf{+6\%} & \textbf{$-$9\%} & \textbf{+13\%} & \textbf{+7\%} & \textbf{+8\%} & \textbf{+7\%} \\
\bottomrule
\end{tabular}
\end{table}

**Full Speed and 1.5–3.1× Deterministic Speedup on Hopper.** On H100—the primary target architecture for existing batch-invariant kernels—Table \ref{tab:h100abl} demonstrates the power of decoupling program from schedule: swapping portable `mma.sync` launches for native Hopper `wgmma`, TMA, and $2\times 1$ cluster kernels eliminates a 17% portability gap and slashes prefill TTFT overhead from +52% down to +7–8% **without altering a single output bit**. Across the five cross-engine benchmarks on H100 (Figure 6b, Appendix Table \ref{tab:base}), PI-Attn achieves **+0.7% geomean throughput** (−1.7% to +4.4%) relative to its non-deterministic baseline, running **1.8–3.1× faster than vLLM's batch-invariant mode** and **1.5–1.9× faster than SGLang's deterministic mode**.

**Long-Context Prefill Scaling to 30K Tokens.** Figure 5b tracks time-to-first-token (TTFT) from 512 to 30,000 prompt tokens. On RTX 4090 D (`sm_89`), where Tensor Core math naturally overlaps scalar rescaling, PI-Attn holds TTFT within **−2.8% to +1.6% across the entire 512–30K span**. On A800 (`sm_80`), TTFT matches the stock engine within **−2.9% to +4.0%** up to 4K tokens and stays within **+3.7%–+10.2%** at 16K–30K tokens, where full-rate Tensor Cores complete 32-key MMAs fast enough to expose scalar online-softmax updates (§7).

# Cross-GPU Equivalence and Architectural Generality

## Tensor-Core Equivalence Classes and Exact Lattices

Two GPUs share native output bits if and only if their Tensor Cores evaluate `m16n8k16` identically. Fitting bit-exact MMA models [@mmasim; @hawkeye] across 910,000 random, subnormal, and `K=1024` chain probes per GPU (0 mismatches) partitions modern NVIDIA architectures into two exact classes: **Class A** (`sm_80`/`sm_86`/`sm_89` — Ampere & Ada), which truncates scaled products to $F=24$ bits and reduces a $k16$ step in two chained 8-product groups; and **Class H** (`sm_90` — Hopper), which retains $F=25$ bits in a single 16-product group. Exhaustive sweeps over all $2^{32}$ `fp32` inputs confirm that all 30 SFU and IEEE operations used by PI-Attn return identical bits across A100, A10G, L4, and H100. Consequently, whereas stock cuBLAS and FlashInfer shatter even across GPUs in the same family—diverging on **69% of GSM8K generations** between A800 and RTX 4090 D—**PI-Attn delivers one identical bitstream across every GPU in a class** (Table \ref{tab:xgpu}): 100% kernel agreement across A800, A100, A10G, L4, and RTX 4090 D, **14,293 / 14,293** identical rollout log-probabilities on A10G, and **1,319 / 1,319** identical GSM8K answers on RTX 4090 D.

\begin{table}[t]
\centering
\small
\caption{Cross-GPU bitwise identity relative to NVIDIA A800 (\texttt{sm\_80}) across Qwen3 0.6B–8B GEMM shapes, attention geometries, and full engine replays (64 seeded rollouts on A10G/H100; 1,319 GSM8K generations on RTX 4090 D$^\dagger$). Every GPU passes 100\% of within-GPU \texttt{detcheck} and invariance suites.}
\label{tab:xgpu}
\begin{tabular}{@{} l c c c c c l @{}}
\toprule
\textbf{GPU} & \textbf{Class} & \textbf{PI GEMM} & \textbf{PI Attn} & \textbf{cuBLAS} & \textbf{FlashInfer} & \textbf{Engine Replay vs.\ A800} \\
\midrule
A100-80GB (\texttt{sm\_80}) & A & \textbf{108 / 108} & \textbf{5 / 5} & 108 / 108 & 5 / 5 & Kernel-hash identical \\
A10G (\texttt{sm\_86}) & A & \textbf{108 / 108} & \textbf{5 / 5} & 52 / 108 & 0 / 5 & \textbf{14,293 / 14,293} log-probs \\
L4 (\texttt{sm\_89}) & A & \textbf{108 / 108} & \textbf{5 / 5} & 39 / 108 & 0 / 5 & Kernel-hash identical \\
RTX 4090 D (\texttt{sm\_89})$^\dagger$ & A & \textbf{80 / 80} & \textbf{5 / 5} & 14 / 80 & 1 / 5 & \textbf{1,319 / 1,319} GSM8K gens \\
\midrule
H100-80GB (\texttt{sm\_90}) & H & 6 / 108 & 0 / 5 & --- & --- & Class-H bits; \textbf{14,442/14,442} trainer \\
\bottomrule
\multicolumn{7}{@{}l@{}}{\footnotesize $^\dagger$Evaluated on the 80-shape Qwen3 GEMM suite; stock serving diverges on 906–910 of 1,319 GSM8K generations vs.\ A800.}
\end{tabular}
\end{table}

**Universal Cross-Generation Identity via Exact Lattices.** Refuting claims that bitwise inference cannot cross GPU generations [@heal], our Tensor Core models yield an exact algebraic condition for universal identity: whenever an MMA step's accumulator $c$ and products $a_i b_i$ lie on a common quantum $u$ satisfying $|c| + \sum_{i=1}^{16} |a_i b_i| \le 2^{24} u$, *zero internal truncation occurs on any $F \ge 24$ Tensor Core*. Projecting block operands onto power-of-two lattices (**Exact-Lattice** GEMM and attention; Appendix C) produces **100% bitwise-identical outputs across A100, A10/A10G, L4, H100, and an IEEE CPU reference** (`mma.sync` and `wgmma` alike), achieving *lower* numerical error than native chains ($0.34\text{--}1.7\times 10^{-6}$ vs.\ $1.3\text{--}8.1\times 10^{-6}$) at just **7–19% decode overhead** with offline 12-bit weight lattices.

## Beyond Softmax Attention and Single-GPU Serving

**Exact Gated DeltaNet Decode with Free Speculation.** Hybrid models (Qwen3.5, Qwen3-Next, Kimi Linear) train linear-attention Gated DeltaNet (GDN) layers with a 64-token chunked forward (`fla`) but serve decode with a step-wise recurrent kernel—corrupting 100% of rollout rows relative to the trainer while requiring $1 + K_{\text{draft}}$ full state copies for speculative rollback. Turning `fla`'s row-causal 64-token chunk program into a resumable decode and verification kernel solves both problems simultaneously: inside **vLLM 0.17.1** paired with PI-Attn's GEMM, Qwen3.5-0.8B and 4B achieve **2,048 / 2,048 bitwise decode–prefill log-probability agreement** across chunked prefill, preemption, and n-gram or MTP speculation at **0.98–1.08× stock vLLM throughput**—running **3.6× faster** at batch 1 than vLLM's built-in batch-invariant matmul ($0.27\times$) while requiring **zero state copies** for speculative rollback (Appendix D).

**Elastic Tensor-Parallel Invariance.** Standard Megatron-style row-parallel sharding splits $K$ reductions across ranks, altering 77–87% of Qwen3-8B logits whenever cluster elasticity changes the tensor-parallel degree $t$. Keeping all projections **column-parallel** (whole-$K$ columns per rank with bit-exact `all-gather`) and sizing segments $S(N, K, t_{\max})$ from a declared ceiling $t_{\max}$ decouples arithmetic from cluster topology—reproducing Qwen3-8B logits **100% bit-for-bit across $t \in \{1, 2, 3, 4, 8\}$** while keeping per-rank GEMMs on 70B shapes within 2–26% of Megatron's sharding at $t=2$.

# Related Work

**Schedule-Pinned & Speculative Determinism.** Thinking Machines Lab [@tml25], SGLang [@sgl25], and vLLM's batch-invariant mode (`VLLM_BATCH_INVARIANT`) pin GEMM tiles and restrict split-KV. DeepSeek-V4 [@dsv4] drops split-KV and mimics single-SM order; vLLM PR #62365 [@vllm62365] routes prefills through unsplit decode (0.84–0.91× speed); CoRun [@corun] pads dynamic batches to fixed shapes; and LLM-42 [@llm42] speculates non-deterministically and rolls back mismatches. PI-Attn instead fixes the *program*, leaving dynamic batching, split-KV, radix caching, and autotuning completely unconstrained while guaranteeing prefill–decode identity by construction.

**Deterministic Kernels, Cross-GPU Models, & RL Alignment.** Fixed-split attention in MLX [@mlxbi] (3.9–4.4× slower) and `llama.cpp` [@lcpp] sacrifices prefill speed; Vosti [@vosti] verifies single-loop attention without split-KV decode; FoldAttention [@foldattn] declares an upfront softmax reference for Hopper decode and deterministic backward passes. PI-Attn preserves FlashAttention-2 [@dao23] inner loops inside 512-key fold groups shared with split-KV decode [@ye25; @fd23]. For GEMMs, Yang et al. [@taming] verify static reduction descriptors; we co-design an $M$- and TP-invariant [@tbik] serving descriptor realized across five kernel families including accumulator-handoff stream-K (avoiding partial-tile merges [@rocm13151] and costly reproducible accumulators [@dn13; @cooper]). Across GPUs, MMA-Sim [@mmasim] and Hawkeye [@hawkeye] model Tensor Core arithmetic, EigenAI [@eigenai] re-executes on matching GPUs, HEAL [@heal] bounds drift without bitwise identity, and CertiSum [@certisum] certifies and repairs MXFP4 training sums—complementing our compile-time canonical BF16 programs and exact-lattice projections. In RL, VeXact [@vexact] shows that rollout–trainer drift alone can collapse policy optimization, a failure mode PI-Attn eliminates cleanly at full serving speed.

# Discussion and Future Directions

**Class-Parameterized Contracts for Wider Tensor Cores.** Using one universal `(tile=32, group=512)` contract across all GPUs maximizes simplicity while delivering +3.3% speedup on Ampere and full parity through $c=64$ on Hopper. At extreme Hopper concurrencies ($c \in \{128, 256\}$, 4.3–8.0% slower) and ultra-long full-rate prefills (Figure 5b), 32-key scalar rescaling and single-SM 512-key KV loads begin to surface—and Blackwell's wider `tcgen05` tensor-memory warpgroups push native MMA granularity further to 128 keys. Because Class A ($F{=}24$), Class H ($F{=}25$), and Blackwell already form distinct native equivalence classes, parameterizing `(tile, group)` *per Tensor Core class* (e.g., 128-key online steps for `wgmma`/`tcgen05`, paired with operand-swapped Hopper decode and $3\times 1$ TMA clusters) naturally aligns the contract with wider Tensor Cores at zero loss of within-class reproducibility.

**Frontier-Scale Serving, Low-Precision Contracts, & Closed-Loop RL.** While our kernel suites span 0.6B–8B (and 70B TP shapes) and end-to-end serving covers 0.6B–8B on Ampere/Ada and 1.7B on H100, the canonical-program paradigm opens rich avenues at frontier scale: benchmarking 70B+ dense and routed-MoE models with multi-node `all-gather` overlap, integrating Exact-Lattice kernels directly into heterogeneous serving pools, co-designing canonical contracts for FP8 and MXFP4 [@certisum], and combining our zero-mismatch RL forward with deterministic backward kernels [@foldattn] across multi-step GRPO training runs.

# Conclusion

Bitwise-reproducible LLM serving does not require sacrificing speed. By fixing canonical per-row floating-point programs whose associative boundaries match the natural parallel partitions of modern GPUs, dynamic batching, radix caching, chunked prefill, split-KV decode, CUDA graphs, speculation, and elastic tensor parallelism become pure scheduling freedoms that cannot alter a single bit—and an RL trainer sees the exact log-probabilities the rollout engine produced. Determinism, properly formulated, is free.

# References {-}

::: {#refs}
:::

\clearpage

# Appendix A: Extended Serving Benchmarks and Ablations {-}

\begin{table}[h!]
\centering
\small
\caption{Same-GPU cross-engine serving throughput (greedy output tok/s) on Qwen3-1.7B across NVIDIA A800 (\texttt{sm\_80}) and NVIDIA H100 (\texttt{sm\_90}). Stock and PI-Attn run with radix caching, chunked prefill, mixed batching, and CUDA graphs enabled; vLLM 0.11.2's batch-invariant flag fails numeric invariance on \texttt{sm\_80}, and vLLM 0.17.1 batch-invariant fails seeded re-prefill (8/24) on both GPUs.}
\label{tab:base}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{} l l r r r r r @{\hspace{16pt}} r r r r r @{}}
\toprule
 &  & \multicolumn{5}{c}{\textbf{NVIDIA A800 (\texttt{sm\_80})}} & \multicolumn{5}{c}{\textbf{NVIDIA H100 (\texttt{sm\_90})}} \\
\cmidrule(lr){3-7} \cmidrule(l){8-12}
\textbf{Serving Engine} & \textbf{Mode} & \textbf{c1} & \textbf{c8} & \textbf{c32} & \textbf{c16 (2K)} & \textbf{c4 (4K)} & \textbf{c1} & \textbf{c8} & \textbf{c32} & \textbf{c16 (2K)} & \textbf{c4 (4K)} \\
\midrule
\texttt{llm\_infer\_rust} (Stock) & Non-determ. & 283 & 1,908 & 5,060 & 1,897 & 540 & 436 & 2,998 & 8,537 & 3,375 & 974 \\
vLLM 0.17.1 & Non-determ. & 270 & 1,842 & 4,890 & 1,885 & 553 & 442 & 3,066 & 8,738 & 3,648 & 1,081 \\
SGLang 0.5.5 & Non-determ. & 253 & 1,768 & 4,865 & 1,837 & 530 & 407 & 2,758 & 7,812 & 3,296 & 956 \\
vLLM 0.11.2 & Non-determ. & 237 & 1,599 & 4,149 & 1,694 & 508 & --- & --- & --- & --- & --- \\
\midrule
vLLM 0.17.1 batch-invariant & Deterministic & 80 & 609 & 2,228 & 921 & 223 & 149 & 1,168 & 4,442 & 1,889 & 467 \\
SGLang 0.5.5 deterministic & Deterministic & 73 & 554 & 1,993 & 843 & 192 & 243 & 1,710 & 5,446 & 2,311 & 577 \\
\textbf{PI-Attn (\texttt{--deterministic})} & \textbf{Deterministic} & \textbf{306} & \textbf{1,899} & \textbf{5,021} & \textbf{1,906} & \textbf{557} & \textbf{455} & \textbf{2,946} & \textbf{8,624} & \textbf{3,400} & \textbf{966} \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[h!]
\centering
\small
\caption{Component ablation on NVIDIA A800: output throughput change relative to the stock engine and per-output-token latency delta ($\mu$s) when enabling PI-Attn attention alone, row-invariant GEMMs alone, or both.}
\label{tab:ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{} l l r r r r r @{}}
\toprule
\textbf{Model} & \textbf{Workload} & \textbf{Stock tok/s} & \textbf{PI Attn Only} & \textbf{PI GEMM Only} & \textbf{Both (Full PI)} & \textbf{$\Delta$ TPOT: Attn, GEMM, Both ($\mu$s)} \\
\midrule
0.6B & 64 / 256 / c1 & 512.7 & $+7.2\%$ & $-6.3\%$ & $+0.2\%$ & $-132,\; +133,\; -4$ \\
0.6B & 512 / 256 / c8 & 2,585.9 & $+1.2\%$ & $+15.0\%$ & $+16.7\%$ & $-39,\; -412,\; -450$ \\
0.6B & 512 / 256 / c32 & 7,426.3 & $+1.7\%$ & $-0.5\%$ & $+0.8\%$ & $-78,\; +32,\; -51$ \\
0.6B & 512 / 256 / c128 & 11,559.7 & $+2.1\%$ & $-1.6\%$ & $0.0\%$ & $-249,\; +158,\; -91$ \\
\midrule
1.7B & 64 / 256 / c1 & 282.7 & $+4.2\%$ & $+3.7\%$ & $+8.5\%$ & $-144,\; -128,\; -281$ \\
1.7B & 512 / 256 / c8 & 1,907.4 & $+0.7\%$ & $-1.7\%$ & $-0.4\%$ & $-33,\; +74,\; +19$ \\
1.7B & 512 / 256 / c32 & 5,056.5 & $+1.5\%$ & $-1.4\%$ & $0.0\%$ & $-101,\; +113,\; +12$ \\
1.7B & 2048 / 256 / c16 & 1,897.5 & $+0.2\%$ & $0.0\%$ & $+0.4\%$ & $-95,\; -32,\; -28$ \\
1.7B & 4096 / 128 / c4 & 539.8 & $+0.8\%$ & $+2.2\%$ & $+2.8\%$ & $-39,\; -122,\; -137$ \\
1.7B & 2048 / 128 / c8 & 1,115.1 & $-0.2\%$ & $+1.1\%$ & $+0.7\%$ & $-8,\; -39,\; -35$ \\
1.7B & 512 / 256 / c64 & 7,024.2 & $+1.9\%$ & $-1.2\%$ & $+0.6\%$ & $-187,\; +87,\; -83$ \\
\midrule
8B & 64 / 256 / c1 & 89.9 & $+5.5\%$ & $+1.5\%$ & $+7.0\%$ & $-583,\; -165,\; -736$ \\
8B & 512 / 256 / c8 & 631.0 & $+3.7\%$ & $-1.0\%$ & $+3.3\%$ & $-472,\; +109,\; -444$ \\
8B & 512 / 256 / c32 & 1,808.6 & $+0.4\%$ & $+0.7\%$ & $+1.2\%$ & $-117,\; -155,\; -274$ \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[tbp]
\centering
\small
\caption{Long-context, shared-prefix, and high-concurrency workloads on NVIDIA A800 (PI-Attn vs.\ Stock; median of 3 interleaved launches). Throughput reports output tok/s, or req/s for short-generation prefill benchmarks (1-, 4-, and 16-token outputs).}
\label{tab:extended}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{} l l r r @{\hspace{20pt}} l l r r @{}}
\toprule
\textbf{Model} & \textbf{Workload (in / out / conc)} & \textbf{$\Delta$ Thpt} & \textbf{$\Delta$ TTFT} & \textbf{Model} & \textbf{Workload (in / out / conc)} & \textbf{$\Delta$ Thpt} & \textbf{$\Delta$ TTFT} \\
\midrule
1.7B & 64 / 256 / c1 & $+8.3\%$ & $+13.1\%$ & 8B & 4096 / 128 / c8 (3,968 shared) & $+8.2\%$ & $+3.3\%$ \\
1.7B & 512 / 256 / c32 & $-0.4\%$ & $+0.4\%$ & 8B & 8192 / 64 / c4 (8,000 shared) & $+5.8\%$ & $+5.3\%$ \\
1.7B & 2048 / 16 / c8 & $+0.5\%$ & $+2.0\%$ & 8B & 16384 / 1 / c1 (16K prefill) & $-9.2\%$ & $+10.2\%$ \\
1.7B & 4096 / 4 / c4 & $+3.2\%$ & $-2.9\%$ & 0.6B & 512 / 256 / c64 & $+2.2\%$ & $-1.1\%$ \\
1.7B & 4096 / 128 / c8 (3,968 shared) & $+1.7\%$ & $+2.3\%$ & 0.6B & 512 / 256 / c128 & $-0.3\%$ & $+13.0\%$ \\
1.7B & 8192 / 64 / c4 (8,000 shared) & $+1.1\%$ & $+13.6\%$ & 0.6B & 128 / 512 / c128 & $+1.4\%$ & $0.0\%$ \\
1.7B & 16384 / 256 / c2 & $-0.2\%$ & $+3.7\%$ & 0.6B & 2048 / 128 / c64 & $+0.7\%$ & $0.0\%$ \\
1.7B & 30000 / 128 / c1 & $-4.3\%$ & $+8.6\%$ & 1.7B & 512 / 256 / c64 & $+1.1\%$ & $+1.7\%$ \\
8B & 64 / 256 / c1 & $+7.1\%$ & $+4.5\%$ & 1.7B & 512 / 256 / c128 & $+1.3\%$ & $-1.4\%$ \\
8B & 512 / 256 / c32 & $+1.0\%$ & $+3.5\%$ & 1.7B & 128 / 512 / c128 & $+1.0\%$ & $-2.8\%$ \\
8B & 2048 / 16 / c8 & $-3.4\%$ & $+4.0\%$ & 1.7B & 2048 / 128 / c64 & $+3.5\%$ & $-4.6\%$ \\
8B & 4096 / 4 / c4 & $-3.5\%$ & $+3.6\%$ & 4B & 512 / 256 / c64 & $+6.9\%$ & $+3.4\%$ \\
 &  &  &  & 4B & 128 / 512 / c64 & $+7.1\%$ & $+4.6\%$ \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[tbp]
\centering
\small
\caption{GSM8K evaluation stability across client concurrencies ($c=1$ vs.\ $c=16$) and teacher-forced top-1 logit agreement against an \texttt{fp32} HuggingFace reference over 1,280 positions (0.6B: 99.53\% stock vs.\ 99.61\% PI; 4B: 99.30\% vs.\ 98.91\%; every difference occurs at near-ties with top-2 logit margin $<0.17$).}
\label{tab:gsm8k}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{} l l r r r r @{}}
\toprule
\textbf{Model ($N$ problems)} & \textbf{Engine Mode} & \textbf{Strict Acc.\ ($c=1$)} & \textbf{Strict Acc.\ ($c=16$)} & \textbf{Generations Changed} & \textbf{Top-1 vs.\ \texttt{fp32}} \\
\midrule
Qwen3-1.7B (1,319) & Stock (FlashInfer) & 69.37\% & 68.69\% & 888 / 1,319 (67.3\%) & 99.14\% \\
Qwen3-1.7B (1,319) & \textbf{PI-Attn} & \textbf{69.07\%} & \textbf{69.07\%} & \textbf{0 / 1,319 (0.0\%)} & \textbf{99.30\%} \\
\midrule
Qwen3-8B (500) & Stock (FlashInfer) & 89.80\% & 89.80\% & 322 / 500 (64.4\%) & 99.53\% \\
Qwen3-8B (500) & \textbf{PI-Attn} & \textbf{90.20\%} & \textbf{90.20\%} & \textbf{0 / 500 (0.0\%)} & \textbf{99.53\%} \\
\bottomrule
\end{tabular}%
}
\end{table}

![**Rollout–trainer log-probability alignment per model (A800).** (a) Percentage of rollout tokens whose PyTorch trainer forward matches the serving engine bit-for-bit (PI-Attn: 100% across all packings; standard HF BF16: 1.0–40.8%). (b) Observed min–max range of importance ratios $r_t$ across 55,593 tokens: PI-Attn is identically $1.00\times$ everywhere, whereas standard BF16 exhibits extreme spikes ($0.49\times$ to $2.51\times$ across packings).](figs/fig6_rollout.pdf){width=100%}

![**Cross-GPU bitwise identity within Class A** (NVIDIA A800 `sm_80` vs.\ A10G/L4 and RTX 4090 D `sm_89`). PI-Attn achieves 100% agreement across kernel suites and full GSM8K evaluations.](figs/fig8_xgpu.pdf){width=100%}

![**Batch-1 decode step latency delta** (PI-Attn 4-warp canonical attention vs.\ Stock, node-level `nsys` trace). Canonical single-kernel attention saves 86–679 µs per decode step across every model scale by eliminating FlashInfer's separate merge launch.](figs/fig7_breakdown.pdf){width=100%}

\FloatBarrier

# Appendix B: Kernel Implementation Details and Ada (`sm_89`) Evaluation {-}

**Class-A Prefill Pipeline and Scheduling.** Prefill stages $Q, K, V$ in 128-byte-swizzled shared memory (64-row $Q$ tile plus double-buffered 32-key $K/V$ slots in 48 KB at 2 CTAs/SM, or an 80 KB 8-warp register-$Q$ variant when 128-row CTAs fill at least two SM waves). Unmasked interior tiles execute a branch-free loop with precomputed `cp.async` offsets and paired `bf16` rounding, cutting per-step warp instructions from 588 to ~450. Cycle-level microbenchmarks on `sm_80` show that each 32-key step takes ~2,100 cycles (1,040 cycles in Tensor Core MMAs overlapped with 177 contract-fixed floating-point softmax/rescaling instructions). Under Lemma 1, the launcher dynamically applies longest-first CTA ordering when $K/V$ fit in L2 (up to 24% faster), wave-boundary chunk cuts on multi-wave prefills over long prefixes (2–22% faster), and separate decode-kernel launches for trailing 1-token decode rows in mixed batches (5–38% faster on 27/28 shapes).

**Class-A Decode & Row-Invariant GEMM Families.** In the 4-warp decode kernel, warps split each 32-key tile into 8-key slices, exchange the tile maximum via shared memory, and accumulate four 16-column output slices; multi-group requests fold via a lock-free arrival counter with group index on the slowest grid axis. In the 16-warp `sm_80` decode kernel (1 CTA/SM, 138 KB shared memory), warp $t \in [0,15]$ computes $S^\top = K Q^\top$ for tile $t$ in two operand-swapped $16\times 8$ MMA chains while dedicated warps form prefix maxima, canonical row sums $\sum P$, and $O^\top = V^\top P^\top$. For projections, our five row-invariant GEMM families cover deep `cp.async` warp-streaming ($M \le 8$), DRAM-byte-balanced tiles, 32/64-stage CUTLASS `m16n8k16` tiles ($32\times 64$ to $128\times 256$), segment-parallel/warp-split kernels ($S > 1$), and CTA/warp-level deterministic stream-K.

**NVIDIA RTX 4090 D (`sm_89`) Evaluation.** Table \ref{tab:4090d} reports end-to-end greedy throughput on an Ada RTX 4090 D. PI-Attn passes 100% of `detcheck` checks across all four models and eight server configurations, outperforms the stock engine on **18 of 20 workloads**, and keeps TTFT within **−2.8% to +1.6%** from 512 to 30K tokens.

\begin{table}[h!]
\centering
\small
\caption{End-to-end greedy throughput delta of PI-Attn relative to the stock engine on NVIDIA RTX 4090 D (\texttt{sm\_89}).}
\label{tab:4090d}
\begin{tabular}{@{} l r r r r r @{}}
\toprule
\textbf{Model} & \textbf{64/256 c1} & \textbf{512/256 c8} & \textbf{512/256 c32} & \textbf{2048/256 c16} & \textbf{4096/128 c4} \\
\midrule
Qwen3-0.6B & $+2.0\%$ & $+8.5\%$ & $+4.3\%$ & $+5.6\%$ & $+5.8\%$ \\
Qwen3-1.7B & $-1.4\%$ & $+5.7\%$ & $+4.9\%$ & $+3.4\%$ & $+3.9\%$ \\
Qwen3-4B & $+0.1\%$ & $+5.9\%$ & $+6.1\%$ & $+3.2\%$ & $+3.0\%$ \\
Qwen3-8B & $-0.6\%$ & $+1.5\%$ & $+3.3\%$ & $+0.6\%$ & $+0.9\%$ \\
\bottomrule
\end{tabular}
\end{table}

# Appendix C: Universal Cross-Class Kernels via Exact Lattices {-}

When strict bitwise identity is required across *different* Tensor Core equivalence classes (e.g., Ampere + Ada + Hopper + CPU), **Exact-Lattice** programs project block operands onto power-of-two lattices satisfying $|c| + \sum_{i=1}^{16} |a_i b_i| \le 2^{24} u$ so no internal Tensor Core truncation ever fires on any $F \ge 24$ architecture (Table \ref{tab:lattice}).

\begin{table}[h!]
\centering
\small
\caption{Exact-Lattice cross-generation kernels verified 100\% bit-identical across Ampere (A100, A10/A10G), Ada (L4), Hopper (H100), and an IEEE CPU reference (\texttt{mma.sync} and \texttt{wgmma} alike). Overhead is relative to native PI/FlashInfer kernels streaming from DRAM.}
\label{tab:lattice}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{} l l l l l @{}}
\toprule
\textbf{Exact-Lattice Program} & \textbf{Mechanism (per $k16$ block)} & \textbf{Verified Platforms} & \textbf{Numerical Error} & \textbf{Latency Overhead} \\
\midrule
\textbf{XC-Triangular GEMM} & Operand block-lattice, 3 exact MMAs & A100, A10G, L4, H100, CPU & $0.35\text{--}1.5\times 10^{-6}$ & General-$M$ reference \\
\textbf{Portable Decode GEMM} & Register weight-slicing, 2 chained + 1 MMA & A100, A10, L4, H100, CPU & $0.34\text{--}1.7\times 10^{-6}$ & \textbf{+26\% to +40\%} ($M \le 16$) \\
\textbf{Offline-12b Decode GEMM} & One-time 12-bit checkpoint lattice, 2 MMAs & A100, A10, L4, H100, CPU & $1.654\times 10^{-3}$ (bf16) & \textbf{+7\% to +19\%} ($M \le 16$) \\
\textbf{XC Attention} & Power-of-two KV cache lattice ($QK^\top$, $PV$) & A100, L4, H100 (8.7K checks) & $2.19\times 10^{-3}$ vs.\ fp64 & \textbf{+1\% to +14\%} decode \\
\bottomrule
\end{tabular}%
}
\end{table}

**Portable Decode GEMM** slices weights in registers and chains two of three exact slice products inside the Tensor Core, streaming each weight byte once from DRAM; across 11 shapes $\times$ 6 variants (including $\pm 0$, subnormals, and $2^{\pm 20}$ adversarial exponents), it matches CPU `fp32` bit-for-bit across L4, A10, A100, and H100 with lower pre-rounding error than native chains ($0.34\text{--}1.7\times 10^{-6}$ vs.\ $1.3\text{--}8.1\times 10^{-6}$), and pre-rounding weights once offline onto a 12-bit block lattice cuts decode overhead to **7–19%**. **XC Attention** quantizes $Q, K, V$ onto power-of-two lattices on cache write, preserving 100% of Lemma 1's schedule invariances across 8,700 checks at 1–14% decode overhead (10–36% prefill). An integer-core variant (`I4 GEMM`, 4 `int8` products per `bf16` block) also achieves cross-class identity on L4/A100/H100 at 2.9–6.4× cuBLAS latency.

# Appendix D: Exact Gated DeltaNet Decode in vLLM {-}

![**Exact Gated DeltaNet (GDN) in vLLM 0.17.1.** (a) Bitwise agreement between decode and prefill `fp32` log-probabilities across 2,048 generated tokens on Qwen3.5-0.8B and 4B. (b) Serving throughput on Qwen3.5-4B relative to stock non-deterministic vLLM: replacing vLLM's built-in batch-invariant matmul ($0.27\times$ at $c=1$) with PI-Attn's row-invariant GEMM restores **0.98–1.08×** full stock speed.](figs/fig9_gdn.pdf){width=100%}

In hybrid GDN models, `fla`'s 64-token chunked forward (`chunk_gated_delta_rule`) is row-causal and bitwise resumable from any 64-token boundary state. Our `exact_decode` kernel stores each slot's exact `fp32` boundary state losslessly alongside its `bf16` copy in 4.33 MiB/layer (encoding the dropped lower 16 bits plus 1 rounding bit), evaluates each new token's chunk row incrementally, and folds the boundary state lazily only once row 64 is accepted. Speculative windows of up to 16 draft tokens evaluate additional rows of the identical chunk program, rolling back rejected drafts with zero state copies (whereas stock vLLM maintains $1 + K_{\text{draft}}$ full `fp32` states, consuming 10 MiB/layer for 4 drafts). At batch $\ge 32$, `exact_decode` runs **15–27% faster** per layer than vLLM's recurrent verifier because full state writes occur only once per 64 tokens. Integrated into vLLM 0.17.1 with PI-Attn's GEMM (`pi_attn::linear`), decode matches prefill bit-for-bit across all 2,048 generated tokens on Qwen3.5-0.8B and 4B under chunked prefill, preemption, and n-gram or MTP speculation at **0.98–1.08×** stock throughput (Figure 10; under chunked prefill + MTP, vLLM's own prompt log-probabilities shift, while decode hashes remain bitwise identical to unspeculated runs).

# Appendix E: Artifact Availability {-}

All source code, Ampere/Ada/Hopper CUDA & CUTLASS kernels (`llm_infer_rust/src/kernels/pi/`), the `llm_infer_rust` serving engine, PyTorch trainer bindings, `detcheck`, Exact Gated DeltaNet (`gdn/`), Tensor Core models and Exact-Lattice kernels (`xclass/`), raw benchmark logs across all six GPU architectures (`results/`), and build scripts (`eval/run_all.sh`, `eval/baselines.sh`, `paper/figures.py`, `paper/build.sh`) are available at `https://github.com/HaochengLu/pi-attn`.
