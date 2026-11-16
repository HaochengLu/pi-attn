# A800, round 6: the class-A prefill schedule, validated and measured (2026-11-16)

One A800-SXM4-80GB (sm_80), driver 535.129.03, CUDA 12.6, PyTorch 2.13, FlashInfer 0.6.18, commit 427e5c6
(`ENV.txt`; SM clock per minute in `run/clocks.log`). Everything here is written by one run of `run_r6.sh`
(stage log `run/run.out`), with the repository's own scripts and `PI_RESULTS=results/a800_r6`.

Round 6 changes scheduling and instruction selection of prefill on sm_80, and no bit:

- **Decode tail** (engine): the one-token requests at the end of a prefill-phase batch (the decode rows the
  scheduler appends to new prompts) run through the decode kernel on a grid for their longest request, when there
  are eight or more of them or they leave a single prompt of at least 1024 tokens (`decode_tail` in
  `llm_infer_rust/src/kernels/pi.rs`).
- **Wave cut** (engine): a single request whose launch would leave its last wave of SMs at most half full, over a
  prefix at least as long as the chunk, runs as the tokens whose CTAs fill whole waves and the rest
  (`prefill_wave_head`).
- **Grid order**: one-split launches that do not take the longest-first grid run their row tiles descending within
  a (request, kv head); longest-first only when the new tokens are at least an eighth of the keys.
- **`prefill_kernel`**: `cp.async` addresses formed once, P rounded once per pair, and for one-split launches of at
  least 1024 tokens a loop without the causal mask for the tiles every row of a warp sees
  (`prefill_kernel<QREG, WARPS, PLAIN>`).

How they were found, the attribution behind them, what was measured and not shipped, and what a contract change
would buy: [`archive/ceiling_probes/a800_r6/`](../../archive/ceiling_probes/a800_r6/README.md).

| paper | file | written by |
|---|---|---|
| §5.1 kernel hashes, kernel tests and mutants on the round-6 tree | `kernel_tests/`, `run/kernel_tests.log` | `eval/xgpu_hashes.py`, `eval/xgpu_compare.py`, `kernels/test_*.py` |
| §5.1 engine: A800 reference rollouts replayed, detcheck, four models | `engine_bits/`, `run/engine_bits.log` | `eval/xgpu_rollout.py`, `detcheck/detcheck.py --logprobs` |
| Rust unit tests, the engine's CUDA test of the PI backend | `rust_tests/` | `cargo test`, `cargo test -- --ignored pi_` |
| sm_90a builds of the engine and of the kernel extension (compiled, not run) | `sm90_build/` | `FLASHINFER_CUDA_ARCH=90 cargo build`, `PI_ATTN_ARCH=90` |
| §5.3 throughput matrix, Figure 5a | `e2e_matrix/SUMMARY.md`, one JSON per launch | `eval/e2e_matrix.sh`, `eval/summarize_matrix.py` |
| §5.3 components, Appendix A | `e2e_ablation/SUMMARY.md` | `eval/e2e_ablation.sh`, `eval/summarize_ablation.py` |
| §5.3 long context and shared prefixes, Figure 5b | `e2e_extended/SUMMARY.md` | `eval/e2e_extended.sh`, `eval/summarize_extended.py` |
| §5.3 high concurrency | `e2e_highconc/SUMMARY.md` | `eval/e2e_highconc.sh`, `eval/summarize_extended.py` |
| §5.3 attention kernels | `kernels/attention.{txt,json}`, `kernels/prefill_configs.txt` | `kernels/bench_attention.py`, `kernels/bench_prefill.py` |
| Appendix B prefill-phase batches with decode rows | `kernels/mixed_batch.txt` | `mixed_batch_bench.py` |
| Appendix B wave cut | `kernels/wave_split.txt` | `wave_split_bench.py` |
| Appendix B prefill grid order | `kernels/prefill_order.txt` | `prefill_order_bench.py` |

Not rerun (still `results/a800/`): the SGLang and vLLM baselines, GSM8K, fp32 agreement, the trainer's rescoring of
the rollouts, the cross-configuration check, the GEMM benches and the batch-1 traces. The decode planner bench of
`results/a800_r5/` was not rerun either: round 6 changes no decode kernel.

## Bits

`run/run.out`:

    HASHES gemm: PI identical 108/108; PI and vendor both identical 108/108; cases in new 108
    HASHES attention: PI identical 5/5; PI and vendor both identical 5/5; cases in new 5
    test_invariance exit=0      checks failed: 0 []
    test_directed exit=0        DIRECTED FAILURES: 0
    test_matrix exit=0          TOTAL FAILURES: 0
    test_gemm exit=0            INVARIANCE FAILURES: 0 []
    test_mutants exit=0         UNCAUGHT MUTANTS: 0 []
    cargo test exit=0 | test result: ok. 164 passed; 0 failed; 30 ignored
    cargo test --ignored pi_ exit=0 | test models::attention::backend::tests::pi_prefill_matches_reference_and_decode_bitwise ... ok
    engine sm_90a build exit=0
    kernel extension sm_90a build exit=0
    Qwen3-0.6B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 13063/13063 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-0.6B detcheck exit=0 | DETERMINISTIC
    Qwen3-1.7B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 14293/14293 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-1.7B detcheck exit=0 | DETERMINISTIC
    Qwen3-4B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 13975/13975 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-4B detcheck exit=0 | DETERMINISTIC
    Qwen3-8B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 14262/14262 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-8B detcheck exit=0 | DETERMINISTIC

The hashes are compared with `results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json` and the rollouts with
`results/a800/rollout/<model>.gen.json`. `test_invariance` forces the three prefill grid orders and both loop
variants on every prefill variant and split count. The engine's CUDA test prefills batches that end in one-token
requests and a request that is cut in two launches, and compares every row with a reference and with decode.
The shipped prefill kernel against the 98aff15 build, launch for launch (20,400 pairs over eight input regimes,
0 mismatches): `archive/ceiling_probes/a800_r6/logs/eq_final_A800.log`. Not run: class A beyond the A800 (on other
devices the launcher keeps the round-5 grid order, the masked loop alone and one launch per request; the
`cp.async` addressing and the single rounding of P are common code); the sm_90a build was compiled, not run.

## End to end

FlashInfer mode (FI: FlashInfer attention and cuBLAS) and PI (`--deterministic`) servers interleaved, a fresh server
per launch, medians of three launches; "before" is `results/a800_r5/` (the same scripts, an earlier session). SM
clock under load, one sample a minute: 1140-1410 MHz (46 of 255 samples below 1300).

**Throughput matrix** (`e2e_matrix/SUMMARY.md`; PI's output tok/s against FI's, before -> **after**):

| model, sampling | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B, greedy | +0.6% -> **+0.2%** | +16.9% -> **+16.9%** | +0.7% -> **+1.0%** | +4.9% -> **+6.8%** | +5.2% -> **+8.5%** |
| 0.6B, T = 0.8 | -0.6% -> **-0.9%** | +14.7% -> **+14.8%** | -0.9% -> **-0.9%** | +4.9% -> **+6.1%** | +4.7% -> **+7.7%** |
| 1.7B, greedy | +8.3% -> **+8.1%** | -0.5% -> **-0.5%** | -0.4% -> **-0.8%** | -0.5% -> **+0.5%** | +1.0% -> **+3.1%** |
| 1.7B, T = 0.8 | +7.6% -> **+7.3%** | -1.4% -> **-1.3%** | -1.6% -> **-1.6%** | -0.5% -> **+0.2%** | +0.5% -> **+2.7%** |
| 4B, greedy | +15.7% -> **+15.4%** | +5.7% -> **+5.8%** | -0.4% -> **-0.2%** | -1.8% -> **-1.2%** | -1.0% -> **+0.1%** |
| 4B, T = 0.8 | +15.6% -> **+15.4%** | +5.2% -> **+5.3%** | -0.9% -> **-0.6%** | -1.9% -> **-1.1%** | -1.0% -> **+0.2%** |
| 8B, greedy | +7.2% -> **+7.0%** | +3.4% -> **+3.4%** | +1.1% -> **+1.0%** | -1.0% -> **-0.5%** | -2.3% -> **-1.4%** |
| 8B, T = 0.8 | +7.1% -> **+6.8%** | +3.1% -> **+3.0%** | +0.8% -> **+0.8%** | -1.0% -> **-0.5%** | -2.3% -> **-1.5%** |

PI / FI over the 40 points: -1.6% to +16.9%, geometric mean +3.3% (before: -2.3% to +16.9%, +2.7%). The gains are
on the 2048- and 4096-token workloads, whose prefill-phase batches carry decode rows.

**Long context and shared prefixes** (`e2e_extended/SUMMARY.md`; change against FI; throughput is output tok/s, or
requests/s for the 16- and 4-token outputs):

| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| 1.7B | 64/256/c1 | +8.3% -> **+8.3%** | +6.5% -> **+13.1%** |
| 1.7B | 512/256/c32 | -0.2% -> **-0.4%** | -1.1% -> **+0.4%** |
| 1.7B | 2048/16/c8 | -0.9% -> **+0.5%** | -1.2% -> **+2.0%** |
| 1.7B | 4096/4/c4 | -2.2% -> **+3.2%** | +2.5% -> **-2.9%** |
| 1.7B | 4096/128/c8/3968 | +2.0% -> **+1.7%** | +2.2% -> **+2.3%** |
| 1.7B | 8192/64/c4/8000 | +1.7% -> **+1.1%** | +12.1% -> **+13.6%** |
| 1.7B | 16384/256/c2 | -3.4% -> **-0.2%** | +11.3% -> **+3.7%** |
| 1.7B | 30000/128/c1 | -9.1% -> **-4.3%** | +17.7% -> **+8.6%** |
| 8B | 64/256/c1 | +7.1% -> **+7.1%** | +3.1% -> **+4.5%** |
| 8B | 512/256/c32 | +1.2% -> **+1.0%** | +3.3% -> **+3.5%** |
| 8B | 2048/16/c8 | -3.9% -> **-3.4%** | +1.9% -> **+4.0%** |
| 8B | 4096/4/c4 | -5.2% -> **-3.5%** | +5.6% -> **+3.6%** |
| 8B | 4096/128/c8/3968 | +8.1% -> **+8.2%** | +3.5% -> **+3.3%** |
| 8B | 8192/64/c4/8000 | +5.6% -> **+5.8%** | +6.4% -> **+5.3%** |
| 8B | 16384/256/c1 | -11.2% -> **-9.2%** | +12.6% -> **+10.2%** |

**High concurrency** (`e2e_highconc/SUMMARY.md`):

| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| 0.6B | 512/256/c64 | +1.4% -> **+2.2%** | +7.7% -> **-1.1%** |
| 0.6B | 512/256/c128 | -1.8% -> **-0.3%** | +17.4% -> **+13.0%** |
| 0.6B | 128/512/c128 | +0.8% -> **+1.4%** | -0.2% -> **-0.0%** |
| 0.6B | 2048/128/c64 | -6.4% -> **+0.7%** | +8.6% -> **-0.0%** |
| 1.7B | 512/256/c64 | +0.8% -> **+1.1%** | +1.9% -> **+1.7%** |
| 1.7B | 512/256/c128 | -0.1% -> **+1.3%** | +1.4% -> **-1.4%** |
| 1.7B | 128/512/c128 | +0.8% -> **+1.0%** | -2.5% -> **-2.8%** |
| 1.7B | 2048/128/c64 | -1.9% -> **+3.5%** | +1.8% -> **-4.6%** |
| 4B | 512/256/c64 | +6.6% -> **+6.9%** | +4.0% -> **+3.4%** |
| 4B | 128/512/c64 | +7.7% -> **+7.1%** | +2.0% -> **+4.6%** |

Time to first token on the shortest rows moves by a few tenths of a millisecond between sessions and between the
first and later workloads of a server, which is several percent of a 6 ms first token. The 64/256/c1 row of the
long-context table is each server's first workload: in a three-way interleaved run (the round-5 binary, the
round-5 engine with the round-6 kernel, this tree; `archive/ceiling_probes/a800_r6/engine/ab5/`) all three took
6.5-6.8 ms there against FlashInfer mode's 5.7-5.9 and 6.3-6.4 against 6.0-6.1 on the same workload repeated, in
both server configurations, so that row's change from +6.5% to +13.1% is not a change of this round. The same holds
for the 0.6B rows at 64 and 128 requests of 512-token prompts, whose time to first token was +5.4% and +10.1% in
this tree's previous complete run and is -1.1% and +13.0% here.

## Kernels

us per call, one CUDA graph over a ring of per-layer K/V caches larger than L2.

Prefill, one launch (`kernels/attention.txt`, PI / FlashInfer over the 18 prefill shapes of `bench_attention.py`;
before = `results/a800_r5/kernels/attention.txt`):

| requests x new tokens (+ prefix) | 16 query heads: before | after | 32 query heads: before | after |
|---|---|---|---|---|
| 1 x 512 | 0.49 | 0.47 | 0.88 | 0.87 |
| 1 x 2048 | 0.93 | 0.89 | 1.09 | 1.04 |
| 1 x 8192 | 1.21 | 1.15 | 1.29 | 1.23 |
| 8 x 1024 | 1.34 | 1.29 | 1.43 | 1.38 |
| 32 x 512 | 1.52 | 1.49 | 1.45 | 1.44 |
| 1 x 512 + 8192 | 1.56 | 1.53 | 1.23 | 1.20 |
| 1 x 128 + 4096 | 1.30 | 1.29 | 1.34 | 1.33 |
| 1 x 128 + 16384 | 1.80 | 1.79 | 1.72 | 1.68 |
| 4 x 2048 + 2048 | 1.36 | 1.27 | 1.37 | 1.28 |

Prefill-phase batches (`kernels/mixed_batch.txt`, 14 batches x 2 head counts): one prefill launch 1.01-1.72x
FlashInfer's time on the whole batch, prefill + decode 0.89-1.29x; the two launches take 0.62-0.95x the one
launch's time on 27 of the 28 and 1.04x on one (four 512-token prompts with four decode rows of 300 keys, which
`decode_tail` leaves in one launch). One 2048-token prompt with 58 decode rows of 2100 keys, 16 heads: FlashInfer
500.6, one launch 722.6, two launches 458.2.

Wave cut (`kernels/wave_split.txt`, 17 shapes x 2 head counts): the rule cuts twelve, at 0.78-0.98x the one
launch's time, and leaves 22 in one launch. 2048 tokens over 14336 / 28000 keys with 16 heads: 2097.4 -> 1946.3 and
3937.2 -> 3591.8 us (FlashInfer 1704.1 and 3188.1).

Grid order (`kernels/prefill_order.txt`, 32 shapes x 2 head counts): the rule takes longest-first on 33 shapes
(0.77-1.02x the ascending grid's time; above 1.00 only 8 x 1024 and 16 x 512 with 32 heads, +1.1% and +2.4%, as in
round 5), descending on 23 (0.97-1.00x) and ascending on the 8 split launches.
