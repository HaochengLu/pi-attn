# A800, round 5: the class-A scheduling changes, validated and measured (2026-11-14)

One A800-SXM4-80GB (sm_80), driver 535.129.03, CUDA 12.6, PyTorch 2.13, FlashInfer 0.6.18, commit b73b7c8
(`ENV.txt`; SM clock per minute in `run/clocks.log`). Everything here is written by one run of `run_r5.sh`
(stage log `run/run.out`), with the repository's own scripts and `PI_RESULTS=results/a800_r5`.

Round 5 changes scheduling on sm_80 only, and no bit:

- **Decode**: `decode16_kernel` (`llm_infer_rust/src/kernels/pi/pi_attn_decode16.cuh`) runs a whole fold group per
  16-warp CTA with the keys in the mma's 16-row operand; `decode_plan` takes it, per step, when every fold group of
  the longest running request has a CTA of its own in one wave, or one short group more.
- **Prefill**: `prefill_kernel`'s grid runs the longest row tiles first when the launch is one split and its K/V fit
  L2.

How they were found, what was measured on the way and what was not shipped:
[`archive/ceiling_probes/a800_r5/`](../../archive/ceiling_probes/a800_r5/README.md).

| paper | file | written by |
|---|---|---|
| §5.1 kernel hashes, kernel tests and mutants on the round-5 tree | `kernel_tests/`, `run/kernel_tests.log` | `eval/xgpu_hashes.py`, `eval/xgpu_compare.py`, `kernels/test_*.py` |
| §5.1 engine: A800 reference rollouts replayed, detcheck, four models | `engine_bits/`, `run/engine_bits.log` | `eval/xgpu_rollout.py`, `detcheck/detcheck.py --logprobs` |
| §5.3 throughput matrix, Figure 5a | `e2e_matrix/SUMMARY.md`, one JSON per launch | `eval/e2e_matrix.sh`, `eval/summarize_matrix.py` |
| §5.3 components, Appendix A | `e2e_ablation/SUMMARY.md` | `eval/e2e_ablation.sh`, `eval/summarize_ablation.py` |
| §5.3 long context and shared prefixes, Figure 5b | `e2e_extended/SUMMARY.md` | `eval/e2e_extended.sh`, `eval/summarize_extended.py` |
| §5.3 high concurrency | `e2e_highconc/SUMMARY.md` | `eval/e2e_highconc.sh`, `eval/summarize_extended.py` |
| §5.3 attention kernels | `kernels/attention.{txt,json}`, `kernels/prefill_configs.txt` | `kernels/bench_attention.py`, `kernels/bench_prefill.py` |
| §5.3 decode planner against `decode_kernel` and FlashInfer | `kernels/decode_plan.txt` (16 query heads), `kernels/decode_plan_hq32.txt` | `decode_plan_bench.py` |
| Appendix B prefill grid order | `kernels/prefill_order.txt` | `prefill_order_bench.py` |

Not rerun (still `results/a800/`, on the round-4 schedule): the SGLang and vLLM baselines, GSM8K, fp32 agreement,
the trainer's rescoring of the rollouts, the cross-configuration check, the GEMM benches and the batch-1 traces.
Round 5 changes no GEMM and no bit, so only their PI timings are older; on the baseline workloads the round-5
schedule is at or above the round-4 schedule in the same binary (`archive/ceiling_probes/a800_r5/`,
`component_ablation/schedule_ab/SUMMARY.md`).

## Bits

`run/run.out`:

    HASHES gemm: PI identical 108/108; PI and vendor both identical 108/108; cases in new 108
    HASHES attention: PI identical 5/5; PI and vendor both identical 5/5; cases in new 5
    test_invariance exit=0      checks failed: 0 []
    test_directed exit=0        DIRECTED FAILURES: 0
    test_matrix exit=0          TOTAL FAILURES: 0
    test_gemm exit=0            INVARIANCE FAILURES: 0 []
    test_mutants exit=0         UNCAUGHT MUTANTS: 0 []
    Qwen3-0.6B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 13063/13063 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-0.6B detcheck exit=0 | DETERMINISTIC
    Qwen3-1.7B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 14293/14293 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-1.7B detcheck exit=0 | DETERMINISTIC
    Qwen3-4B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 13975/13975 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-4B detcheck exit=0 | DETERMINISTIC
    Qwen3-8B xgpu_rollout exit=0 | 64 rollouts: 64 with identical token ids; 14262/14262 log-probabilities bitwise equal | ROLLOUTS IDENTICAL TO REFERENCE: True
    Qwen3-8B detcheck exit=0 | DETERMINISTIC

The hashes are compared with `results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json` and the rollouts with
`results/a800/rollout/<model>.gen.json`, both written by the A800 run of `results/a800/`. `test_invariance` forces
both prefill grid orders on every prefill variant and split count, and `decode16_kernel` with and without helper
CTAs on every position of every test sequence (`DECODE_VARIANTS [1, 3, 4]`). The mutants include two of
`decode16_kernel` (`kernel_tests/test_mutants.log`):

    mutant decode16_d_unfused: 21 failed checks, kinds ['path']
    mutant decode16_pv_k_reversed: 994 failed checks, kinds ['path']

Each engine above captured 4 decode graphs of a second launch plan (`decode16_kernel` for graphs of 1, 2, 4 and 8
requests whose longest request has at most 6784, 3200, 1664 and 768 keys), so the rollouts and detcheck ran both
decode kernels. Not run: the Rust unit tests; class A beyond the A800 (the kernels of sm_86
and sm_89 devices are untouched: `decode_plan` and `prefill_longest_first` return the old choice there); the
sm_90a build was compiled on this machine (`pi_launch.cu`, `pi_attn_torch.cu`), not run.

## Kernels

us per call, one CUDA graph over a ring of per-layer K/V caches larger than L2.

Decode (`kernels/decode_plan.txt`: 16 query heads, 8 kv heads, graphs sized for 8192 keys; `v1` = `decode_kernel`,
the only class-A decode kernel before; `v0` = the planner):

| requests \ keys | 96 | 192 | 320 | 512 | 768 | 1024 | 2176 | 4200 | 8000 |
|---|---|---|---|---|---|---|---|---|---|
| 1, FlashInfer | 6.8 | 11.0 | 11.4 | 11.7 | 12.3 | 12.8 | 15.4 | 19.2 | 30.5 |
| 1, PI: before -> after | 6.3 -> **6.0** (0.88) | 10.0 -> **7.5** (0.68) | 15.9 -> **9.9** (0.87) | 23.4 -> **12.3** (1.05) | 24.8 -> **14.5** (1.18) | 24.9 -> **14.7** (1.15) | 26.4 -> **16.3** (1.06) | 30.1 -> **20.1** (1.05) | 33.8 (1.11) |
| 2, FlashInfer | 7.0 | 11.6 | 12.1 | 12.9 | 13.6 | 14.6 | 19.2 | 31.0 | 51.0 |
| 2, PI: before -> after | 7.4 -> **6.1** (0.87) | 12.1 -> **8.3** (0.72) | 18.1 -> **10.1** (0.83) | 26.6 -> **12.7** (0.98) | 28.2 -> **15.4** (1.13) | 28.5 -> **15.7** (1.08) | 29.7 -> **19.9** (1.04) | 33.9 (1.09) | 50.4 (0.99) |
| 4, FlashInfer | 7.8 | 12.4 | 13.5 | 14.7 | 16.1 | 18.2 | 31.5 | 52.2 | 88.0 |
| 4, PI: before -> after | 8.2 -> **7.1** (0.91) | 12.4 -> **8.7** (0.70) | 18.3 -> **10.8** (0.80) | 27.1 -> **14.0** (0.95) | 28.8 -> **17.3** (1.07) | 29.2 -> **19.1** (1.05) | 33.7 (1.07) | 53.3 (1.02) | 89.3 (1.01) |
| 8, FlashInfer | 8.1 | 14.6 | 15.8 | 18.6 | 23.4 | 30.0 | 53.7 | 92.5 | 162.7 |
| 8, PI: before -> after | 8.5 -> **7.8** (0.96) | 12.8 -> **9.9** (0.68) | 19.0 -> **12.8** (0.81) | 27.9 -> **17.2** (0.92) | 30.6 -> **25.1** (1.07) | 32.4 (1.08) | 54.9 (1.02) | 91.6 (0.99) | 170.3 (1.05) |
| 13, FlashInfer | 8.6 | 16.0 | 19.2 | 25.1 | 34.9 | 43.4 | 81.2 | 142.2 | 257.7 |
| 13, PI: before -> after | 9.0 -> **8.6** (1.00) | 13.7 -> **11.7** (0.73) | 20.2 -> **15.5** (0.81) | 29.1 -> **22.7** (0.90) | 36.2 -> **34.6** (0.99) | 41.7 (0.96) | 78.0 (0.96) | 150.3 (1.06) | 270.8 (1.05) |
| 16, FlashInfer | 9.3 | 17.2 | 21.3 | 30.4 | 41.1 | 52.3 | 95.7 | 172.6 | 317.8 |
| 16, PI: before -> after | 9.5 (1.02) | 14.3 (0.83) | 20.9 (0.98) | 30.9 (1.02) | 41.4 (1.01) | 52.1 (1.00) | 93.2 (0.97) | 178.7 (1.04) | 328.3 (1.03) |
| 32, FlashInfer | 12.0 | 24.6 | 37.7 | 51.5 | 72.0 | 91.1 | 179.4 | 331.0 | 613.2 |
| 32, PI: before -> after | 13.3 (1.11) | 21.5 (0.87) | 32.8 (0.87) | 49.9 (0.97) | 72.1 (1.00) | 90.0 (0.99) | 180.9 (1.01) | 339.3 (1.03) | 640.9 (1.05) |

PI cells: us per layer of `decode_kernel` -> **the planner** where it takes `decode16_kernel`, one number where it keeps `decode_kernel`; in brackets the planner's time over the better FlashInfer kernel's.

Where the planner takes `decode16_kernel` (36 of the 80 points: 1 to 13 requests) it runs at 0.48-0.96x
`decode_kernel`'s time and 0.68-1.21x the better FlashInfer kernel's, where `decode_kernel` ran at 0.86-2.10x; with
32 query heads 0.48-0.99x and 0.58-1.22x (0.72-2.10x). The other 44 points are unchanged (0.83-1.15x FlashInfer).

Prefill (`kernels/attention.txt` against `results/a800/kernels/attention.txt`, PI / FlashInfer over the 18 prefill
shapes of `bench_attention.py`):

| requests x new tokens (+ prefix) | 16 query heads: before | after | 32 query heads: before | after |
|---|---|---|---|---|
| 1 x 512 | 0.55 | 0.49 | 1.12 | 0.88 |
| 1 x 2048 | 1.22 | 0.93 | 1.28 | 1.09 |
| 1 x 8192 | 1.31 | 1.21 | 1.34 | 1.29 |
| 8 x 1024 | 1.42 | 1.34 | 1.42 | 1.43 |
| 32 x 512 | 1.52 | 1.52 | 1.46 | 1.45 |
| 1 x 512 + 8192 | 1.55 | 1.56 | 1.22 | 1.23 |
| 1 x 128 + 4096 | 1.30 | 1.30 | 1.34 | 1.34 |
| 1 x 128 + 16384 | 1.80 | 1.80 | 1.72 | 1.72 |
| 4 x 2048 + 2048 | 1.36 | 1.36 | 1.37 | 1.37 |

`kernels/prefill_order.txt` has the rule against the ascending grid and against longest-first everywhere on 54
shapes: 0.76-1.02x of ascending.

## End to end

FlashInfer mode (FI: FlashInfer attention and cuBLAS) and PI (`--deterministic`) servers interleaved, a fresh server
per launch, medians of three launches; "before" is `results/a800/` (the same scripts on the round-4 schedule, an
earlier session). SM clock under load, one sample a minute: 1185-1410 MHz here (17 of 200 samples below 1300),
1170-1410 MHz there (35 of 412).

**Throughput matrix** (`e2e_matrix/SUMMARY.md`; PI's output tok/s against FI's, before -> **after**):

| model, sampling | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| 0.6B, greedy | -3.4% -> **+0.6%** | +7.1% -> **+16.9%** | +0.6% -> **+0.7%** | +4.5% -> **+4.9%** | +4.3% -> **+5.2%** |
| 0.6B, T = 0.8 | -4.0% -> **-0.6%** | +6.8% -> **+14.7%** | -1.1% -> **-0.9%** | +4.1% -> **+4.9%** | +3.9% -> **+4.7%** |
| 1.7B, greedy | +5.7% -> **+8.3%** | -5.8% -> **-0.5%** | -0.4% -> **-0.4%** | -0.7% -> **-0.5%** | +0.2% -> **+1.0%** |
| 1.7B, T = 0.8 | +5.0% -> **+7.6%** | -5.9% -> **-1.4%** | -1.6% -> **-1.6%** | -1.1% -> **-0.5%** | +0.3% -> **+0.5%** |
| 4B, greedy | +13.8% -> **+15.7%** | +2.5% -> **+5.7%** | -0.6% -> **-0.4%** | -2.2% -> **-1.8%** | -1.5% -> **-1.0%** |
| 4B, T = 0.8 | +14.2% -> **+15.6%** | +2.2% -> **+5.2%** | -1.1% -> **-0.9%** | -2.2% -> **-1.9%** | -1.7% -> **-1.0%** |
| 8B, greedy | +6.2% -> **+7.2%** | +1.1% -> **+3.4%** | +1.2% -> **+1.1%** | -1.1% -> **-1.0%** | -2.5% -> **-2.3%** |
| 8B, T = 0.8 | +6.0% -> **+7.1%** | +0.9% -> **+3.1%** | +0.9% -> **+0.8%** | -1.1% -> **-1.0%** | -2.5% -> **-2.3%** |

PI / FI over the 40 points: -2.3% to +16.9%, geometric mean +2.7% (before: -5.9% to +14.2%, +1.2%). FI's own
median throughput differs by at most 0.4% between the two sessions on every point, PI's by -0.1% to +9.1%: the
changes are PI's. The batch-1 and 8-request columns are where `decode16_kernel` runs
(decode graphs of 1, 2, 4 and 8 requests); the 16- and 32-request columns keep `decode_kernel`.

**Components** (`e2e_ablation/SUMMARY.md`): PI attention alone now gains 4.2-7.2% at batch 1 and 0.8-3.8% at 8
requests and costs at most 1.0% (1.7B 4096/128 c4); before, it cost 3.8% on 1.7B at 8 requests (the archive's
`component_ablation/before_bf73578/`). The GEMM alone ranges from -6.4% (0.6B at batch 1) to +15.2% (0.6B at 8
requests) and costs at most 1.4% at 32-128 requests. Time to first token at 128 requests on 0.6B is +25% with PI
attention alone (`0.6B-c128`): batched prefill, whose launches exceed L2 and keep the ascending grid, is PI's
remaining cost on this GPU.

**Long context and shared prefixes** (`e2e_extended/SUMMARY.md`; change against FI, before -> **after**; throughput is
output tok/s, or requests/s for the 16- and 4-token outputs):

| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| 1.7B | 64/256/c1 | +5.6% -> **+8.3%** | +8.8% -> **+6.5%** |
| 1.7B | 512/256/c32 | -0.3% -> **-0.2%** | -0.6% -> **-1.1%** |
| 1.7B | 2048/16/c8 | -1.7% -> **-0.9%** | +0.1% -> **-1.2%** |
| 1.7B | 4096/4/c4 | -2.7% -> **-2.2%** | +2.9% -> **+2.5%** |
| 1.7B | 4096/128/c8/3968 | +1.6% -> **+2.0%** | +5.4% -> **+2.2%** |
| 1.7B | 8192/64/c4/8000 | +1.2% -> **+1.7%** | +12.3% -> **+12.1%** |
| 1.7B | 16384/256/c2 | -3.7% -> **-3.4%** | +12.2% -> **+11.3%** |
| 1.7B | 30000/128/c1 | -9.4% -> **-9.1%** | +18.3% -> **+17.7%** |
| 8B | 64/256/c1 | +6.2% -> **+7.1%** | +3.6% -> **+3.1%** |
| 8B | 512/256/c32 | +1.1% -> **+1.2%** | +3.6% -> **+3.3%** |
| 8B | 2048/16/c8 | -4.4% -> **-3.9%** | +5.2% -> **+1.9%** |
| 8B | 4096/4/c4 | -5.8% -> **-5.2%** | +6.3% -> **+5.6%** |
| 8B | 4096/128/c8/3968 | +8.1% -> **+8.1%** | +3.9% -> **+3.5%** |
| 8B | 8192/64/c4/8000 | +5.5% -> **+5.6%** | +6.3% -> **+6.4%** |
| 8B | 16384/256/c1 | -11.6% -> **-11.2%** | +13.2% -> **+12.6%** |

**High concurrency** (`e2e_highconc/SUMMARY.md`):

| model | workload (in / out / concurrency [/ shared prefix]) | throughput: before -> after | TTFT: before -> after |
|---|---|---|---|
| 0.6B | 512/256/c64 | +1.2% -> **+1.4%** | +6.8% -> **+7.7%** |
| 0.6B | 512/256/c128 | -2.0% -> **-1.8%** | +18.9% -> **+17.4%** |
| 0.6B | 128/512/c128 | +1.4% -> **+0.8%** | -2.5% -> **-0.2%** |
| 0.6B | 2048/128/c64 | -8.0% -> **-6.4%** | +11.3% -> **+8.6%** |
| 1.7B | 512/256/c64 | +0.7% -> **+0.8%** | +0.8% -> **+1.9%** |
| 1.7B | 512/256/c128 | -0.3% -> **-0.1%** | +2.3% -> **+1.4%** |
| 1.7B | 128/512/c128 | +0.9% -> **+0.8%** | -0.1% -> **-2.5%** |
| 1.7B | 2048/128/c64 | -2.9% -> **-1.9%** | +3.4% -> **+1.8%** |
| 4B | 512/256/c64 | +6.5% -> **+6.6%** | +2.8% -> **+4.0%** |
| 4B | 128/512/c64 | +7.5% -> **+7.7%** | +2.8% -> **+2.0%** |

Time to first token still rises 11-18% on 16K-30K-token prompts: their chunks' K/V exceed L2 from about 10K keys
on, where the grid stays ascending, and prefill attention there takes 1.33-1.36x FlashInfer's time in either
order (`kernels/prefill_order.txt`, 2048 new tokens over 14336).
