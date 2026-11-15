# Results

The paper's evaluation is on the A800 (`a800/`). Every file there is written by one
`eval/run_all.sh` run on an A800-SXM4-80GB (outer commit a51ba7e, engine 34203cf; environment,
clocks and commits in `a800/ENV.txt` and `a800/run_all/clocks.log`), with two kinds of
exception:
- That run crashed the deterministic engine at 128 running requests on Qwen3-0.6B (a CUTLASS
  scratch buffer freed under captured CUDA graphs; fixed in engine 1ee2bc7, which changes no
  kernel). `logs/close/followup.sh` reran on 1ee2bc7 everything the fix could touch and every
  bitwise check: kernel tests, `kernels/prefill_configs.txt`, the deterministic
  cross-configuration check, detcheck, the high-concurrency and ablation runs, the vLLM 0.17.1
  batch-invariant baseline and the rollout log-probabilities (stage log
  `a800/run_all/followup.out`).
- Cross-GPU files, computed from both GPUs' results:
  - `a800/xgpu/compare_4090D_A800.txt`: `python ../eval/xgpu_compare.py ../results/xgpu/NVIDIA_GeForce_RTX_4090_D.json ../results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json`
  - `a800/accuracy/xgpu_4090D_vs_A800.md`: `python xgpu_accuracy.py ../results/accuracy/<model> ../results/a800/accuracy/<model>` (from `eval/`), for 1.7B and 8B.

`a800_r5/` (see its README) is a later A800 session on the tree with the round-5 class-A scheduling changes
(a 16-warp decode kernel with the mma operands swapped, chosen per step; longest-first prefill grids): the kernel
tests, hashes, engine rollouts and detcheck again, the attention benches, and the §5.3 end-to-end stages. The
paper's §5.3 throughput, component, long-context and high-concurrency numbers, Figure 5 and the attention kernel
numbers come from `a800_r5/`; everything else in the table below is `a800/`.

`a800/baselines_38c32da_tokens/` keeps the previous A800 run's detcheck table (engine 38c32da,
SGLang and vLLM 0.11.2, token ids only), which Section 7.1 cites for token-level counts.
`a800/gdn/` is written by the scripts in `gdn/` (§6.1; see `gdn/README.md`).

| paper | file under `a800/` | written by |
|---|---|---|
| §5.1 detcheck, four models (tokens and log-probabilities) | `detcheck/final_<model>.{log,json}` | `run_all.sh` detcheck stage (`detcheck/detcheck.py --logprobs`) |
| §5.1 seven server configurations | `crossconfig/__deterministic_/`, `crossconfig/__attention_backend_flashinfer_/` | `eval/crossconfig.sh` |
| §5.1 / §5.3 SGLang and vLLM | `baselines/Qwen3-1.7B/SUMMARY.md` and the per-engine JSONs | `eval/baselines.sh` (engines from `BASELINE_ENGINES`) |
| §5.1 GSM8K | `accuracy/<model>/SUMMARY.md`, per-run logs | `eval/accuracy.sh`, `eval/compare_accuracy.py` |
| §5.1 kernel hashes, A800 vs RTX 4090 D | `xgpu/` | `eval/xgpu_hashes.py`, `eval/xgpu_compare.py` |
| §5.1 agreement with fp32 | `fp32_agreement/<model>.json` | `eval/fp32_agreement.sh` (`eval/hf_agreement.py`) |
| §5.2 rollout log-probabilities | `rollout/<model>.json` (report), `rollout/<model>.log`, `rollout/<model>.gen.json` | `eval/rollout.sh` (`eval/rollout_logprobs.py`) |
| §5.3 throughput matrix, components, long context, high concurrency, attention kernels: current numbers | `../a800_r5/` (`e2e_matrix/`, `e2e_ablation/`, `e2e_extended/`, `e2e_highconc/`, `kernels/`) | `../a800_r5/run_r5.sh` (the scripts below with `PI_RESULTS`) |
| §3, Appendix B: class-A decode kernel choice, prefill grid order, mma operand symmetry | `../a800_r5/kernels/decode_plan.txt`, `../a800_r5/kernels/prefill_order.txt`, `../../archive/ceiling_probes/a800_r5/` | see their READMEs |
| §5.3 throughput matrix (before round 5) | `e2e_matrix/SUMMARY.md`, one JSON per launch | `eval/e2e_matrix.sh`, `eval/summarize_matrix.py` |
| §5.3 components (PI attention vs row-invariant GEMM; before round 5) | `e2e_ablation/SUMMARY.md`, one JSON per launch | `eval/e2e_ablation.sh`, `eval/summarize_ablation.py` |
| §5.3 long context and shared prefixes (before round 5) | `e2e_extended/SUMMARY.md` | `eval/e2e_extended.sh`, `eval/summarize_extended.py` |
| §5.3 high concurrency (before round 5) | `e2e_highconc/SUMMARY.md` | `eval/e2e_highconc.sh`, `eval/summarize_extended.py` |
| §5.3 attention kernels (before round 5) | `kernels/attention.{txt,json}`, `kernels/prefill_configs.txt` | `eval/kernel_benches.sh` (`kernels/bench_attention.py`, `kernels/bench_prefill.py`) |
| §5.3 projection GEMMs vs cuBLAS | `kernels/gemm_burst.txt` (M = 1..2048, launches over weight copies), `kernels/gemm_cold.txt`, `kernels/gemm.txt` (warm), `kernels/gemm_step_graph.txt` (one decode step's projections in a graph) | `eval/kernel_benches.sh` |
| §4 GEMM tuning time at startup | `e2e_matrix/tuning.txt` | `run_all.sh` (the engine's `PI GEMM tiles tuned` and `PI GEMM tile choice` log lines) |
| §5.3 batch-1 traces | `nsys/b1_compare.txt` | `scripts/nsys_decode.sh`, `scripts/nsys_compare.py` |
| §5.4 kernel tests and mutants | `run_all/kernel_tests.log` | `kernels/test_*.py` |
| §6.2 tensor-parallel degree | `kernels/bench_tp.log` | `kernels/bench_tp.py`, `kernels/test_tp_model.py` |
| §7 Table 2, kernel hashes and suites on A100, A10G, L4, H100 | `../modal/<GPU>/xgpu_hashes.json`, `../modal/suite_<GPU>.log` | `eval/xgpu_hashes.py`, `eval/xgpu_compare.py`, `kernels/test_*.py` |
| §7 Table 2, engine rollout replay, detcheck, trainer (A10G, H100) | `../modal/<GPU>/engine.log` | `eval/xgpu_rollout.py`, `detcheck/detcheck.py --logprobs`, `eval/rollout_logprobs.py` |
| §7 Hopper kernels in the engine (H100) | `../modal/H100_sm90/` (suites, hashes, rollouts against the pre-change engine, end to end) | same |
| §7 Table 4, H100 same-GPU baselines (SGLang, vLLM 0.17.1, PI; detcheck and loadgen) and PI vs FlashInfer at 64–256 requests | `../modal/H100_baselines/` (`SUMMARY.md`, `hc.*.json`, `run.log`) | `eval/baselines.sh` via `run_h100_baselines.sh` |
| §7 H100 decode/prefill attention probes (not shipped) | `../../archive/ceiling_probes/h100_decode/` | see its README |
| §7 Table 3 final row, Table 4 final PI row, H100 kernel benches and bit-identity of the final tree | `../modal/H100_round4/` (`README.md`, `all_H100.log`, `classA_A10G.log`) | `run.sh`, `dec_grid.py` there |
| §7 intermediate integrated run (round 3 tree) | `../modal/H100_round3/` | same |
| §7 "Where the H100 gap was": component ablation (PI attention only / PI GEMM only) and mid-M GEMM bench | `../../archive/ceiling_probes/h100_component_ablation/` | `abl.sh`, `midm.py` |
| H100 GEMM tiles (ids 103–120): probes, cold tables, negatives | `../../archive/ceiling_probes/h100_gemm/`, `h100_gemm_r4/` | see their READMEs |
| H100 decode (`decode90_kernel`, `decode_plan`), cluster probes and negatives | `../../archive/ceiling_probes/h100_decode_cluster/`, `h100_decode_r4/` | see their READMEs |
| Independent ceiling review (per-SM intake, operand swap) | `../../archive/ceiling_probes/h100_review/` | see its README |
| H100 prefill kernel (`prefill90w_kernel`): before/after against FlashInfer, ablations, probes not shipped | `../../archive/ceiling_probes/h100_prefill/` | see its README |
| §7 tensor-core models and exact-lattice programs | `../../xclass/` | see `xclass/README.md` |

The paper's figures are drawn from these files by `paper/figures.py` (`cd paper && python figures.py ../results figs`).

Server logs (`*.server.log`) are kept on disk but not committed (`.gitignore`).

## RTX 4090 D (paper Section 7.6)

The top-level directories (`accuracy/`, `baselines/`, `e2e_*`, `kernels/`, ...) are the RTX
4090 D results of engine fdc184b (ablation 870c171), with the same layout. The attention
kernel timings in `kernels/attention.txt` there timed one launch at a time, host launch
overhead included, and are superseded by the A800 method; the 4090 D was no longer available
to rerun them. Results of earlier kernel versions are in `archive/stale_results/`.
