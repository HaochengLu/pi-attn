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

`a800/baselines_38c32da_tokens/` keeps the previous A800 run's detcheck table (engine 38c32da,
SGLang and vLLM 0.11.2, token ids only), which Section 7.1 cites for token-level counts.
`a800/gdn/` is written by the scripts in `gdn/` (Section 8; see `gdn/README.md`).

| paper | file under `a800/` | written by |
|---|---|---|
| §7.1 detcheck, four models (tokens and log-probabilities) | `detcheck/final_<model>.{log,json}` | `run_all.sh` detcheck stage (`detcheck/detcheck.py --logprobs`) |
| §7.1 seven server configurations | `crossconfig/__deterministic_/`, `crossconfig/__attention_backend_flashinfer_/` | `eval/crossconfig.sh` |
| §7.1 / §7.4 SGLang and vLLM | `baselines/Qwen3-1.7B/SUMMARY.md` and the per-engine JSONs | `eval/baselines.sh` (engines from `BASELINE_ENGINES`) |
| §7.1 GSM8K | `accuracy/<model>/SUMMARY.md`, per-run logs | `eval/accuracy.sh`, `eval/compare_accuracy.py` |
| §7.2 kernel hashes on both GPUs | `xgpu/` | `eval/xgpu_hashes.py`, `eval/xgpu_compare.py` |
| §7.2 agreement with fp32 | `fp32_agreement/<model>.json` | `eval/fp32_agreement.sh` (`eval/hf_agreement.py`) |
| §7.3 rollout log-probabilities | `rollout/<model>.json` (report), `rollout/<model>.log`, `rollout/<model>.gen.json` | `eval/rollout.sh` (`eval/rollout_logprobs.py`) |
| §7.4 throughput matrix | `e2e_matrix/SUMMARY.md`, one JSON per launch | `eval/e2e_matrix.sh`, `eval/summarize_matrix.py` |
| §7.4 components (PI attention vs row-invariant GEMM) | `e2e_ablation/SUMMARY.md`, one JSON per launch | `eval/e2e_ablation.sh`, `eval/summarize_ablation.py` |
| §7.4 long context and shared prefixes | `e2e_extended/SUMMARY.md` | `eval/e2e_extended.sh`, `eval/summarize_extended.py` |
| §7.4 high concurrency | `e2e_highconc/SUMMARY.md` | `eval/e2e_highconc.sh`, `eval/summarize_extended.py` |
| §7.4 attention kernels | `kernels/attention.{txt,json}`, `kernels/prefill_configs.txt` | `eval/kernel_benches.sh` (`kernels/bench_attention.py`, `kernels/bench_prefill.py`) |
| §7.4 projection GEMMs vs cuBLAS | `kernels/gemm_burst.txt` (M = 1..2048, launches over weight copies), `kernels/gemm_cold.txt`, `kernels/gemm.txt` (warm), `kernels/gemm_step_graph.txt` (one decode step's projections in a graph) | `eval/kernel_benches.sh` |
| §6 GEMM tuning time at startup | `e2e_matrix/tuning.txt` | `run_all.sh` (the engine's `PI GEMM tiles tuned` and `PI GEMM tile choice` log lines) |
| §7.5 batch-1 traces | `nsys/b1_compare.txt` | `scripts/nsys_decode.sh`, `scripts/nsys_compare.py` |
| §7.7 kernel tests and mutants | `run_all/kernel_tests.log` | `kernels/test_*.py` |
| §9 tensor-parallel degree | `kernels/bench_tp.log` | `kernels/bench_tp.py`, `kernels/test_tp_model.py` |

The paper's figures are drawn from these files by `paper/figures.py` (`cd paper && python figures.py ../results figs`).

Server logs (`*.server.log`) are kept on disk but not committed (`.gitignore`).

## RTX 4090 D (paper Section 7.6)

The top-level directories (`accuracy/`, `baselines/`, `e2e_*`, `kernels/`, ...) are the RTX
4090 D results of engine fdc184b (ablation 870c171), with the same layout. The attention
kernel timings in `kernels/attention.txt` there timed one launch at a time, host launch
overhead included, and are superseded by the A800 method; the 4090 D was no longer available
to rerun them. Results of earlier kernel versions are in `archive/stale_results/`.
