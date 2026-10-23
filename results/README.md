# Results

Every file here is written by `eval/run_all.sh` (stage logs in `run_all/`). Results of
earlier kernel versions are in `archive/stale_results/`; nothing in this directory predates
the engine commit named in the paper.

| paper | file | written by |
|---|---|---|
| §5 attention kernel table | `kernels/attention.{txt,json}`, `kernels/prefill_configs.txt` | `eval/kernel_benches.sh` (`kernels/bench_attention.py`, `kernels/bench_prefill.py`) |
| §5 projection GEMMs vs cuBLAS | `kernels/gemm_burst.txt` (M = 1..2048, launches over weight copies), `kernels/gemm_cold.txt`, `kernels/gemm.txt` (warm), `kernels/gemm_step_graph.txt` (one decode step's projections in a graph) | `eval/kernel_benches.sh` |
| §7.1 detcheck, four models | `detcheck/final_<model>.{log,json}` | `run_all.sh` detcheck stage (`detcheck/detcheck.py`) |
| §7.1 seven server configurations | `crossconfig/__deterministic_/`, `crossconfig/__attention_backend_flashinfer_/` | `eval/crossconfig.sh` |
| §7.1 / §7.3 SGLang and vLLM | `baselines/Qwen3-1.7B/SUMMARY.md` and the per-engine JSONs | `eval/baselines.sh` (SGLang and vLLM rows: run separately, see the top-level README) |
| §7.2 GSM8K | `accuracy/<model>/SUMMARY.md`, per-run logs | `eval/accuracy.sh`, `eval/compare_accuracy.py` |
| §7.2 agreement with fp32 | `fp32_agreement/<model>.json` | `eval/fp32_agreement.sh` (`eval/hf_agreement.py`) |
| §7.3 throughput matrix | `e2e_matrix/SUMMARY.md`, one JSON per launch | `eval/e2e_matrix.sh`, `eval/summarize_matrix.py` |
| §7.3 long context and shared prefixes | `e2e_extended/SUMMARY.md` | `eval/e2e_extended.sh`, `eval/summarize_extended.py` |
| §7.3 high concurrency | `e2e_highconc/SUMMARY.md` | `eval/e2e_highconc.sh`, `eval/summarize_extended.py` |
| §7.3 components (PI attention vs row-invariant GEMM) | `e2e_ablation/SUMMARY.md`, one JSON per launch | `eval/e2e_ablation.sh`, `eval/summarize_ablation.py` (engine 870c171) |
| §7.3 GEMM tuning time at startup | `e2e_matrix/tuning.txt` | `run_all.sh` (the engine's `PI GEMM tiles tuned` and `PI GEMM tile choice` log lines) |

Server logs (`*.server.log`) are kept on disk but not committed (`.gitignore`).
