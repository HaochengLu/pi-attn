# PI-Attn: bitwise-deterministic LLM serving with every optimization on

This directory holds the research project built on
[llm_infer_rust](https://github.com/MabinogiX/llm_infer_rust) (branch `pi-attn` in
`llm_infer_rust/`). It makes the engine's outputs a bitwise function of the model and
the request alone — independent of batching, concurrency, prefix caching, chunked
prefill, decode vs re-prefill, CUDA graphs, page size and restarts — without turning
any of those optimizations off.

| Path | What |
|---|---|
| `llm_infer_rust/` (branch `pi-attn`) | Engine with `--deterministic`; kernels in `src/kernels/pi/`, docs in `docs/deterministic-mode.md` |
| `kernels/` | PyTorch bindings, invariance tests (`test_invariance.py`, `test_matrix.py`, `test_directed.py`, `test_gemm.py`) and microbenchmarks for the kernels (sources, including the GEMM id table `gemm_dispatch.cuh`, symlinked from the engine) |
| `gdn/` | Exact Gated DeltaNet decode for hybrid models: decode bitwise equal to fla's chunked training forward (`exact_decode.py`), its test, benchmark and experiments, and its vLLM 0.17.1 integration (`gdn/vllm/`: decode bitwise the engine's prefill, speculation on or off); findings in `gdn/README.md` |
| `detcheck/detcheck.py` | Black-box determinism checker for any OpenAI-compatible server |
| `eval/` | Serving benchmark (`loadgen.py`), cross-configuration check (`crossconfig.sh`), end-to-end matrices (`e2e_matrix.sh`, `e2e_extended.sh`), same-GPU SGLang/vLLM baselines (`baselines.sh`), GSM8K reproducibility (`accuracy.sh`, `compare_accuracy.py`), fp32 agreement (`fp32_agreement.sh`), kernel benchmarks (`kernel_benches.sh`), and table generators (`summarize_*.py`) |
| `results/` | Results of the latest `eval/run_all.sh`; `results/README.md` maps each paper table to its files and script |
| `paper/` | Paper draft |
| `requirements/` | Exact package lists of the three Python environments and the model file hashes |
| `archive/` | Earlier kernel iterations, ceiling probes (`archive/ceiling_probes/README.md`: every measured attempt, kept or rejected), superseded results |

## Environment (self-contained, reproducible)

Everything, caches included, is installed under `env/` by `setup_env.sh` (idempotent);
nothing is written to the host system or `$HOME`. The checkout can live anywhere: `env.sh`
finds it from its own path. Python packages come from the lock files in `requirements/`,
the models from ModelScope (checked against `requirements/models.sha256`), and
`VERSIONS.txt` is regenerated on every run.

```bash
# llm_infer_rust/: the engine on branch pi-attn (upstream main e7e2806 plus this work; the
# branch is not published yet, so copy the checkout along with this directory)
bash setup_env.sh            # CUDA 12.6, gcc 13, Rust 1.95, Python 3.12, three venvs, Qwen3 models
source env.sh                # activate
cd llm_infer_rust && FLASHINFER_CUDA_ARCH=89 cargo build --release   # 80 for A100/A800
```

Download sources default to mirrors reachable from mainland China; `setup_env.sh` lists
the variables that point them elsewhere.

Notes recorded while setting up the RTX 4090 D dev box (driver 535 / CUDA 12.2):
- torch cu126 wheels run on driver 535 via CUDA minor-version compatibility; the default
  torch 2.13 wheel (CUDA 13) does not.
- CUDA 12.6 nvcc needs gcc <= 13; the host has gcc 14, so `env.sh` points CC/CXX/NVCC_CCBIN
  at a conda-forge gcc 13 in `env/gcc13`.
- FlashInfer's JIT compresses fatbins (`--compress-mode=size`), which driver 535 rejects as
  "device kernel image is invalid"; `env/bin/nvcc-nocompress` strips that flag.
- The host exports `CXXFLAGS=-std=c++17`, which overrides the engine's C++20 build flags;
  `env.sh` unsets it.
- Python is fetched from a mirror (`UV_PYTHON_INSTALL_MIRROR`); the FlashInfer JIT cache
  lives in `env/flashinfer-home`.
- On the A800 box pypi.org is slow or unreachable directly: set `HTTP_PROXY=HTTPS_PROXY=100.68.165.249:3128`
  with `NO_PROXY=mirrors.ivolces.com` (the proxy cannot tunnel to the internal mirror); astral.sh is unreachable, so
  `setup_env.sh` takes uv from its PyPI wheel.

## Running

```bash
scripts/serve.sh 8000 models/Qwen3-0.6B --deterministic          # deterministic engine
scripts/serve.sh 8001 models/Qwen3-0.6B --attention-backend flashinfer   # original engine
python detcheck/detcheck.py --url http://127.0.0.1:8000 --tokenizer models/Qwen3-0.6B
eval/crossconfig.sh --deterministic      # bitwise identity across server configurations
```

## Reproducing every result

`eval/run_all.sh` rebuilds the engine for the local GPU and regenerates every result the
paper uses, stage by stage (logs in `results/run_all/`): kernel tests, kernel benchmarks,
cross-configuration and detcheck runs, end-to-end matrices, our engine's same-GPU baseline
rows, GSM8K and fp32 agreement. It takes about 9 hours on one RTX 4090 D (24 GB) and needs
the whole GPU. The SGLang and vLLM baseline rows are produced separately by
`eval/baselines.sh Qwen3-1.7B sglang sglang-det sglang-det-triton vllm vllm-det`.

## Tests

The kernel tests JIT-build the torch extensions on first import. One engine CUDA test,
`segmented_prefill_matches_eager_logits_and_kv...` (FlashInfer backend), also fails on
the upstream `main` branch and is unrelated to this work.

```bash
cd kernels && python test_invariance.py && python test_matrix.py && python test_directed.py && python test_gemm.py
cd llm_infer_rust && VENV_DIR=$VIRTUAL_ENV scripts/run-rust-tests.sh cuda
SGLANG_E2E_BASE_URL=http://127.0.0.1:8000/v1 SGLANG_E2E_DETERMINISTIC=1 python -m unittest discover -s tests -p "test_*.py"
```
