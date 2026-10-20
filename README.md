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
| `kernels/` | PyTorch bindings, invariance tests (`test_invariance.py`, `test_matrix.py`, `test_directed.py`, `test_gemm.py`) and microbenchmarks for the kernels (sources symlinked from the engine) |
| `ref/` | Bit-level CPU reference model of the attention construction |
| `detcheck/detcheck.py` | Black-box determinism checker for any OpenAI-compatible server |
| `eval/` | Serving benchmark (`loadgen.py`), cross-configuration check (`crossconfig.sh`), end-to-end matrix (`e2e_matrix.sh`), fp32 agreement (`hf_agreement.py`) |
| `results/` | Phase reports and raw results |
| `paper/` | Paper draft |

## Environment (self-contained, reproducible)

Everything is installed under `env/` by `setup_env.sh` (idempotent); nothing touches the
host system. Versions are recorded in `VERSIONS.txt`.

```bash
bash setup_env.sh            # CUDA 12.6 toolkit, gcc 13, Rust 1.95, Python 3.12, torch 2.13.0+cu126, FlashInfer 0.6.18
source env.sh                # activate
cd llm_infer_rust && FLASHINFER_CUDA_ARCH=89 cargo build --release   # 80 for A100/A800
```

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

## Running

```bash
scripts/serve.sh 8000 models/Qwen3-0.6B --deterministic          # deterministic engine
scripts/serve.sh 8001 models/Qwen3-0.6B --attention-backend flashinfer   # original engine
python detcheck/detcheck.py --url http://127.0.0.1:8000 --tokenizer models/Qwen3-0.6B
eval/crossconfig.sh --deterministic      # bitwise identity across server configurations
```

## Tests

```bash
cd kernels && python test_matrix.py && python test_directed.py && python test_gemm.py
cd llm_infer_rust && VENV_DIR=$VIRTUAL_ENV scripts/run-rust-tests.sh cuda
SGLANG_E2E_BASE_URL=http://127.0.0.1:8000/v1 SGLANG_E2E_DETERMINISTIC=1 python -m unittest discover -s tests -p "test_*.py"
```
