#!/usr/bin/env bash
# Record of the verification run (it ran from logs/narrow/, the gitignored scratch directory).
# Verification after the narrow-shard GEMM ids (63, 64; CUTLASS stream-K with S > 1).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env.sh
out=logs/narrow
clk() { nvidia-smi --query-gpu=clocks.sm,clocks.applications.graphics --format=csv,noheader; }
(cd kernels && python test_gemm.py > ../$out/v_test_gemm.log 2>&1); echo "test_gemm rc=$?"
(cd kernels && python test_tp_model.py > ../$out/v_test_tp_model.log 2>&1); echo "test_tp_model rc=$?"
(cd kernels && { echo "clock before: $(clk)"; python bench_tp.py 8; echo "clock after: $(clk)"; } > ../$out/v_bench_tp.log 2>&1); echo "bench_tp rc=$?"
(cd llm_infer_rust && CARGO_TARGET_DIR=../build/a800/llm_infer_rust/target FLASHINFER_CUDA_ARCH=80 cargo build --release > ../$out/v_cargo.log 2>&1); echo "cargo rc=$?"
bash logs/tp/engine_regress.sh
