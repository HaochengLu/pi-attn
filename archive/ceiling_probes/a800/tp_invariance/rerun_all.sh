#!/usr/bin/env bash
# Record of the verification run (it ran from logs/tp/, the gitignored scratch directory).
# Final verification of the TP phase after the review fixes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source env.sh
out=logs/tp
(cd kernels && python test_gemm.py > ../$out/test_gemm4.log 2>&1); echo "test_gemm rc=$?"
(cd kernels && python test_tp_model.py > ../$out/test_tp_model2.log 2>&1); echo "test_tp_model rc=$?"
(cd llm_infer_rust && CARGO_TARGET_DIR=../build/a800/llm_infer_rust/target FLASHINFER_CUDA_ARCH=80 cargo build --release > ../$out/cargo3.log 2>&1); echo "cargo rc=$?"
bash $out/engine_regress.sh
