#!/usr/bin/env bash
# Kernel-level benchmarks with stored outputs (results/kernels/).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
out=$ROOT/results/kernels
mkdir -p $out
cd $ROOT/kernels
python bench_attention.py --out $out/attention.json > $out/attention.txt 2>&1; echo "attention exit=$?"
python test_gemm.py --bench > $out/gemm.txt 2>&1; echo "gemm exit=$?"
python gemm_cold.py 1 8 32 64 128 > $out/gemm_cold.txt 2>&1; echo "gemm_cold exit=$?"
python gemm_burst.py 1 8 16 32 64 128 256 512 1024 2048 > $out/gemm_burst.txt 2>&1; echo "gemm_burst exit=$?"
python gemm_step_graph.py > $out/gemm_step_graph.txt 2>&1; echo "gemm_step_graph exit=$?"
python bench_prefill.py > $out/prefill_configs.txt 2>&1; echo "prefill exit=$?"
echo KB_DONE
