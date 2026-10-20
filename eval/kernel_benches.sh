#!/usr/bin/env bash
# Kernel-level benchmarks with stored outputs (results/kernels/).
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
out=$ROOT/results/kernels
mkdir -p $out
cd $ROOT/kernels
python bench_attention.py --out $out/attention.json > $out/attention.txt 2>&1; echo "attention exit=$?"
python test_gemm.py --bench > $out/gemm.txt 2>&1; echo "gemm exit=$?"
python gemm_cold.py > $out/gemm_cold.txt 2>&1; echo "gemm_cold exit=$?"
python gemm_step_graph.py > $out/gemm_step_graph.txt 2>&1; echo "gemm_step_graph exit=$?"
python bench_prefill.py > $out/prefill_configs.txt 2>&1; echo "prefill exit=$?"
echo KB_DONE
