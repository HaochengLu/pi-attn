#!/bin/bash
# Kernel-level validation of the round-4 tree on H100: every GEMM id against the reference hashes, the cold table, test_gemm.
set -u
t0=$(date +%s)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /pi && git apply /work/k.diff && echo "patch applied" || exit 1
cd /pi/kernels
python -c "import test_gemm" > /work/build.log 2>&1; echo "gemm ext build + import exit=$? ($(( $(date +%s) - t0 ))s)"; grep -E "error|C7511" /work/build.log | cut -c1-600 | head; tail -2 /work/build.log | cut -c1-300
timeout -s KILL 420 python /work/cold_table_r4.py hash 128 192 256 512 1024 2048 2304 4096 2>&1 | grep -E "^(TABLE|HASHX)|Error|error" | cut -c1-1100; echo "cold_table exit=${PIPESTATUS[0]} ($(( $(date +%s) - t0 ))s)"
timeout -s KILL 600 python test_gemm.py > /work/test_gemm.log 2>&1; echo "test_gemm exit=$? ($(( $(date +%s) - t0 ))s)"
grep -c "" /work/test_gemm.log; grep -E "GEMM candidate|INVARIANCE|Error|error" /work/test_gemm.log | cut -c1-600; tail -3 /work/test_gemm.log | cut -c1-300
