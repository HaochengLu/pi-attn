#!/bin/bash
set -u
t0=$(date +%s)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /pi && git apply /work/gemm_final.diff && echo "patch applied" || exit 1
cd /pi/kernels
python -c "import test_gemm" > /work/build.log 2>&1; echo "gemm ext build exit=$? ($(( $(date +%s) - t0 ))s)"; tail -5 /work/build.log | cut -c1-600
timeout 600 python /work/cold_table.py hash 2>&1 | grep -E "^(TABLE|HASHX)|Error|error" | cut -c1-900; echo "cold_table exit=${PIPESTATUS[0]} ($(( $(date +%s) - t0 ))s)"
python ../eval/xgpu_hashes.py > /work/hashes_new.json 2> /work/hashes.err; echo "xgpu_hashes exit=$? ($(( $(date +%s) - t0 ))s)"
python - <<'PY'
import json
a = json.load(open("/work/hashes_new.json")); b = json.load(open("/pi/results/modal/H100_sm90/hashes_H100.json"))
for kind in ("gemm", "attention"):
    same = sum(a[kind][c] == b[kind][c] for c in b[kind])
    print(f"HASHES {kind}: {same}/{len(b[kind])} cases identical to results/modal/H100_sm90/hashes_H100.json (pi and vendor hashes)")
print("HASHES gemm pi-only identical:", sum(a["gemm"][c]["pi"] == b["gemm"][c]["pi"] for c in b["gemm"]), "/", len(b["gemm"]))
PY
timeout 900 python test_gemm.py > /work/test_gemm.log 2>&1; echo "test_gemm exit=$? ($(( $(date +%s) - t0 ))s)"
grep -c "" /work/test_gemm.log; grep -E "GEMM candidate|INVARIANCE|Error|error" /work/test_gemm.log | cut -c1-600; tail -3 /work/test_gemm.log | cut -c1-300
