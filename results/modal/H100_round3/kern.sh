#!/bin/bash
# Round 3 integrated kernel validation on the merged tree (61ff596 + merged.diff). $1 = H100 | classA.
# H100: suites, mutants, hashes vs results/modal/H100_sm90/hashes_H100.json, every wgmma id's hashes, benches.
# classA (A10G): test_invariance, test_directed, xgpu_hashes vs the A800 reference.
set -u
t0=$(date +%s); el() { echo "$(( $(date +%s) - t0 ))s"; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /pi && git apply /work/merged.diff && echo "patch applied (HEAD $(git rev-parse --short HEAD) + merged.diff)" || exit 1
cd /pi/kernels
python -c "import torch, flashinfer; print('torch', torch.__version__, 'flashinfer', flashinfer.__version__, 'cc', torch.cuda.get_device_capability())"
python -c "import test_invariance as T; print('attn ext WGMMA', T.ext.WGMMA, 'VARIANTS', T.VARIANTS, 'DECODE_VARIANTS', T.DECODE_VARIANTS)" 2>&1 | tail -3; echo "attn ext build exit=${PIPESTATUS[0]} ($(el))"
python test_invariance.py > /work/inv.log 2>&1; echo "test_invariance exit=$? ($(el))"; tail -6 /work/inv.log | cut -c1-300
python test_directed.py > /work/dir.log 2>&1; echo "test_directed exit=$? ($(el))"; tail -4 /work/dir.log | cut -c1-300
python ../eval/xgpu_hashes.py > /work/hashes_new.json 2> /work/hashes.err; echo "xgpu_hashes exit=$? ($(el))"
REF=/pi/results/modal/H100_sm90/hashes_H100.json; [ "$1" = classA ] && REF=/pi/results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json
python - $REF <<'PY'
import json, sys
a = json.load(open("/work/hashes_new.json")); b = json.load(open(sys.argv[1]))
print("device", a["device"], "vs reference", b["device"], sys.argv[1])
for kind in ("gemm", "attention"):
    pi = sum(a[kind][c]["pi"] == b[kind][c]["pi"] for c in b[kind])
    full = sum(a[kind][c] == b[kind][c] for c in b[kind])
    print(f"HASHES {kind}: PI identical {pi}/{len(b[kind])}; PI and vendor both identical {full}/{len(b[kind])}; cases in new {len(a[kind])}")
    for c in b[kind]:
        if a[kind][c]["pi"] != b[kind][c]["pi"]: print("  DIFFERENT", kind, c, a[kind][c]["pi"], "ref", b[kind][c]["pi"])
PY
echo "=== HASHES_JSON_BEGIN"; python -c "import json;print(json.dumps(json.load(open('/work/hashes_new.json'))))"; echo "=== HASHES_JSON_END"
if [ "$1" = classA ]; then echo "total $(el)"; exit 0; fi
python test_matrix.py > /work/mat.log 2>&1; echo "test_matrix exit=$? ($(el))"; tail -2 /work/mat.log | cut -c1-300
timeout 1200 python test_gemm.py > /work/test_gemm.log 2>&1; echo "test_gemm exit=$? ($(el))"
grep -E "GEMM candidate|INVARIANCE|Error|error" /work/test_gemm.log | cut -c1-600; tail -2 /work/test_gemm.log | cut -c1-300
python test_mutants.py > /work/mut.log 2>&1; echo "test_mutants exit=$? ($(el))"; grep -E "mutant|UNCAUGHT|control|rror" /work/mut.log | cut -c1-300
timeout 900 python /work/cold_table.py hash 2>&1 | grep -E "^(TABLE|HASHX)|Error|error" | cut -c1-1200; echo "cold_table exit=${PIPESTATUS[0]} ($(el))"
python /work/dec90_bench.py eq 2>&1 | grep -E "EQ|MISMATCH|rror" | cut -c1-300
PI_DEC_VARIANTS=1,2,0 python /work/dec90_bench.py full 2>&1 | grep -E "DEC|rror"; echo "dec bench done ($(el))"
PI_PRE_VARIANTS=5,6,7,0 python -u /work/pre_bench.py 2>&1 | grep -E "PRE|rror"; echo "pre bench done ($(el))"
PI_PRE_ENGINE=1 PI_PRE_VARIANTS=5,6,7,0 python -u /work/pre_bench.py 2>&1 | grep -E "PRE|rror" | sed 's/^/ENGSHAPE /'; echo "pre bench (engine shapes) done ($(el))"
echo "total $(el)"
