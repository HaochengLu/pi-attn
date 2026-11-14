set -u
t0=$(date +%s)
cd /pi && git apply /work/final.diff && echo "patch applied" && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd kernels
python test_invariance.py > /work/inv.log 2>&1; echo "test_invariance exit=$? ($(( $(date +%s) - t0 ))s)"; tail -4 /work/inv.log
python ../eval/xgpu_hashes.py > /work/hashes.json 2> /work/hashes.err; echo "xgpu_hashes exit=$? ($(( $(date +%s) - t0 ))s)"
python - <<'PY'
import json
new = json.load(open("/work/hashes.json")); ref = json.load(open("/pi/results/modal/H100_sm90/hashes_H100.json"))
for sec in ("attention", "gemm"):
    same = sum(new[sec][k]["pi"] == ref[sec][k]["pi"] for k in ref[sec])
    print(f"HASHES {sec}: {same}/{len(ref[sec])} PI entries equal to results/modal/H100_sm90/hashes_H100.json")
for k, v in new["attention"].items():
    print("  ", k, v["pi"], "ref", ref["attention"][k]["pi"], "EQUAL" if v["pi"] == ref["attention"][k]["pi"] else "DIFFERENT")
PY
python /work/dec90_bench.py eq 2>&1 | grep -E "EQ|MISMATCH|rror"
PI_DEC_VARIANTS=1,0 python /work/dec90_bench.py full 2>&1 | grep -E "DEC|rror" | sed 's/^/PLANNER /'
PI_DEC_VARIANTS=1,2 python /work/dec90_bench.py grid2 2>&1 | grep -E "DEC|rror" | sed 's/^/GRID2 /'
PI_DEC_SLACK=13 PI_DEC_VARIANTS=1,0 python /work/dec90_bench.py quick 2>&1 | grep -E "DEC|rror" | sed 's/^/SLACK13 /'
echo "benches done ($(( $(date +%s) - t0 ))s)"
PI_MUTANTS=decode90_d_unfused,decode_fold_unfused python test_mutants.py > /work/mut.log 2>&1; echo "test_mutants exit=$? ($(( $(date +%s) - t0 ))s)"; grep -E "mutant|UNCAUGHT|control" /work/mut.log
