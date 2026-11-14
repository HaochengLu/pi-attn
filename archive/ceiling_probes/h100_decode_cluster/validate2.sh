set -u
t0=$(date +%s)
cd /pi && git apply /work/final3.diff && echo "patch applied" && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd kernels
python test_invariance.py > /work/inv.log 2>&1; echo "test_invariance exit=$? ($(( $(date +%s) - t0 ))s)"; tail -3 /work/inv.log
python /work/dec90_bench.py eq 2>&1 | grep -E "EQ|MISMATCH|rror"
PI_DEC_VARIANTS=1,0 python /work/dec90_bench.py full 2>&1 | grep -E "DEC|rror" | sed 's/^/PLANNER /'
PI_DEC_SLACK=13 PI_DEC_VARIANTS=1,0 python /work/dec90_bench.py quick 2>&1 | grep -E "DEC|rror" | sed 's/^/SLACK13 /'
echo "done ($(( $(date +%s) - t0 ))s)"
