set -u
t0=$(date +%s)
cd /pi && git apply /work/r4.diff && echo "patch applied" && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd kernels
python -c "import build; build.build(verbose=True)" > /work/ext.log 2>&1; echo "ext build exit=$? ($(( $(date +%s) - t0 ))s)"
grep -E "rror:|spill|stack" -A4 /work/ext.log | head -40
PI_EQ_VARIANTS=2,3,4,0 timeout 300 python /work/dec_r4_bench.py eq 2>&1 | grep -E "EQ|MISMATCH|rror|Trace" | head -40
echo "eq done ($(( $(date +%s) - t0 ))s)"
PI_HASH_DECODE=1,2,3,4 timeout 200 python /work/attn_hashes.py 2>&1 | grep -E "HASH|hashes|rror" | head
echo "hash done ($(( $(date +%s) - t0 ))s)"
PI_DEC_VARIANTS=1,2,3,4 timeout 600 python /work/dec_r4_bench.py grid 2>&1 | grep -E "DEC|rror|Trace"
echo "done ($(( $(date +%s) - t0 ))s)"
