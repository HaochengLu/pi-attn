# H100 kernel-level validation of the round-4 tree: bits, decode mutants, the engine-regime table, and
# pi_launch.cu compiled for sm_90a as the engine builds it.
set -u
t0=$(date +%s)
cd /pi && git apply /work/r4.diff && echo "patch applied" && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
CUT=$(python -c "import flashinfer,os;print(os.path.join(os.path.dirname(flashinfer.__file__),'data','cutlass','include'))")
(cd /pi/llm_infer_rust && nvcc -std=c++17 --expt-relaxed-constexpr -O3 -DPI_SM90A=1 -gencode=arch=compute_90a,code=sm_90a \
   -Isrc/kernels/pi -I$CUT -c src/kernels/pi/pi_launch.cu -o /work/pi_launch.o > /work/launch.log 2>&1; \
 echo "pi_launch.cu sm_90a compile exit=$? ($(( $(date +%s) - t0 ))s)"; grep -E "rror" -A5 /work/launch.log | head -30) &
cd kernels
python test_invariance.py > /work/inv.log 2>&1; echo "test_invariance exit=$? ($(( $(date +%s) - t0 ))s)"; tail -2 /work/inv.log
python -c "import test_invariance as T; print('DECODE_VARIANTS', T.DECODE_VARIANTS)" 2>&1 | tail -1
python /work/dec_r4_bench.py eq 2>&1 | grep -E "EQ|MISMATCH|rror|Trace"
python /work/attn_hashes.py 2>&1 | grep -E "HASH|hashes|rror"
echo "bits done ($(( $(date +%s) - t0 ))s)"
python /work/dec_r4_bench.py table 2>&1 | grep -E "TABLE|rror|Trace"
echo "table done ($(( $(date +%s) - t0 ))s)"
python /work/decode_mutants.py 2>&1 | grep -E "mutant|MUTANTS|rror|Trace"
wait
echo "done ($(( $(date +%s) - t0 ))s)"
