#!/usr/bin/env bash
# Regenerate every result on the current kernels, in order: build, kernel tests,
# kernel-level output hashes for cross-GPU comparison (eval/xgpu_compare.py),
# kernel benchmarks, cross-configuration and detcheck runs, end-to-end matrices,
# high-concurrency decode, the attention/GEMM ablation, same-GPU baselines (our engine only
# unless BASELINE_ENGINES lists more, e.g. on a new GPU "rust rust-det sglang sglang-det vllm vllm-det"), GSM8K,
# fp32 agreement, rollout log-probabilities vs a trainer's forward and node-level traces of batch-1 decode. Each stage logs to results/run_all/<stage>.log.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
log=$ROOT/results/run_all; mkdir -p $log
stage() { echo "[$(date +%H:%M)] $1"; }
stage build
arch=${FLASHINFER_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}  # 89 = 4090, 80 = A800
(cd $ROOT/llm_infer_rust && FLASHINFER_CUDA_ARCH=$arch cargo build --release) > $log/build.log 2>&1 || { echo build failed; exit 1; }
stage kernel-tests
(cd $ROOT/kernels && for t in test_invariance test_directed test_matrix test_gemm test_mutants; do python $t.py; echo "$t exit=$?"; done) > $log/kernel_tests.log 2>&1
grep "exit=" $log/kernel_tests.log
stage xgpu-hashes
mkdir -p $ROOT/results/xgpu
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' _)
(cd $ROOT/kernels && python $ROOT/eval/xgpu_hashes.py > $ROOT/results/xgpu/$gpu.json) 2> $log/xgpu.log
stage kernel-benches
bash $ROOT/eval/kernel_benches.sh > $log/kernel_benches.log 2>&1
stage crossconfig
bash $ROOT/eval/crossconfig.sh --deterministic > $log/crossconfig_det.log 2>&1
bash $ROOT/eval/crossconfig.sh --attention-backend flashinfer > $log/crossconfig_fi.log 2>&1
stage detcheck
port=18097; mkdir -p $ROOT/results/detcheck
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  $ROOT/scripts/serve.sh $port $ROOT/models/$M --deterministic --max-running-req 32 --max-seq-len 8192 \
    --max-total-tokens 32768 --chunked-prefill-size 2048 --log-dir $ROOT/logs > $ROOT/results/detcheck/final_$M.server.log 2>&1 &
  for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $ROOT/models/$M --logprobs \
    --out $ROOT/results/detcheck/final_$M.json > $ROOT/results/detcheck/final_$M.log 2>&1
  echo "$M detcheck exit=$? $(tail -1 $ROOT/results/detcheck/final_$M.log)"
  bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
done > $log/detcheck.log 2>&1
stage e2e-matrix
bash $ROOT/eval/e2e_matrix.sh > $log/e2e_matrix.log 2>&1
python $ROOT/eval/summarize_matrix.py $ROOT/results/e2e_matrix > $ROOT/results/e2e_matrix/SUMMARY.md 2>&1
# Startup tuning time and the tile chosen per M bucket (server logs are not committed).
for f in $ROOT/results/e2e_matrix/*.deterministic.L1.server.log; do
  echo "== $(basename $f .deterministic.L1.server.log)"; grep -h 'PI GEMM tile' $f | sed 's/\x1b\[[0-9;]*m//g; s/^.*INFO | //'
done > $ROOT/results/e2e_matrix/tuning.txt
stage e2e-extended
bash $ROOT/eval/e2e_extended.sh 3 > $log/e2e_extended.log 2>&1
python $ROOT/eval/summarize_extended.py $ROOT/results/e2e_extended > $ROOT/results/e2e_extended/SUMMARY.md 2>&1
stage e2e-highconc
bash $ROOT/eval/e2e_highconc.sh 3 > $log/e2e_highconc.log 2>&1
python $ROOT/eval/summarize_extended.py $ROOT/results/e2e_highconc > $ROOT/results/e2e_highconc/SUMMARY.md 2>&1
stage e2e-ablation
bash $ROOT/eval/e2e_ablation.sh > $log/e2e_ablation.log 2>&1
stage baselines
bash $ROOT/eval/baselines.sh Qwen3-1.7B ${BASELINE_ENGINES:-rust rust-det} > $log/baselines.log 2>&1
python $ROOT/eval/summarize_baselines.py $ROOT/results/baselines/Qwen3-1.7B > $ROOT/results/baselines/Qwen3-1.7B/SUMMARY.md 2>&1
stage accuracy
bash $ROOT/eval/accuracy.sh Qwen3-1.7B > $log/accuracy_1.7B.log 2>&1
bash $ROOT/eval/accuracy.sh Qwen3-8B 500 > $log/accuracy_8B.log 2>&1
python $ROOT/eval/compare_accuracy.py $ROOT/results/accuracy/Qwen3-1.7B > $ROOT/results/accuracy/Qwen3-1.7B/SUMMARY.md 2>&1
python $ROOT/eval/compare_accuracy.py $ROOT/results/accuracy/Qwen3-8B > $ROOT/results/accuracy/Qwen3-8B/SUMMARY.md 2>&1
stage fp32
bash $ROOT/eval/fp32_agreement.sh > $log/fp32.log 2>&1
stage rollout
bash $ROOT/eval/rollout.sh > $log/rollout.log 2>&1
stage nsys
for M in 0.6B 1.7B 4B 8B; do
  MODEL=Qwen3-$M bash $ROOT/scripts/nsys_decode.sh b1_${M}_fi --attention-backend flashinfer --chunked-prefill-size 2048
  MODEL=Qwen3-$M bash $ROOT/scripts/nsys_decode.sh b1_${M}_pi --deterministic --chunked-prefill-size 2048
done > $log/nsys.log 2>&1
for M in 0.6B 1.7B 4B 8B; do
  echo "== Qwen3-$M: FlashInfer mode -> deterministic"
  python $ROOT/scripts/nsys_compare.py $ROOT/results/nsys/b1_${M}_fi.sqlite $ROOT/results/nsys/b1_${M}_pi.sqlite
done > $ROOT/results/nsys/b1_compare.txt 2>&1
stage done
echo RUN_ALL_DONE
