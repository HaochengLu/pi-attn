#!/usr/bin/env bash
# results/a800_r6: validation and evaluation of the round-6 tree on the A800 (one session).
#   kernel tests and hashes against the A800 reference; engine rollouts against the A800 reference and detcheck, four
#   models; the Rust unit tests and the engine's CUDA test of the PI backend; an sm_90a build of the engine;
#   attention kernel benches; the §5.3 end-to-end stages (matrix, components, extended, high concurrency),
#   FlashInfer and PI servers interleaved as in eval/run_all.sh. Usage: run_r6.sh [stage...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
export PI_RESULTS=$ROOT/results/a800_r6
log=$PI_RESULTS/run; mkdir -p $log
stage() { echo "[$(date -u +%H:%M)] $1" | tee -a $log/run.out; }
want() { [ $# -eq 1 ] || return 0; }
stages=${@:-kernel-tests rust-tests sm90-build engine-bits kernel-benches e2e-matrix e2e-ablation e2e-extended e2e-highconc}
( while true; do echo "$(date -u +%FT%T) $(nvidia-smi --query-gpu=clocks.sm,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader)"; sleep 60; done ) >> $log/clocks.log &
clk=$!
trap "kill $clk 2>/dev/null" EXIT
{ date -u; nvidia-smi --query-gpu=name,driver_version,vbios_version,pstate,clocks.max.sm,clocks.max.memory,clocks.applications.graphics,power.limit --format=csv
  nvcc --version | tail -2; python -c "import torch, flashinfer; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'flashinfer', flashinfer.__version__)"
  echo "commit $(git -C $ROOT rev-parse --short HEAD)"; git -C $ROOT status --short | grep -v '^??' | head; } > $PI_RESULTS/ENV.txt 2>&1
for s in $stages; do
  stage $s
  case $s in
  kernel-tests)
    out=$PI_RESULTS/kernel_tests; mkdir -p $out; cd $ROOT/kernels
    REF=../results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json
    { python ../eval/xgpu_hashes.py > $out/xgpu_hashes.json 2> $out/xgpu_hashes.err; echo "xgpu_hashes exit=$?"
      python ../eval/xgpu_compare.py $out/xgpu_hashes.json $REF > $out/xgpu_compare.txt 2>&1; echo "xgpu_compare exit=$?"; tail -1 $out/xgpu_compare.txt; grep -E "^gemm|gemm:" $out/xgpu_compare.txt | tail -1
      python - $out/xgpu_hashes.json $REF <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print("device", a["device"], "vs reference", b["device"])
for kind in ("gemm", "attention"):
    pi = sum(a[kind][c]["pi"] == b[kind][c]["pi"] for c in b[kind])
    full = sum(a[kind][c] == b[kind][c] for c in b[kind])
    print(f"HASHES {kind}: PI identical {pi}/{len(b[kind])}; PI and vendor both identical {full}/{len(b[kind])}; cases in new {len(a[kind])}")
PY
      for t in test_invariance test_directed test_matrix test_gemm test_mutants; do
        python $t.py > $out/$t.log 2>&1; echo "$t exit=$?"; tail -2 $out/$t.log | cut -c1-300
      done; } > $log/kernel_tests.log 2>&1
    grep -E "exit=|HASHES|UNCAUGHT|checks failed|FAILURES" $log/kernel_tests.log | tee -a $log/run.out ;;
  rust-tests)
    out=$PI_RESULTS/rust_tests; mkdir -p $out; cd $ROOT/llm_infer_rust
    { LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so cargo test --release --locked > $out/unit.log 2>&1; echo "cargo test exit=$? | $(grep -E '^test result' $out/unit.log | tr '\n' ' ')"
      LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so cargo test --release --locked --lib -- --ignored pi_ single_token > $out/pi_cuda.log 2>&1
      echo "cargo test --ignored pi_ exit=$? | $(grep -E '^test (.*pi_|result)' $out/pi_cuda.log | tr '\n' ' ')"; } 2>&1 | tee $log/rust_tests.log | tee -a $log/run.out ;;
  sm90-build)
    # compiles the wgmma kernels of this tree (Hopper); nothing of it runs on this GPU
    out=$PI_RESULTS/sm90_build; mkdir -p $out; cd $ROOT/llm_infer_rust
    { FLASHINFER_CUDA_ARCH=90a CARGO_TARGET_DIR=$ROOT/build/sm90a cargo build --release --locked > $out/cargo_sm90a.log 2>&1; echo "engine sm_90a build exit=$? | $(tail -1 $out/cargo_sm90a.log)"
      (cd $ROOT/kernels && PI_ATTN_ARCH=90 PI_ATTN_SUFFIX=_r6check python -c "from build import build; build(); print('ok')" > $out/ext_sm90a.log 2>&1); echo "kernel extension sm_90a build exit=$?"; } 2>&1 | tee $log/sm90_build.log | tee -a $log/run.out ;;
  engine-bits)
    out=$PI_RESULTS/engine_bits; mkdir -p $out; port=18197
    for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
      $ROOT/scripts/serve.sh $port $ROOT/models/$M --deterministic --max-running-req 32 --max-seq-len 8192 \
        --max-total-tokens 32768 --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$M.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      python $ROOT/eval/xgpu_rollout.py --url http://127.0.0.1:$port --ref $ROOT/results/a800/rollout/$M.gen.json \
        --out $out/$M.rollout.gen.json > $out/$M.rollout.log 2>&1
      echo "$M xgpu_rollout exit=$? | $(grep -E 'rollouts:' $out/$M.rollout.log) | $(grep 'ROLLOUTS IDENTICAL' $out/$M.rollout.log)"
      python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $ROOT/models/$M --logprobs \
        --out $out/$M.detcheck.json > $out/$M.detcheck.log 2>&1
      echo "$M detcheck exit=$? | $(tail -1 $out/$M.detcheck.log)"
      echo "$M decode graphs captured for another launch plan: $(grep -c 'decode graph of another launch plan' $out/$M.server.log)"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done 2>&1 | tee $log/engine_bits.log | tee -a $log/run.out ;;
  kernel-benches)
    out=$PI_RESULTS/kernels; mkdir -p $out; cd $ROOT/kernels
    { python bench_attention.py --out $out/attention.json > $out/attention.txt 2>&1; echo "attention exit=$?"
      python $PI_RESULTS/mixed_batch_bench.py 3 > $out/mixed_batch.txt 2>&1; echo "mixed_batch exit=$?"
      python $PI_RESULTS/wave_split_bench.py 3 > $out/wave_split.txt 2>&1; echo "wave_split exit=$?"
      python $PI_RESULTS/prefill_order_bench.py 3 > $out/prefill_order.txt 2>&1; echo "prefill_order exit=$?"
      python bench_prefill.py > $out/prefill_configs.txt 2>&1; echo "prefill exit=$?"; } 2>&1 | tee $log/kernel_benches.log | tee -a $log/run.out ;;
  e2e-matrix)
    bash $ROOT/eval/e2e_matrix.sh 3 > $log/e2e_matrix.log 2>&1
    python $ROOT/eval/summarize_matrix.py $PI_RESULTS/e2e_matrix > $PI_RESULTS/e2e_matrix/SUMMARY.md
    for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do grep -h "PI GEMM tiles tuned" $PI_RESULTS/e2e_matrix/$M.deterministic.L1.server.log | sed 's/\x1b\[[0-9;]*m//g' | sed "s/^[^ ]* */$M: /" | tail -1; done > $PI_RESULTS/e2e_matrix/tuning.txt 2>/dev/null ;;
  e2e-ablation)
    bash $ROOT/eval/e2e_ablation.sh 3 > $log/e2e_ablation.log 2>&1 ;;
  e2e-extended)
    bash $ROOT/eval/e2e_extended.sh 3 > $log/e2e_extended.log 2>&1
    python $ROOT/eval/summarize_extended.py $PI_RESULTS/e2e_extended > $PI_RESULTS/e2e_extended/SUMMARY.md ;;
  e2e-highconc)
    bash $ROOT/eval/e2e_highconc.sh 3 > $log/e2e_highconc.log 2>&1
    python $ROOT/eval/summarize_extended.py $PI_RESULTS/e2e_highconc > $PI_RESULTS/e2e_highconc/SUMMARY.md ;;
  esac
done
stage done
echo RUN_R6_DONE | tee -a $log/run.out
