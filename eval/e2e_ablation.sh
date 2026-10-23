#!/usr/bin/env bash
# Component ablation: which half of --deterministic causes each remaining difference.
# Four modes over interleaved launches: original (FlashInfer attention + cuBLAS),
# PI attention only (--attention-backend pi), row-invariant GEMM only
# (--row-invariant-gemm with FlashInfer attention), and --deterministic (both).
# Greedy, so the sampler is argmax in every mode. Usage: eval/e2e_ablation.sh [launches]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
launches=${1:-3}
port=18096
out=$ROOT/results/e2e_ablation
mkdir -p $out
declare -A margs=( [flashinfer]="--attention-backend flashinfer" [pi_attn]="--attention-backend pi"
                   [pi_gemm]="--row-invariant-gemm --attention-backend flashinfer" [deterministic]="--deterministic" )
# name: model, max running requests, KV pool tokens, requests, workloads
configs=(
  "0.6B Qwen3-0.6B 32 131072 32 64:256:1 512:256:32"
  "0.6B-c128 Qwen3-0.6B 128 131072 256 512:256:128"
  "1.7B Qwen3-1.7B 32 131072 32 64:256:1 512:256:32"
  "8B Qwen3-8B 32 32768 32 64:256:1 512:256:32" )
for c in "${configs[@]}"; do
  read -r name model maxreq pool num work <<< "$c"
  for l in $(seq 1 $launches); do
    for mode in flashinfer pi_attn pi_gemm deterministic; do
      $ROOT/scripts/serve.sh $port $ROOT/models/$model ${margs[$mode]} --max-running-req $maxreq --max-seq-len 8192 \
        --max-total-tokens $pool --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$name.$mode.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num $num --repeats 1 --workloads $work \
        --out $out/$name.$mode.L$l.json > $out/$name.$mode.L$l.log 2>&1
      echo "$name $mode launch $l exit=$?"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
python $ROOT/eval/summarize_ablation.py $out > $out/SUMMARY.md
echo ABLATION_DONE
