#!/usr/bin/env bash
# High-concurrency decode (64-128 running requests), where projections run at M = 64-128
# and the mid-M GEMM kernels matter. Interleaved launches (FI, PI, FI, PI, ...).
# 4B runs c64 only: its c128 KV does not fit next to the weights on 24 GB.
# Usage: eval/e2e_highconc.sh [launches]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
launches=${1:-3}
port=18096
out=$ROOT/results/e2e_highconc
mkdir -p $out
declare -A pool=( [Qwen3-0.6B]=131072 [Qwen3-1.7B]=131072 [Qwen3-4B]=65536 )
declare -A work=(
  [Qwen3-0.6B]="512:256:64 512:256:128 128:512:128 2048:128:64"
  [Qwen3-1.7B]="512:256:64 512:256:128 128:512:128 2048:128:64"
  [Qwen3-4B]="512:256:64 128:512:64" )
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B; do
  for l in $(seq 1 $launches); do
    for mode in flashinfer deterministic; do
      if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
      $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 128 --max-seq-len 8192 \
        --max-total-tokens ${pool[$model]} --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$model.$mode.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num 256 --repeats 1 --workloads ${work[$model]} \
        --out $out/$model.$mode.L$l.json > $out/$model.$mode.L$l.log 2>&1
      echo "$model $mode launch $l exit=$?"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
echo HIGHCONC_DONE
