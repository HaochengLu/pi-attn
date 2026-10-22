#!/usr/bin/env bash
# Extended serving matrix: prefill-heavy, shared-prefix (prefix-cache extend) and
# long-context workloads, plus two of the base workloads, measured over interleaved
# server launches (FI, PI, FI, PI, ...) so launch-to-launch variance is visible.
# Usage: eval/e2e_extended.sh [launches]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
launches=${1:-3}
port=18096
out=$ROOT/results/e2e_extended
mkdir -p $out
declare -A pool=( [Qwen3-1.7B]=131072 [Qwen3-8B]=32768 )
declare -A seqlen=( [Qwen3-1.7B]=32768 [Qwen3-8B]=16384 )
declare -A work=(
  [Qwen3-1.7B]="64:256:1 512:256:32 2048:16:8 4096:4:4 4096:128:8:3968 8192:64:4:8000 16384:256:2 30000:128:1"
  [Qwen3-8B]="64:256:1 512:256:32 2048:16:8 4096:4:4 4096:128:8:3968 8192:64:4:8000 16384:256:1" )
for model in Qwen3-1.7B Qwen3-8B; do
  for l in $(seq 1 $launches); do
    for mode in flashinfer deterministic; do
      if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
      $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len ${seqlen[$model]} \
        --max-total-tokens ${pool[$model]} --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$model.$mode.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num 32 --repeats 1 --workloads ${work[$model]} \
        --out $out/$model.$mode.L$l.json > $out/$model.$mode.L$l.log 2>&1
      echo "$model $mode launch $l exit=$?"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
echo EXT_DONE
