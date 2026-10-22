#!/usr/bin/env bash
# End-to-end serving matrix: models x {flashinfer, deterministic} x workloads x {greedy, sampled},
# over interleaved server launches (FI, PI, FI, PI, ...), one measurement per launch, so
# launch-to-launch variance hits both modes alike. Usage: eval/e2e_matrix.sh [launches]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
launches=${1:-3}
port=18096
out=$ROOT/results/e2e_matrix
mkdir -p $out
declare -A pool=( [Qwen3-0.6B]=131072 [Qwen3-1.7B]=131072 [Qwen3-4B]=65536 [Qwen3-8B]=32768 )
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  for l in $(seq 1 $launches); do
    for mode in flashinfer deterministic; do
      if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
      $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len 8192 \
        --max-total-tokens ${pool[$model]} --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$model.$mode.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      for temp in 0 0.8; do
        python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --temperature $temp --num 32 --repeats 1 \
          --workloads 64:256:1 512:256:8 512:256:32 2048:256:16 4096:128:4 --out $out/$model.$mode.t$temp.L$l.json \
          > $out/$model.$mode.t$temp.L$l.log 2>&1
        echo "$model $mode t=$temp launch $l done"
      done
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
