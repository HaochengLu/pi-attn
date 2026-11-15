#!/usr/bin/env bash
# A800 component ablation (round 5): flashinfer / pi_attn / pi_gemm / deterministic, interleaved launches.
# Usage: abl.sh <out-dir> <launches> <config>...   config = "name model maxreq pool num workloads..."
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$HERE/repo}
source $REPO/env.sh
out=$1; launches=$2; shift 2
port=18196
mkdir -p $out
declare -A margs=( [flashinfer]="--attention-backend flashinfer" [pi_attn]="--attention-backend pi"
                   [pi_gemm]="--row-invariant-gemm --attention-backend flashinfer" [deterministic]="--deterministic"
                   [det_old]="--deterministic" [det_decode]="--deterministic" [det_prefill]="--deterministic" )
# round-5 scheduling A/B (same bits): det_old = decode_kernel only and ascending prefill grids (the round-4 tree's
# schedule), det_decode = only the decode plan new, det_prefill = only the prefill grid order new
declare -A menv=( [det_old]="PI_DECODE90=0 PI_PREFILL_ORDER=1" [det_decode]="PI_PREFILL_ORDER=1" [det_prefill]="PI_DECODE90=0" )
MODES=${MODES:-"flashinfer pi_attn pi_gemm deterministic"}
for c in "$@"; do
  read -r name model maxreq pool num work <<< "$c"
  for l in $(seq 1 $launches); do
    for mode in $MODES; do
      [ -s $out/$name.$mode.L$l.json ] && continue
      env ${menv[$mode]:-} $ROOT/scripts/serve.sh $port $ROOT/models/$model ${margs[$mode]} --max-running-req $maxreq --max-seq-len 8192 \
        --max-total-tokens $pool --chunked-prefill-size 2048 --log-dir $out/srvlogs > $out/$name.$mode.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      ( while kill -0 $pid 2>/dev/null; do nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader; sleep 5; done ) > $out/$name.$mode.L$l.clock &
      python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num $num --repeats ${REPEATS:-1} --workloads $work \
        --out $out/$name.$mode.L$l.json > $out/$name.$mode.L$l.log 2>&1
      echo "$(date +%T) $name $mode launch $l exit=$?"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
echo ABL_DONE
