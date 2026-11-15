#!/usr/bin/env bash
# A800 round 6 engine A/B: modes interleaved over launches, one fresh server per mode and launch.
# Usage: r6_abl.sh <out-dir> <launches> <config>...   config = "name model maxreq seqlen pool num workloads..."
# modes (MODES): flashinfer, pi_attn, pi_gemm, deterministic; REPO: the engine tree (default: this clone);
# <mode>@<tree> runs the binary of another tree of $R6 (e.g. deterministic@base).
set -uo pipefail
R6=${R6:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source $R6/repo/env.sh
out=$1; launches=$2; shift 2
port=18196
mkdir -p $out
declare -A margs=( [flashinfer]="--attention-backend flashinfer" [pi_attn]="--attention-backend pi"
                   [pi_gemm]="--row-invariant-gemm --attention-backend flashinfer" [deterministic]="--deterministic" )
MODES=${MODES:-"flashinfer pi_attn pi_gemm deterministic"}
for c in "$@"; do
  read -r name model maxreq seqlen pool num work <<< "$c"
  for l in $(seq 1 $launches); do
    for m in $MODES; do
      mode=${m%@*}; tree=repo; [[ $m == *@* ]] && tree=${m#*@}
      [ -s $out/$name.$m.L$l.json ] && continue
      LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so $R6/$tree/llm_infer_rust/target/release/sglang-rust --model-path $ROOT/models/$model --port $port --device cuda \
        ${margs[$mode]} --max-running-req $maxreq --max-seq-len $seqlen --max-total-tokens $pool --chunked-prefill-size 2048 \
        --log-dir $out/srvlogs > $out/$name.$m.L$l.server.log 2>&1 &
      pid=$!
      for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
      ( while kill -0 $pid 2>/dev/null; do nvidia-smi --query-gpu=clocks.sm,utilization.gpu,power.draw --format=csv,noheader; sleep 5; done ) > $out/$name.$m.L$l.clock &
      python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num $num --repeats ${REPEATS:-1} --workloads $work \
        --out $out/$name.$m.L$l.json > $out/$name.$m.L$l.log 2>&1
      echo "$(date +%T) $name $m launch $l exit=$?"
      kill $pid; wait $pid 2>/dev/null; sleep 4
    done
  done
done
echo ABL_DONE
