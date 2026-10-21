#!/usr/bin/env bash
# Where does --deterministic spend its time on a workload? nsys traces (CUDA-graph
# nodes traced) of the engine with FlashInfer and with PI on the same loadgen
# workloads; kernel time by category and the top kernels of each trace.
# Usage: eval/profile_engine.sh <model> <tag> <workload>... (loadgen in:out:conc[:shared])
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
model=${1:-Qwen3-8B}; tag=${2:-shared}; shift 2 || true
work=("$@"); [[ ${#work[@]} -gt 0 ]] || work=(4096:128:8:3968)
out=$ROOT/results/engine_profile/$model.$tag
mkdir -p $out
port=18095
declare -A pool=([Qwen3-0.6B]=131072 [Qwen3-1.7B]=131072 [Qwen3-8B]=32768)
declare -A seqlen=([Qwen3-0.6B]=32768 [Qwen3-1.7B]=32768 [Qwen3-8B]=16384)
for mode in flashinfer pi; do
  if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
  nsys profile -t cuda --cuda-graph-trace=node -o $out/$mode -f true \
    $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len ${seqlen[$model]} \
    --max-total-tokens ${pool[$model]} --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$mode.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num 32 --repeats 1 --workloads "${work[@]}" \
    --out $out/$mode.json > $out/$mode.log 2>&1
  kill -INT $pid; wait $pid 2>/dev/null; bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
  nsys stats -r cuda_gpu_kern_sum -f csv -o $out/$mode $out/$mode.nsys-rep > /dev/null 2>&1
done
{
  echo "# $model $tag: ${work[*]}"; echo
  python $ROOT/eval/kernel_categories.py $out/flashinfer_cuda_gpu_kern_sum.csv $out/pi_cuda_gpu_kern_sum.csv
  for mode in flashinfer pi; do
    echo; echo "## top kernels: $mode"; echo
    python - $out/${mode}_cuda_gpu_kern_sum.csv <<'PY'
import csv, sys
rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: -float(r["Total Time (ns)"]))
for r in rows[:12]:
    print(f'- {float(r["Total Time (ns)"])/1e6:9.1f} ms  {int(r["Instances"]):7d}x  {r["Name"][:110]}')
PY
  done
  for mode in flashinfer pi; do echo; echo "$mode: $(cat $out/$mode.log | tail -n +1 | grep -v '^\s*$' | tail -3)"; done
} > $out/SUMMARY.md
echo PROFILE_ENGINE_DONE
