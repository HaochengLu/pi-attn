#!/usr/bin/env bash
# Quick interleaved FI vs PI comparison on a few loadgen workloads.
# Usage: eval/quick_e2e.sh <model> <launches> <workload>...
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
model=$1; launches=$2; shift 2
out=$ROOT/results/quick/$model.$(date +%H%M%S); mkdir -p $out
port=18099
declare -A pool=([Qwen3-0.6B]=131072 [Qwen3-1.7B]=131072 [Qwen3-4B]=65536 [Qwen3-8B]=32768)
declare -A seqlen=([Qwen3-0.6B]=32768 [Qwen3-1.7B]=32768 [Qwen3-4B]=16384 [Qwen3-8B]=16384)
for l in $(seq 1 $launches); do
  for mode in flashinfer pi; do
    if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
    $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len ${seqlen[$model]} \
      --max-total-tokens ${pool[$model]} --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$mode.L$l.server.log 2>&1 &
    for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
    python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --num 32 --repeats 1 --workloads "$@" \
      --out $out/$mode.L$l.json > $out/$mode.L$l.log 2>&1
    bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
  done
done
python - $out <<'PY'
import glob, json, statistics, sys
d = sys.argv[1]
res = {}
for f in sorted(glob.glob(f"{d}/*.json")):
    mode = f.rsplit("/", 1)[1].split(".")[0]
    for r in json.load(open(f)):  # loadgen rows
        key = f"{r['in']}/{r['out']}/c{r['concurrency']}" + (f"/{r['shared']}" if r.get("shared") else "")
        res.setdefault(key, {}).setdefault(mode, []).append((r["out_tok_s"], r["ttft_ms"]))
for key, m in res.items():
    fi, pi = statistics.median(x[0] for x in m["flashinfer"]), statistics.median(x[0] for x in m["pi"])
    ft, pt = statistics.median(x[1] for x in m["flashinfer"]), statistics.median(x[1] for x in m["pi"])
    print(f"{key}: out tok/s FI {fi:.1f} PI {pi:.1f} ({(pi / fi - 1) * 100:+.1f}%) | TTFT FI {ft:.1f} PI {pt:.1f} ({(pt / ft - 1) * 100:+.1f}%)")
PY
