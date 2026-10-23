#!/usr/bin/env bash
# Probe 19: where 0.6B at 512/256/c128 loses 1.6%. Traces the same 128 greedy requests
# (512 prompt / 256 output tokens, ignore_eos) in each mode and sums GPU time per category
# over the whole run (cat_totals.py); both modes do identical work.
# Usage: archive/ceiling_probes/c128/trace_c128.sh <name> [server args...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../env.sh"
name=$1; shift
out=$ROOT/archive/ceiling_probes/c128/$name
port=18099
export LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so
nsys profile -t cuda --cuda-graph-trace=node -o $out -f true \
  $ROOT/llm_infer_rust/target/release/sglang-rust --model-path $ROOT/models/Qwen3-0.6B --port $port --device cuda \
  --max-running-req 128 --max-seq-len 8192 --max-total-tokens 131072 --chunked-prefill-size 2048 \
  --log-dir $ROOT/logs "$@" > $ROOT/logs/c128_$name.out 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
python - <<PY
import random, requests, concurrent.futures as cf
rng = random.Random(0)
prompts = [[rng.randrange(1000, 50000) for _ in range(512)] for _ in range(128)]
def go(p):
    requests.post("http://127.0.0.1:$port/v1/completions", json={"prompt": p, "max_tokens": 256, "temperature": 0, "ignore_eos": True})
go(prompts[0])  # warm-up
with cf.ThreadPoolExecutor(128) as ex:
    list(ex.map(go, prompts))
PY
kill $pid; wait $pid || true
nsys export -t sqlite -f true -o $out.sqlite $out.nsys-rep > /dev/null
