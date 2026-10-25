#!/usr/bin/env bash
# Trace a greedy generation (CONC concurrent requests of PROMPT prompt / 256 output tokens,
# default 1 x 64) under nsys with CUDA-graph node tracing; prints GPU time per decode step
# by kernel (scripts/nsys_step.py).
# Usage: [MODEL=Qwen3-1.7B] [CONC=8] [PROMPT=512] scripts/nsys_decode.sh <name> [server args...]  (default Qwen3-0.6B)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
name=$1; shift
out=$ROOT/results/nsys/$name
mkdir -p $ROOT/results/nsys
port=18099
export LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so
nsys profile -t cuda --cuda-graph-trace=node --cuda-flush-interval=100 -o $out -f true \
  $ROOT/llm_infer_rust/target/release/sglang-rust --model-path $ROOT/models/${MODEL:-Qwen3-0.6B} --port $port --device cuda \
  --max-running-req 8 --max-seq-len 4096 --max-total-tokens 65536 --log-dir $ROOT/logs "$@" > $ROOT/logs/nsys_$name.out 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
python - <<PY
import concurrent.futures, requests
def gen(i):
    prompt = [1000 + (i * 7919 + j) % 30000 for j in range(${PROMPT:-64})]
    requests.post("http://127.0.0.1:$port/v1/completions", json={"prompt": prompt, "max_tokens": 256, "temperature": 0, "ignore_eos": True})
with concurrent.futures.ThreadPoolExecutor(${CONC:-1}) as ex:
    for _ in range(2):
        list(ex.map(gen, range(${CONC:-1})))
PY
kill $pid; wait $pid || true
nsys export -t sqlite -f true -o $out.sqlite $out.nsys-rep > /dev/null
python $ROOT/scripts/nsys_step.py $out.sqlite
