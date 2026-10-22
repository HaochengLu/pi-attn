#!/usr/bin/env bash
# Trace one batch-1 generation (64 prompt / 256 output tokens) under nsys with
# CUDA-graph node tracing; prints the per-kernel time summary.
# Usage: scripts/nsys_decode.sh <name> [server args...]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
name=$1; shift
out=$ROOT/results/nsys/$name
mkdir -p $ROOT/results/nsys
port=18099
export LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so
nsys profile -t cuda --cuda-graph-trace=graph -o $out -f true \
  $ROOT/llm_infer_rust/target/release/sglang-rust --model-path $ROOT/models/Qwen3-0.6B --port $port --device cuda \
  --max-running-req 8 --max-seq-len 4096 --max-total-tokens 65536 --log-dir $ROOT/logs "$@" > $ROOT/logs/nsys_$name.out 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
python - <<PY
import requests
for _ in range(2):
    requests.post("http://127.0.0.1:$port/v1/completions", json={"prompt": list(range(1000, 1064)), "max_tokens": 256, "temperature": 0, "ignore_eos": True})
PY
kill $pid; wait $pid || true
nsys stats -r cuda_gpu_kern_sum -f csv $out.nsys-rep 2>/dev/null | head -40
