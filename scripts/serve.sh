#!/usr/bin/env bash
# Start the llm_infer_rust server from this project's environment.
# Usage: scripts/serve.sh <port> <model-dir> [server args...]
set -euo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
port=$1; model=$2; shift 2
bin=$ROOT/llm_infer_rust/target/release/sglang-rust
export LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so
exec "$bin" --model-path "$model" --port "$port" --device cuda "$@"
