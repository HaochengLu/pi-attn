#!/usr/bin/env bash
# Rollout log-probabilities vs a trainer's forward (eval/rollout_logprobs.py), all four models: the
# deterministic engine generates 64 GSM8K rollouts with logprobs, then the server stops and two trainers
# (the engine's kernels in PyTorch, and HF transformers bf16) score the same sequences.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
port=18096
out=$ROOT/results/rollout
mkdir -p $out
for M in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  $ROOT/scripts/serve.sh $port $ROOT/models/$M --deterministic --max-running-req 32 --max-seq-len 4096 \
    --max-total-tokens 65536 --chunked-prefill-size 1024 --log-dir $ROOT/logs > $out/$M.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  python $ROOT/eval/rollout_logprobs.py gen --url http://127.0.0.1:$port --model $ROOT/models/$M --out $out/$M.gen.json
  kill $pid; wait $pid 2>/dev/null; sleep 4
  python $ROOT/eval/rollout_logprobs.py score --model $ROOT/models/$M --gen $out/$M.gen.json --out $out/$M.json \
    > $out/$M.log 2>&1
  echo "$M score exit=$? $(grep 'PI TRAINER' $out/$M.log)"
done
