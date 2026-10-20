#!/usr/bin/env bash
# Top-1 agreement with HF transformers fp32 (teacher forcing on each engine's greedy
# output), original vs --deterministic, all four models. 8B is scored on the CPU.
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
port=18095
out=$ROOT/results/fp32_agreement
mkdir -p $out
for model in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  for mode in flashinfer deterministic; do
    if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
    $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len 4096 \
      --max-total-tokens 32768 --log-dir $ROOT/logs > $out/$model.$mode.server.log 2>&1 &
    pid=$!
    for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
    python $ROOT/eval/hf_agreement.py gen --url http://127.0.0.1:$port --out $out/$model.$mode.gen.json
    kill $pid; wait $pid 2>/dev/null; sleep 4
  done
  dev=cuda; [[ $model == Qwen3-8B ]] && dev=cpu
  python $ROOT/eval/hf_agreement.py score --model $ROOT/models/$model --device $dev \
    --gens $out/$model.flashinfer.gen.json $out/$model.deterministic.gen.json --out $out/$model.json
done
echo FP32_DONE
