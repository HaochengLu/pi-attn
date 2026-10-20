#!/usr/bin/env bash
# GSM8K (5-shot, greedy) through lm-eval's local-completions client against the
# engine in both modes, each at client concurrency 1 and 16. Reports accuracy and
# keeps per-sample generations so runs can be compared item by item.
# Usage: eval/accuracy.sh <model-name> [limit]
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
model=$1; limit=${2:-0}
port=18097
out=$ROOT/results/accuracy/$model
mkdir -p $out
for mode in deterministic flashinfer; do
  if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
  $ROOT/scripts/serve.sh $port $ROOT/models/$model $margs --max-running-req 32 --max-seq-len 8192 \
    --max-total-tokens 65536 --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$mode.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  for conc in 1 16; do
    lim=(); [[ $limit != 0 ]] && lim=(--limit $limit)
    lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 "${lim[@]}" \
      --model_args "model=default,base_url=http://127.0.0.1:$port/v1/completions,tokenizer=$ROOT/models/$model,num_concurrent=$conc,max_retries=3,tokenized_requests=False" \
      --gen_kwargs "temperature=0" --log_samples --output_path $out/$mode.c$conc > $out/$mode.c$conc.log 2>&1
    echo "$model $mode c=$conc: $(grep -E "exact_match" $out/$mode.c$conc.log | head -2 | tr -s " " | tr "\n" " ")"
  done
  kill $pid; wait $pid 2>/dev/null; sleep 4
done
