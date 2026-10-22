#!/usr/bin/env bash
# Cross-restart / cross-configuration determinism: save references under a base
# server configuration, then restart the server under other configurations and
# compare. Usage: eval/crossconfig.sh <mode-args...>   (e.g. --deterministic)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
mode=("$@")
tag=$(echo "${mode[*]}" | tr -c "a-z0-9" "_")
port=18095
model=$ROOT/models/Qwen3-0.6B
out=$ROOT/results/crossconfig/$tag
mkdir -p $out
base="--max-running-req 32 --max-seq-len 8192 --max-total-tokens 65536 --chunked-prefill-size 1024"
configs=(
  "base|"
  "chunk512|--chunked-prefill-size 512"
  "nochunk|--chunked-prefill-size 0"
  "eager|--cuda-graph-bs 0 --prefill-cuda-graph-max-tokens 0"
  "maxreq4|--max-running-req 4"
  "nooverlap|--disable-cpu-gpu-overlap"
  "page32|--page-size 32"
  "restart|"
)
for entry in "${configs[@]}"; do
  name=${entry%%|*}; extra=${entry#*|}
  $ROOT/scripts/serve.sh $port $model "${mode[@]}" $base $extra --log-dir $ROOT/logs > $out/$name.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 150); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  if [[ $name == base ]]; then
    python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $model --checks none \
      --concurrency 8 --save $out/refs.json > $out/$name.txt 2>&1
  else
    python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $model --checks none \
      --concurrency 8 --compare $out/refs.json --out $out/$name.json > $out/$name.txt 2>&1
  fi
  echo "$name: $(grep -E "^compare|saved|DETERM|NONDET|Error" $out/$name.txt | tr "\n" " ")"
  kill $pid; wait $pid 2>/dev/null; sleep 3
done
