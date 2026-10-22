#!/usr/bin/env bash
# Same-GPU baselines: SGLang and vLLM, each with and without its deterministic mode,
# next to llm_infer_rust (flashinfer / --deterministic). Every server runs detcheck and
# the greedy loadgen workloads of eval/e2e_matrix.sh on one model.
# Usage: eval/baselines.sh [model] [engines...]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
model=${1:-Qwen3-1.7B}; shift || true
engines=${*:-rust rust-det sglang sglang-det sglang-det-triton vllm vllm-det}
port=18098
out=$ROOT/results/baselines/$model
mkdir -p $out
M=$ROOT/models/$model
for eng in $engines; do
  api=openai; flush=flood:270000  # our engine has no flush route; evicting needs > 2x its 131072-token pool
  case $eng in
    rust|rust-det)
      margs="--attention-backend flashinfer"; [[ $eng == rust-det ]] && margs="--deterministic"
      $ROOT/scripts/serve.sh $port $M $margs --max-running-req 32 --max-seq-len 8192 --max-total-tokens 131072 \
        --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/$eng.server.log 2>&1 & ;;
    sglang|sglang-det|sglang-det-triton)
      d=(); [[ $eng == sglang-det* ]] && d=(--enable-deterministic-inference)
      [[ $eng == sglang-det-triton ]] && d+=(--attention-backend triton)
      api=sglang; flush=sglang
      LD_LIBRARY_PATH=$ROOT/env/syslibs/lib:${LD_LIBRARY_PATH:-} $ROOT/env/venv-sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name default --port $port \
        --max-running-requests 32 --context-length 8192 --chunked-prefill-size 2048 --mem-fraction-static 0.8 --cuda-graph-max-bs 32 \
        "${d[@]}" > $out/$eng.server.log 2>&1 & ;;
    vllm|vllm-det)
      flush=vllm
      VLLM_SERVER_DEV_MODE=1 VLLM_BATCH_INVARIANT=$([[ $eng == vllm-det ]] && echo 1 || echo 0) \
        $ROOT/env/venv-vllm/bin/vllm serve $M --served-model-name default --port $port --max-num-seqs 32 \
        --max-model-len 8192 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8 > $out/$eng.server.log 2>&1 & ;;
  esac
  pid=$!
  ok=0
  for i in $(seq 1 450); do curl -s -m 2 localhost:$port/health >/dev/null && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 2; done
  if [[ $ok == 1 ]]; then
    python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $M --api $api --flush $flush \
      --out $out/$eng.detcheck.json > $out/$eng.detcheck.log 2>&1
    echo "$model $eng detcheck exit=$?"
    python $ROOT/eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 32 --repeats 3 \
      --workloads 64:256:1 512:256:8 512:256:32 2048:256:16 4096:128:4 --out $out/$eng.t0.json > $out/$eng.t0.log 2>&1
    echo "$model $eng loadgen exit=$?"
  else
    echo "$model $eng server failed to start (see $out/$eng.server.log)"
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; bash $ROOT/scripts/stop_all.sh >/dev/null 2>&1; sleep 8
done
echo BASELINES_DONE
