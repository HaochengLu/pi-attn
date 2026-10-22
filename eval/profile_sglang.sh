#!/usr/bin/env bash
# Where does SGLang's deterministic mode spend its time? nsys traces of batch-1
# generation (10 requests x 64 prompt / 256 output tokens, CUDA-graph nodes traced)
# with and without --enable-deterministic-inference; kernel time by category.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
out=$ROOT/results/sglang_profile
mkdir -p $out
port=18094
M=$ROOT/models/Qwen3-1.7B
for eng in sglang sglang-det; do
  d=(); [[ $eng == sglang-det ]] && d=(--enable-deterministic-inference)
  LD_LIBRARY_PATH=$ROOT/env/syslibs/lib:${LD_LIBRARY_PATH:-} nsys profile -t cuda --cuda-graph-trace=node \
    -o $out/$eng -f true $ROOT/env/venv-sglang/bin/python -m sglang.launch_server --model-path $M --port $port \
    --max-running-requests 32 --context-length 8192 --chunked-prefill-size 2048 --mem-fraction-static 0.8 \
    --cuda-graph-max-bs 32 "${d[@]}" > $out/$eng.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  python - <<PY
import requests, time
t = time.perf_counter()
for _ in range(10):
    requests.post("http://127.0.0.1:$port/generate", json={"input_ids": list(range(1000, 1064)),
                  "sampling_params": {"max_new_tokens": 256, "temperature": 0, "ignore_eos": True}})
print("$eng wall per request (s):", round((time.perf_counter() - t) / 10, 3))
PY
  kill -INT $pid; wait $pid 2>/dev/null; bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
  nsys stats -r cuda_gpu_kern_sum -f csv -o $out/$eng $out/$eng.nsys-rep > /dev/null 2>&1
done
python $ROOT/eval/kernel_categories.py $out/sglang_cuda_gpu_kern_sum.csv $out/sglang-det_cuda_gpu_kern_sum.csv | tee $out/SUMMARY.md
