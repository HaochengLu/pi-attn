#!/usr/bin/env bash
# Record of the verification run (it ran from logs/tp/, the gitignored scratch directory).
# Engine regression after the explicit-S / slot-id GEMM change: the deterministic engine (Qwen3-0.6B,
# whose o/down run S = 2) must reproduce the A800 crossconfig references bit for bit, then pass detcheck.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
bin=$ROOT/build/a800/llm_infer_rust/target/release/sglang-rust
model=$ROOT/models/Qwen3-0.6B
out=$ROOT/logs/tp/engine
mkdir -p $out
base="--max-running-req 32 --max-seq-len 8192 --max-total-tokens 65536 --chunked-prefill-size 1024"
LD_PRELOAD=$TORCH_LIB/libtorch_cuda.so $bin --model-path $model --port 18095 --device cuda --deterministic $base \
  --log-dir $ROOT/logs > $out/server.log 2>&1 &
pid=$!
for i in $(seq 1 150); do curl -s -m 2 localhost:18095/health >/dev/null && break; sleep 2; done
python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:18095 --tokenizer $model --checks none --concurrency 8 \
  --compare $ROOT/results/a800/crossconfig/__deterministic_/refs.json --out $out/compare.json > $out/compare.txt 2>&1
python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:18095 --tokenizer $model > $out/detcheck.txt 2>&1
kill $pid; wait $pid 2>/dev/null
echo "compare: $(grep -E '^compare|DETERM|NONDET|Error' $out/compare.txt | tr '\n' ' ')"
echo "detcheck: $(grep -E 'DETERM|NONDET|Error|mismatch' $out/detcheck.txt | tail -3 | tr '\n' ' ')"
