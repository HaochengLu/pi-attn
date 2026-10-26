#!/bin/bash
# vLLM 0.17.1 throughput, Qwen3.5-4B, 256 requests x 200 prompt tokens x 256 new tokens: stock vs exact GDN, with
# vLLM's default and batch-invariant dense ops, and n-gram speculation (piecewise CUDA graphs); KV-cache size and
# concurrency from vLLM's log.  usage: gdn/vllm/run_throughput.sh RESULTS_DIR   (from the repo root)
set -u
cd "$(dirname "$0")/../.." && source env.sh && OUT=$PWD/$1 && cd gdn/vllm && mkdir -p $OUT
PY=../../env/venv-vllm17/bin/python; export PYTHONPATH=../../env/fla_pkgs
TMP=$(mktemp); M=../../../models/Qwen3.5-4B
mem() { grep -ohE 'GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $TMP | tr '\n' ' '; }
run() { $PY throughput.py $M 256 200 256 "$@" > $TMP 2>&1; echo "Qwen3.5-4B $(grep RESULT $TMP || tail -2 $TMP) | $(mem)"; }
rm -f $OUT/throughput.log
for bi in 0 1; do for ex in "" --exact; do VLLM_BATCH_INVARIANT=$bi run $ex >> $OUT/throughput.log; done; done
for ex in "" --exact; do CUDAGRAPH_MODE=PIECEWISE run --spec $ex | sed 's/^Qwen3.5-4B /Qwen3.5-4B piecewise /' >> $OUT/throughput.log; done
rm -f $TMP
