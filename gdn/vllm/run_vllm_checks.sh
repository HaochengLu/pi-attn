#!/bin/bash
# vLLM 0.17.1 checks on this GPU: selfcheck.py for Qwen3.5-0.8B and 4B (stock and exact GDN, batch-invariant dense ops
# off and on; with them on: plain and n-gram speculative, whole and 128-token-chunked prefill, short prompts on reused
# slots, prompts ending on a 64-token boundary), the pure-attention control (Qwen3-0.6B) and the conv checks.
# usage: gdn/vllm/run_vllm_checks.sh RESULTS_DIR [GRAPH_MODE]   (e.g. results/a800/gdn/vllm; from the repo root)
# GRAPH_MODE applies to the non-speculative runs (default: vLLM's); speculative runs use PIECEWISE (see selfcheck.py).
set -u
cd "$(dirname "$0")/../.." && source env.sh && OUT=$PWD/$1 && GM=${2:-} && cd gdn/vllm && mkdir -p $OUT
PY=../../env/venv-vllm17/bin/python; export PYTHONPATH=../../env/fla_pkgs
TMP=$(mktemp); M=../../../models
run() { path=$1; shift; $PY selfcheck.py $path 16 128 "$@" > $TMP 2>&1
        echo "$(basename $path) $* | $(grep RESULT $TMP || tail -2 $TMP)"; }
rm -f $OUT/selfcheck.log
for m in $M/Qwen3.5-0.8B $M/Qwen3.5-4B; do
  for ex in "" --exact; do VLLM_BATCH_INVARIANT=0 CUDAGRAPH_MODE=$GM run $m $ex >> $OUT/selfcheck.log; done
  for opt in "" --prefill-chunk=128 --short --aligned; do
    for ex in "" --exact; do
      [ -n "$opt" ] && [ -z "$ex" ] && continue                      # stock: plain prompts only
      VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m $ex $opt >> $OUT/selfcheck.log
      VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --spec $ex $opt >> $OUT/selfcheck.log
    done
  done
done
rm -f $OUT/selfcheck_qwen3_attention_only.log
for opt in "" --prefill-chunk=128; do
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run ../../models/Qwen3-0.6B $opt >> $OUT/selfcheck_qwen3_attention_only.log
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run ../../models/Qwen3-0.6B --spec $opt >> $OUT/selfcheck_qwen3_attention_only.log
done
for f in check_conv check_conv_spec; do $PY $f.py 2>&1 | grep -v -i warn > $OUT/$f.log; done
rm -f $TMP
