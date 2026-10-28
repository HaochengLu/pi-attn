#!/bin/bash
# vLLM 0.17.1 checks on this GPU: selfcheck.py for Qwen3.5-0.8B and 4B (stock and exact GDN, batch-invariant dense ops
# off and on; with them on: plain and n-gram speculative, whole and 128-token-chunked prefill, short prompts on reused
# slots, prompts ending on a 64-token boundary; and 33 requests with a token budget that cuts the long prompt into
# chunks of a few tokens, run with EXACT_GDN_DEBUG=1, whose counters show the one-token-chunk cases occurred; and
# requests preempted and recomputed in chunks, stock with speculation and exact; and MTP drafting, stock and exact,
# with the same prompt sets, preemption and EXACT_GDN_DEBUG=1; and sampling at temperature 0.8; and exact GDN with
# every dense projection on PI-Attn's GEMM (--pi-gemm) under the same transformations), the pure-attention control
# (Qwen3-0.6B), the conv checks and the check of vLLM's prompt log-probabilities under MTP drafting.
# usage: gdn/vllm/run_vllm_checks.sh RESULTS_DIR [GRAPH_MODE]   (e.g. results/a800/gdn/vllm; from the repo root)
# GRAPH_MODE applies to the non-speculative runs (default: vLLM's); speculative runs use PIECEWISE (see selfcheck.py).
set -u
cd "$(dirname "$0")/../.." && source env.sh && OUT=$PWD/$1 && GM=${2:-} && cd gdn/vllm && mkdir -p $OUT
PY=../../env/venv-vllm17/bin/python; export PYTHONPATH=../../env/fla_pkgs
TMP=$(mktemp); M=../../../models
run() { path=$1; shift; $PY selfcheck.py $path ${NP:-16} 128 "$@" > $TMP 2>&1
        echo "$(basename $path) $* | $(grep RESULT $TMP || tail -2 $TMP) $(grep '^DEBUG\|^PREEMPTIONS\|^PI_GEMM' $TMP)"; }
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
  NP=33 EXACT_GDN_DEBUG=1 VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --exact --one-token-chunks >> $OUT/selfcheck.log
  NP=33 EXACT_GDN_DEBUG=1 VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --spec --exact --one-token-chunks \
    >> $OUT/selfcheck.log
  for ex in "" --exact; do                                         # preempted requests recomputed in 128-token chunks
    [ -n "$ex" ] && VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m $ex --preempt --prefill-chunk=128 >> $OUT/selfcheck.log
    VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --spec $ex --preempt --prefill-chunk=128 >> $OUT/selfcheck.log
  done
  # MTP drafting (3 drafts, vLLM's async scheduling): stock, exact under the same prompt sets, preemption, debug checks
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --mtp=3 >> $OUT/selfcheck.log
  for opt in "" --prefill-chunk=128 --short --aligned "--preempt --prefill-chunk=128"; do
    VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --exact --mtp=3 $opt >> $OUT/selfcheck.log
  done
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --mtp=3 --preempt --prefill-chunk=128 >> $OUT/selfcheck.log
  EXACT_GDN_DEBUG=1 VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --exact --mtp=3 >> $OUT/selfcheck.log
  # sampling (temperature 0.8, seeded): stock and exact, MTP and n-gram drafting, the exact ones with the debug checks
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --mtp=3 --temperature=0.8 >> $OUT/selfcheck.log
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --exact --temperature=0.8 >> $OUT/selfcheck.log
  for sp in --mtp=3 --spec; do
    EXACT_GDN_DEBUG=1 VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --exact $sp --temperature=0.8 >> $OUT/selfcheck.log
  done
  # every dense projection on PI-Attn's GEMM (kernels/pi_vllm.py) instead of vLLM's batch-invariant kernel
  for opt in "" --prefill-chunk=128 --short --aligned "--preempt --prefill-chunk=128" --temperature=0.8; do
    VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --exact --pi-gemm $opt >> $OUT/selfcheck.log
  done
  for opt in "" --prefill-chunk=128 "--preempt --prefill-chunk=128" --temperature=0.8; do
    VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run $m --exact --pi-gemm --mtp=3 $opt >> $OUT/selfcheck.log
    VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --exact --pi-gemm --spec $opt >> $OUT/selfcheck.log
  done
  NP=33 EXACT_GDN_DEBUG=1 VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run $m --spec --exact --pi-gemm \
    --one-token-chunks >> $OUT/selfcheck.log
done
rm -f $OUT/selfcheck_qwen3_attention_only.log
for opt in "" --prefill-chunk=128; do
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=$GM run ../../models/Qwen3-0.6B $opt >> $OUT/selfcheck_qwen3_attention_only.log
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run ../../models/Qwen3-0.6B --spec $opt >> $OUT/selfcheck_qwen3_attention_only.log
done
for f in check_conv check_conv_spec; do $PY $f.py 2>&1 | grep -v -i warn > $OUT/$f.log; done
$PY check_mtp_prompt_logprobs.py $M/Qwen3.5-0.8B 2>&1 | grep -v -i warn > $OUT/check_mtp_prompt_logprobs.log
rm -f $TMP
