#!/bin/bash
# vLLM 0.17.1 throughput, Qwen3.5-4B, stock vs exact GDN: 256 requests x 200 prompt tokens x 256 new tokens with
# vLLM's default and batch-invariant dense ops and with n-gram speculation (piecewise CUDA graphs); then prefill-bound
# (256 x 2000 prompt tokens, 1 new token) and decode-bound (256 x 16 prompt tokens, 256 new tokens) shapes, and MTP
# drafting (3 drafts, vLLM's default graphs and async scheduling), and batches of 1 and 8 requests (200 prompt tokens,
# 512 new); and fully exact runs, exact GDN with the batch-invariant dense ops of vLLM or of PI-Attn (--pi-gemm), at
# 1, 8 and 256 requests, prefill-bound, and with MTP and n-gram drafting. KV-cache size and concurrency from vLLM's
# log. Three rounds, stock and exact interleaved (run-to-run spread is ~2%; the README reports medians). Then
# pool(): 1024 requests, where the exact state pool caps the batch (throughput_pool.log).
# usage: gdn/vllm/run_throughput.sh RESULTS_DIR [pool]   (from the repo root; pool: only the pool-bound runs)
set -u
cd "$(dirname "$0")/../.." && source env.sh && OUT=$PWD/$1 && cd gdn/vllm && mkdir -p $OUT
PY=../../env/venv-vllm17/bin/python; export PYTHONPATH=../../env/fla_pkgs
TMP=$OUT/run.tmp; M=../../../models/Qwen3.5-4B
mem() { grep -ohE 'GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $TMP | tr '\n' ' '; }
run() { $PY throughput.py $M "$@" > $TMP 2>&1; echo "Qwen3.5-4B ${CUDAGRAPH_MODE:+piecewise }$(grep RESULT $TMP || tail -2 $TMP) | $(mem)"; }
pool() {  # pool-bound: 1024 requests, more than the exact configurations' pool holds at once (~430 vs ~880)
  rm -f $OUT/throughput_pool.log
  for round in 1 2 3; do
    for ex in "" --exact; do run 1024 200 256 $ex >> $OUT/throughput_pool.log; done
    VLLM_BATCH_INVARIANT=1 run 1024 200 256 --exact --pi-gemm >> $OUT/throughput_pool.log
  done
}
if [ "${2:-}" = pool ]; then pool; rm -f $TMP; exit; fi
rm -f $OUT/throughput.log
for round in 1 2 3; do
  for ex in "" --exact; do VLLM_BATCH_INVARIANT=0 run 256 200 256 $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do VLLM_BATCH_INVARIANT=1 run 256 200 256 $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do CUDAGRAPH_MODE=PIECEWISE run 256 200 256 --spec $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do run 256 2000 1 $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do CUDAGRAPH_MODE=PIECEWISE run 256 2000 1 --spec $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do CUDAGRAPH_MODE=PIECEWISE run 256 16 256 --spec $ex >> $OUT/throughput.log; done
  for ex in "" --exact; do run 256 200 256 --mtp=3 $ex >> $OUT/throughput.log; done
  for n in 1 8; do for ex in "" --exact; do run $n 200 512 $ex >> $OUT/throughput.log; done; done
  # fully exact: exact GDN with batch-invariant dense ops, vLLM's kernel vs PI-Attn's GEMM (--pi-gemm)
  for n in 1 8; do for pi in "" --pi-gemm; do VLLM_BATCH_INVARIANT=1 run $n 200 512 --exact $pi >> $OUT/throughput.log; done; done
  VLLM_BATCH_INVARIANT=1 run 256 200 256 --exact --pi-gemm >> $OUT/throughput.log
  VLLM_BATCH_INVARIANT=1 run 256 2000 1 --exact --pi-gemm >> $OUT/throughput.log
  for pi in "" --pi-gemm; do VLLM_BATCH_INVARIANT=1 run 256 200 256 --mtp=3 --exact $pi >> $OUT/throughput.log; done
  VLLM_BATCH_INVARIANT=1 CUDAGRAPH_MODE=PIECEWISE run 256 200 256 --spec --exact --pi-gemm >> $OUT/throughput.log
done
pool
rm -f $TMP
