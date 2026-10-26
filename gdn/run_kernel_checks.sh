#!/bin/bash
# Kernel-level checks on this GPU, under the engine venv (Triton 3.7.1) and the vLLM venv (Triton 3.6.0):
# test_exact.py, e4 (autotune configs vs bits), e10 (alone vs packed), e14 (fla output kernel, varlen vs fixed-length)
# and, under Triton 3.6.0, e14b (test_exact with each output-kernel config pinned).
# usage: gdn/run_kernel_checks.sh RESULTS_DIR      (e.g. results/a800/gdn; from the repo root)
set -u
cd "$(dirname "$0")/.." && source env.sh && OUT=$PWD/$1 && cd gdn && mkdir -p $OUT
export PYTHONPATH=../env/fla_pkgs:.
clean() { grep -v -i 'warn'; }
rm -f $OUT/e14_fla_o_varlen.log $OUT/e10_pack.log $OUT/e14b_o_config_pins.log
for v in venv venv-vllm17; do
  PY=../env/$v/bin/python; tv=$($PY -c 'import triton; print(triton.__version__)')
  log=$OUT/test_exact.log; [ $v = venv-vllm17 ] && log=$OUT/test_exact_triton360.log
  $PY test_exact.py 2>&1 | clean > $log
  $PY experiments/e4_autotune.py 2>&1 | clean > $OUT/e4_autotune_triton$tv.log
  echo "== Triton $tv" >> $OUT/e10_pack.log
  for s in 0 1 2; do $PY experiments/e10_pack.py $s 2>&1 | grep seed >> $OUT/e10_pack.log; done
  $PY experiments/e14_fla_o_varlen.py 4096 4 2>&1 | clean >> $OUT/e14_fla_o_varlen.log
  if [ $tv = 3.6.0 ]; then
    for i in 0 1 2; do OCFG=$i $PY experiments/e14b_o_config_pins.py 2>&1 | clean >> $OUT/e14b_o_config_pins.log; done
  fi
done
tail -n1 $OUT/test_exact.log $OUT/test_exact_triton360.log
