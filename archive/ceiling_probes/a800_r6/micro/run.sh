#!/usr/bin/env bash
# Builds and runs the micro-benchmarks and exhaustive checks of this directory on the local GPU.
# usage: run.sh [SM clock in MHz under load, default 1410] > out_<gpu>.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mhz=${1:-1410}
for p in pack_exhaustive mma_negzero mma_rate op_rate lat smem_rate; do
  nvcc -O3 -arch=sm_80 $p.cu -o /tmp/r6_$p || exit 1
  echo "== $p"; /tmp/r6_$p $mhz; rm -f /tmp/r6_$p
done
nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader
