#!/bin/bash
set -u
t0=$(date +%s)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /pi && git apply /work/k.diff && echo "patch applied" || exit 1
cd /work && python tiles_r4.py
python probe_r4.py build > build.log 2>&1; echo "build exit=$? ($(( $(date +%s) - t0 ))s)"; grep -E "error|C7511" build.log | cut -c1-600 | head
fams=""
for f in cl rs; do
  timeout -s KILL 90 python probe_r4.py sanity $f > sanity_$f.log 2>&1; rc=$?
  echo "sanity $f exit=$rc ($(( $(date +%s) - t0 ))s)"; grep -E "SANITY|Error|error" sanity_$f.log | cut -c1-300 | head -60
  [ $rc -eq 0 ] && fams="$fams $f"
done
[ -n "$fams" ] && timeout -s KILL ${MAIN_TIMEOUT:-420} python probe_r4.py main $fams ${MS:-} 2>&1 | grep -E "^(BITCHECK|bitcheck|R4|  ALL)|Error|error" | cut -c1-1500
echo "main exit=${PIPESTATUS[0]} ($(( $(date +%s) - t0 ))s)"
