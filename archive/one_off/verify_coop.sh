#!/usr/bin/env bash
# Verification chain after the cooperative decode change.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd $ROOT/kernels && python test_matrix.py 2>&1 | grep -v -i warn | tail -12; echo "MATRIX_EXIT=${PIPESTATUS[0]}"
cd $ROOT/llm_infer_rust && VENV_DIR=$VIRTUAL_ENV scripts/run-rust-tests.sh cuda 2>&1 | grep -E "test result|FAILED|panicked|failed" | tail -15
out=$ROOT/results/detcheck; mkdir -p $out
port=18097
$ROOT/scripts/serve.sh $port $ROOT/models/Qwen3-1.7B --deterministic --max-running-req 32 --max-seq-len 8192 \
  --max-total-tokens 32768 --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/coop_1.7B.server.log 2>&1 &
for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $ROOT/models/Qwen3-1.7B \
  --out $out/coop_1.7B.json > $out/coop_1.7B.log 2>&1
echo "DETCHECK_EXIT=$?"; tail -12 $out/coop_1.7B.log
bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
[ -d $ROOT/results/e2e_extended_v1 ] || mv $ROOT/results/e2e_extended $ROOT/results/e2e_extended_v1
bash $ROOT/eval/e2e_extended.sh 3 > $ROOT/results/e2e_extended.out 2>&1
python $ROOT/eval/summarize_extended.py $ROOT/results/e2e_extended > $ROOT/results/e2e_extended/SUMMARY.md 2>&1
cat $ROOT/results/e2e_extended/SUMMARY.md
echo VERIFY_DONE
