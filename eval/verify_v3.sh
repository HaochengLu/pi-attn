#!/usr/bin/env bash
# Verification chain after the online-in-group attention rewrite.
set -uo pipefail
source /uniparser/users/luhaocheng/pi_attn/env.sh
cd $ROOT/llm_infer_rust && VENV_DIR=$VIRTUAL_ENV scripts/run-rust-tests.sh cuda 2>&1 | grep -E "test result|FAILED|failed" | tail -8
out=$ROOT/results/detcheck; mkdir -p $out
port=18097
for M in Qwen3-1.7B Qwen3-8B; do
  $ROOT/scripts/serve.sh $port $ROOT/models/$M --deterministic --max-running-req 32 --max-seq-len 8192 \
    --max-total-tokens 32768 --chunked-prefill-size 2048 --log-dir $ROOT/logs > $out/v3_$M.server.log 2>&1 &
  for i in $(seq 1 200); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  python $ROOT/detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $ROOT/models/$M \
    --out $out/v3_$M.json > $out/v3_$M.log 2>&1
  echo "$M DETCHECK_EXIT=$?"; tail -9 $out/v3_$M.log
  bash $ROOT/scripts/stop_all.sh > /dev/null 2>&1; sleep 5
done
[ -d $ROOT/results/e2e_extended_v2coop ] || mv $ROOT/results/e2e_extended $ROOT/results/e2e_extended_v2coop
bash $ROOT/eval/e2e_extended.sh 3 > $ROOT/results/e2e_extended.out 2>&1
python $ROOT/eval/summarize_extended.py $ROOT/results/e2e_extended > $ROOT/results/e2e_extended/SUMMARY.md 2>&1
cat $ROOT/results/e2e_extended/SUMMARY.md
echo VERIFY_DONE
