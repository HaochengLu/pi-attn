#!/usr/bin/env bash
# Stop evaluation drivers and engine servers started from this project.
for pat in "e2e_matrix.sh" "crossconfig.sh" "eval/loadgen.py" "detcheck/detcheck.py" "sglang-rust --model-path" "sglang.launch_server" "sglang::" "vllm serve" "EngineCore"; do
  for p in $(pgrep -f "$pat"); do
    [[ $p == $$ || $p == $PPID ]] || kill "$p" 2>/dev/null
  done
done
sleep 3
pgrep -af "sglang-rust|loadgen|e2e_matrix" | grep -v pgrep || echo "all stopped"
