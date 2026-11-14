#!/bin/bash
# H100 engine check of decode90_kernel: build llm_infer_rust with the patch, detcheck + the baseline loadgen
# workloads (eval/baselines.sh rust-det), the same c1 / c8 workloads with PI_DECODE90=0 in the same sandbox, and
# the H100 reference rollouts replayed bit for bit.
set -u
t0=$(date +%s)
export HF_ENDPOINT=https://huggingface.co
R=/pi; E=$R/env
cd $R && git apply /work/final3.diff && echo "patch applied"
mkdir -p $E/bin $E/gcc13/bin $E/stdcxx $E/syslibs/lib $R/models $R/logs $E/tmp $E/cache /work/out
(apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q software-properties-common libnuma1 && \
 add-apt-repository -y ppa:ubuntu-toolchain-r/test && apt-get update -q && \
 DEBIAN_FRONTEND=noninteractive apt-get install -y -q gcc-13 g++-13) > /work/apt.log 2>&1; echo "apt exit=$?"
ln -sfn /usr/local/cuda $E/cuda-12.6; ln -sfn /venv $E/venv
ln -sf /usr/bin/gcc-13 $E/gcc13/bin/x86_64-conda-linux-gnu-gcc; ln -sf /usr/bin/g++-13 $E/gcc13/bin/x86_64-conda-linux-gnu-g++
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $E/stdcxx/libstdc++.so.6; ln -sf /lib/x86_64-linux-gnu/libgcc_s.so.1 $E/stdcxx/libgcc_s.so.1
ln -sf /usr/lib/x86_64-linux-gnu/libnuma.so.1 $E/syslibs/lib/libnuma.so.1
cat > $E/bin/nvcc-nocompress <<'EON'
#!/usr/bin/env bash
args=()
for a in "$@"; do case "$a" in --compress-mode=*|-Xfatbin=-compress-all) ;; *) args+=("$a") ;; esac; done
exec "$CUDA_HOME/bin/nvcc" "${args[@]}"
EON
chmod +x $E/bin/nvcc-nocompress
(RUSTUP_HOME=$E/rustup CARGO_HOME=$E/cargo bash -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.95.0') > /work/rustup.log 2>&1
(hf download Qwen/Qwen3-1.7B --local-dir $R/models/Qwen3-1.7B > /work/dl.log 2>&1; cd $R/models && \
 grep -E 'Qwen3-1.7B/(model.*safetensors|config.json|tokenizer.json|generation_config.json)' $R/requirements/models.sha256 \
 | sha256sum -c --quiet && echo "model sha256 OK") &
source $R/env.sh
export FLASHINFER_CUDA_ARCH=90 HF_ENDPOINT=https://huggingface.co
cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(( $(date +%s) - t0 ))s)"
grep -E "^error|error\[|error:" -A6 /work/build.log | head -40
wait
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd $R
M=$R/models/Qwen3-1.7B; port=18097
serve() {  # serve <log> <env...> -- server args
  local log=$1; shift
  env "$@" $R/scripts/serve.sh $port $M --deterministic --max-running-req 32 --max-seq-len 8192 --max-total-tokens 131072 \
    --chunked-prefill-size 2048 --log-dir $R/logs > $log 2>&1 &
  pid=$!
  for i in $(seq 1 450); do curl -s -m 2 localhost:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 2; done
  return 1
}
stop() { kill $pid 2>/dev/null; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; bash $R/scripts/stop_all.sh >/dev/null 2>&1; sleep 6; }
# A/B in one sandbox: decode_kernel everywhere (PI_DECODE90=0), then the planner's choice; twice, interleaved.
for rep in 1; do for mode in 0 1; do
  if serve /work/out/ab.$mode.$rep.server.log PI_DECODE90=$mode; then
    python $R/eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 32 --repeats 3 \
      --workloads 64:256:1 512:256:8 2048:128:1 --out /work/out/ab.$mode.$rep.json > /work/out/ab.$mode.$rep.log 2>&1
    echo "AB PI_DECODE90=$mode run $rep loadgen exit=$? ($(( $(date +%s) - t0 ))s)"
    python - <<PY
import json
for l in open("/work/out/ab.$mode.$rep.log"):
    if l.startswith("{"):
        d = json.loads(l); print(f"AB PI_DECODE90=$mode run $rep {d['in']}/{d['out']}/c{d['concurrency']}: out tok/s {d['out_tok_s']} itl_ms {d['itl_ms']} all {d['out_tok_s_all']}")
PY
  else echo "AB PI_DECODE90=$mode server failed"; tail -20 /work/out/ab.$mode.$rep.server.log; fi
  stop
done; done
# The planner's choice: detcheck and the baseline workloads (as results/modal/H100_baselines rust-det).
BASELINES_OUT=/work/out/baselines bash eval/baselines.sh Qwen3-1.7B rust-det 2>&1 | grep -v "^\s*$"
cat /work/out/baselines/rust-det.detcheck.log | cut -c1-300
cat /work/out/baselines/rust-det.t0.log | cut -c1-330
echo "baselines done ($(( $(date +%s) - t0 ))s)"
# The H100 reference rollouts (pre-sm90 engine on H100), replayed bit for bit: concurrency 16, then 1.
if serve /work/out/rollout.server.log PI_DECODE90=1; then
  python eval/xgpu_rollout.py --url http://127.0.0.1:$port --ref results/modal/H100_sm90/H100_prechange.rollout.gen.json --out /work/out/rollout.gen.json 2>&1 | tail -4
  echo "xgpu_rollout exit=$?"
else echo "rollout server failed"; tail -20 /work/out/rollout.server.log; fi
stop
echo "done ($(( $(date +%s) - t0 ))s)"
