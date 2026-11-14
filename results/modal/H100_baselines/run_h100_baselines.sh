#!/bin/bash
# H100 same-GPU baselines: llm_infer_rust (FlashInfer / --deterministic), SGLang 0.5.5 (+deterministic),
# vLLM 0.17.1 (+VLLM_BATCH_INVARIANT) on Qwen3-1.7B with eval/baselines.sh unchanged, then PI vs FlashInfer at
# 64-256 running requests. Builds the repo's env layout (env.sh) from system packages.
set -u
t0=$(date +%s)
export HF_ENDPOINT=https://huggingface.co
R=/pi; E=$R/env
mkdir -p $E/bin $E/gcc13/bin $E/stdcxx $E/syslibs/lib $R/models $R/logs $E/tmp $E/cache
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
for v in venv-sglang:sglang venv-vllm17:vllm17; do
  IFS=: read -r dir lock <<< "$v"
  (uv venv -q -p 3.12.13 $E/$dir && uv pip install -q --python $E/$dir/bin/python --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match -r $R/requirements/$lock.txt) > /work/$dir.log 2>&1 &
done
(hf download Qwen/Qwen3-1.7B --local-dir $R/models/Qwen3-1.7B > /work/dl.log 2>&1; cd $R/models && \
 grep -E 'Qwen3-1.7B/(model.*safetensors|config.json|tokenizer.json|generation_config.json)' $R/requirements/models.sha256 \
 | sha256sum -c --quiet && echo "model sha256 OK") &
source $R/env.sh
export FLASHINFER_CUDA_ARCH=90 HF_ENDPOINT=https://huggingface.co
cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(( $(date +%s) - t0 ))s)"
wait
for d in venv-sglang venv-vllm17; do echo "$d: $($E/$d/bin/python -c 'import torch;print(torch.__version__)' 2>&1 | tail -1) $(tail -1 /work/$d.log)"; done
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "setup done $(( $(date +%s) - t0 ))s"
cd $R
BASELINES_OUT=/work/out/baselines bash eval/baselines.sh Qwen3-1.7B rust rust-det sglang sglang-det vllm17 vllm17-det 2>&1 | grep -v "^\s*$"
python eval/summarize_baselines.py /work/out/baselines | tee /work/out/baselines/SUMMARY.md
for f in /work/out/baselines/*.server.log; do e=$(basename $f .server.log); [ -f /work/out/baselines/$e.detcheck.json ] || { echo "--- $e server log tail"; tail -15 $f; }; done
echo "baselines done $(( $(date +%s) - t0 ))s"
# PI vs FlashInfer at high concurrency, interleaved, two launches
for l in 1 2; do for mode in flashinfer deterministic; do
  if [[ $mode == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
  scripts/serve.sh 18096 $R/models/Qwen3-1.7B $margs --max-running-req 256 --max-seq-len 8192 --max-total-tokens 262144 \
    --chunked-prefill-size 2048 --log-dir $R/logs > /work/out/hc.$mode.L$l.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:18096/health >/dev/null && break; sleep 2; done
  python eval/loadgen.py --url http://127.0.0.1:18096 --num 256 --repeats 1 --workloads 512:256:64 512:256:128 512:256:256 2048:128:64 \
    --out /work/out/hc.$mode.L$l.json > /work/out/hc.$mode.L$l.log 2>&1
  echo "highconc $mode L$l exit=$?"; kill $pid; wait $pid 2>/dev/null; sleep 5
done; done
python - <<'PY'
import glob, json, statistics
res = {}
for f in glob.glob("/work/out/hc.*.json"):
    mode = f.split("/")[-1].split(".")[1]
    for r in json.load(open(f)):
        res.setdefault(f"{r['in']}/{r['out']}/c{r['concurrency']}", {}).setdefault(mode, []).append((r["out_tok_s"], r["ttft_ms"]))
for k, m in res.items():
    fi = statistics.median(x[0] for x in m["flashinfer"]); pi = statistics.median(x[0] for x in m["deterministic"])
    ft = statistics.median(x[1] for x in m["flashinfer"]); pt = statistics.median(x[1] for x in m["deterministic"])
    print(f"HC {k}: out tok/s FI {fi:.1f} PI {pi:.1f} ({(pi/fi-1)*100:+.1f}%) | TTFT FI {ft:.1f} PI {pt:.1f} ({(pt/ft-1)*100:+.1f}%)")
PY
rm -f /work/out/*.server.log /work/out/baselines/*.server.log.bak
cd /work && tar czf out.tgz --exclude='*.server.log' out && echo "=== TGZ_BEGIN" && base64 -w0 out.tgz && echo && echo "=== TGZ_END"
echo "total $(( $(date +%s) - t0 ))s"
