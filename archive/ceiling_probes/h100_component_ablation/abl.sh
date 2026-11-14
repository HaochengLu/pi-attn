#!/bin/bash
# H100 component ablation: flashinfer / pi_attn / pi_gemm / deterministic, Qwen3-1.7B, Table-3 rows + high concurrency.
set -u
t0=$(date +%s)
export HF_ENDPOINT=https://huggingface.co
R=/pi; E=$R/env
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
wait
cd $R
declare -A margs=( [flashinfer]="--attention-backend flashinfer" [pi_attn]="--attention-backend pi"
                   [pi_gemm]="--row-invariant-gemm --attention-backend flashinfer" [deterministic]="--deterministic" )
for mode in flashinfer pi_attn pi_gemm deterministic; do
  ts=$(date +%s)
  scripts/serve.sh 18096 $R/models/Qwen3-1.7B ${margs[$mode]} --max-running-req 256 --max-seq-len 8192 --max-total-tokens 262144 \
    --chunked-prefill-size 2048 --log-dir $R/logs > /work/out/$mode.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:18096/health >/dev/null && break; sleep 2; done
  echo "$mode up after $(( $(date +%s) - ts ))s"
  python eval/loadgen.py --url http://127.0.0.1:18096 --num 64 --repeats 3 \
    --workloads 64:32:1 64:32:8 64:32:32 512:256:8 2048:128:8 1024:512:32 512:256:64 512:256:128 512:256:256 2048:128:64 \
    --out /work/out/$mode.json > /work/out/$mode.log 2>&1
  echo "$mode loadgen exit=$? ($(( $(date +%s) - ts ))s)"; kill $pid; wait $pid 2>/dev/null; sleep 4
done
grep -h "tile choice" /work/out/deterministic.server.log | head -12
python - <<'PY'
import json
M = ("flashinfer", "pi_attn", "pi_gemm", "deterministic")
d = {m: {(r["in"], r["out"], r["concurrency"]): r for r in json.load(open(f"/work/out/{m}.json"))} for m in M}
for k in d["flashinfer"]:
    f = d["flashinfer"][k]
    s = f"ABL {k[0]}/{k[1]}/c{k[2]}: FI {f['out_tok_s']:.0f} tok/s tpot {f['tpot_ms']:.3f} ttft {f['ttft_ms']:.1f}"
    for m in M[1:]:
        r = d[m][k]
        s += f" | {m} {(r['out_tok_s']/f['out_tok_s']-1)*100:+.1f}% tpot {(r['tpot_ms']-f['tpot_ms'])*1e3:+.0f}us ttft {r['ttft_ms']-f['ttft_ms']:+.1f}ms"
    print(s)
    print("   all:", {m: d[m][k]["out_tok_s_all"] for m in M})
PY
echo "total $(( $(date +%s) - t0 ))s"
