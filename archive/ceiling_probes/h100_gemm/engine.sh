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
cd $R && git apply /work/gemm_final.diff && echo "patch applied" && cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(( $(date +%s) - t0 ))s)"
wait
cd $R
declare -A margs=( [flashinfer]="--attention-backend flashinfer" [deterministic]="--deterministic" )
WL="64:32:1 64:32:8 64:32:32 512:256:8 2048:128:8 1024:512:32 512:256:64 512:256:128 512:256:256 2048:128:64"
for rnd in 1 2; do for mode in flashinfer deterministic; do
  ts=$(date +%s)
  scripts/serve.sh 18096 $R/models/Qwen3-1.7B ${margs[$mode]} --max-running-req 256 --max-seq-len 8192 --max-total-tokens 262144 \
    --chunked-prefill-size 2048 --log-dir $R/logs > /work/out/$mode.$rnd.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:18096/health >/dev/null && break; sleep 2; done
  echo "$mode round $rnd up after $(( $(date +%s) - ts ))s"
  if [ $mode = deterministic ] && [ $rnd = 1 ]; then
    python eval/xgpu_rollout.py --url http://127.0.0.1:18096 --ref results/modal/H100_sm90/H100_prechange.rollout.gen.json --out /work/out/rollout.gen.json
    echo "xgpu_rollout exit=$?"
  fi
  python eval/loadgen.py --url http://127.0.0.1:18096 --num 64 --repeats 2 --workloads $WL --out /work/out/$mode.$rnd.json > /work/out/$mode.$rnd.log 2>&1
  echo "$mode round $rnd loadgen exit=$? ($(( $(date +%s) - ts ))s)"; kill $pid; wait $pid 2>/dev/null; sleep 4
done; done
grep -h "tile choice" /work/out/deterministic.1.server.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-400 | head -12
python - <<'PY'
import json, statistics as st
d = {m: {} for m in ("flashinfer", "deterministic")}
for m in d:
    for rnd in (1, 2):
        for r in json.load(open(f"/work/out/{m}.{rnd}.json")):
            e = d[m].setdefault((r["in"], r["out"], r["concurrency"]), {"tok": [], "tpot": [], "ttft": []})
            e["tok"] += r["out_tok_s_all"]; e["tpot"].append(r["tpot_ms"]); e["ttft"].append(r["ttft_ms"])
for k, f in d["flashinfer"].items():
    p = d["deterministic"][k]
    ft, pt = st.median(f["tok"]), st.median(p["tok"])
    print(f"ENG {k[0]}/{k[1]}/c{k[2]}: FI {ft:.0f} tok/s tpot {st.mean(f['tpot']):.3f} ttft {st.mean(f['ttft']):.1f} | deterministic {pt:.0f} tok/s ({(pt/ft-1)*100:+.1f}%) "
          f"tpot {(st.mean(p['tpot'])-st.mean(f['tpot']))*1e3:+.0f}us ttft {st.mean(p['ttft'])-st.mean(f['ttft']):+.1f}ms | all FI {f['tok']} PI {p['tok']}")
PY
echo "total $(( $(date +%s) - t0 ))s"
