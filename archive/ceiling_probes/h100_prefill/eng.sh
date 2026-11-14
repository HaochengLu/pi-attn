#!/bin/bash
# H100 engine run for the new sm_90 prefill kernel: build llm_infer_rust with the uploaded kernel files, replay the
# H100 reference rollouts (bitwise), then TTFT / throughput at the paper's Table 3 workloads, PI and FlashInfer.
set -u
t0=$(date +%s)
export HF_ENDPOINT=https://huggingface.co
R=/pi; E=$R/env
cp /work/pi_attn_sm90.cuh /work/pi_launch.cu /work/pi_attn_kernels.cu $R/llm_infer_rust/src/kernels/pi/
cp /work/pi_attn_torch.cu /work/test_invariance.py $R/kernels/
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
# kernel-level checks of the same files on the GPU while cargo builds on the CPUs
(cd $R/kernels && env -u CC -u CXX -u NVCC_CCBIN -u CUDAHOSTCXX -u TORCH_EXTENSIONS_DIR -u TMPDIR PATH=/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin bash -c '
 python test_invariance.py > /work/ti.log 2>&1; echo "test_invariance exit=$?" >> /work/ti.log
 python /work/attn_hashes.py > /work/hash.log 2>&1
 PI_PRE_ENGINE=1 PI_PRE_VARIANTS=5,0,6,7 python -u /work/pre_bench.py > /work/bench_eng.log 2>&1') &
kpid=$!
cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(( $(date +%s) - t0 ))s)"
wait $kpid
tail -2 /work/ti.log; grep "HASH\|hashes" /work/hash.log | cut -c1-200; grep PRE /work/bench_eng.log
grep -c "C7511" /work/build.log | sed 's/^/C7511 lines in build: /'; grep -E "^error|error\[" -A6 /work/build.log | head -30
wait
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd $R
port=18099
serve() {  # mode
  if [[ $1 == flashinfer ]]; then margs="--attention-backend flashinfer"; else margs="--deterministic"; fi
  scripts/serve.sh $port $R/models/Qwen3-1.7B $margs --max-running-req 32 --max-seq-len 32768 --max-total-tokens 131072 \
    --chunked-prefill-size 2048 --log-dir $R/logs > /work/out/$1.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 300); do curl -s -m 2 localhost:$port/health >/dev/null && break; sleep 2; done
  echo "server $1 up ($(( $(date +%s) - t0 ))s)"
}
stop() { kill $pid; wait $pid 2>/dev/null; sleep 4; }
for l in 1 2 3; do
  serve flashinfer
  python eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 32 --repeats 1 --out /work/out/flashinfer.L$l.json > /work/out/flashinfer.L$l.log 2>&1; echo "loadgen fi L$l exit=$?"
  stop
  serve pi
  if [ $l = 1 ]; then
    python eval/xgpu_rollout.py --url http://127.0.0.1:$port --ref results/modal/H100_sm90/H100_prechange.rollout.gen.json --out /work/out/new.rollout.gen.json 2>&1 | tail -4
    stop; serve pi
  fi
  python eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 32 --repeats 1 --out /work/out/pi.L$l.json > /work/out/pi.L$l.log 2>&1; echo "loadgen pi L$l exit=$?"
  stop
done
python - <<'PY'
import glob, json, statistics
res = {}
for f in sorted(glob.glob("/work/out/*.L*.json")):
    mode = f.rsplit("/", 1)[1].split(".")[0]
    for r in json.load(open(f)):
        key = f"{r['in']}/{r['out']}/c{r['concurrency']}"
        res.setdefault(key, {}).setdefault(mode, []).append((r["out_tok_s"], r["ttft_ms"]))
for key, m in res.items():
    if "flashinfer" not in m or "pi" not in m: print(key, m); continue
    fi, pi = statistics.median(x[0] for x in m["flashinfer"]), statistics.median(x[0] for x in m["pi"])
    ft, pt = statistics.median(x[1] for x in m["flashinfer"]), statistics.median(x[1] for x in m["pi"])
    print(f"E2E {key}: out tok/s FI {fi:.1f} PI {pi:.1f} ({(pi / fi - 1) * 100:+.1f}%) | TTFT FI {ft:.1f} PI {pt:.1f} ({(pt / ft - 1) * 100:+.1f}%) | TTFT runs FI {[round(x[1], 1) for x in m['flashinfer']]} PI {[round(x[1], 1) for x in m['pi']]}")
PY
grep -ci "error\|panic" /work/out/pi.server.log | sed 's/^/pi server error|panic lines: /'
echo "total $(( $(date +%s) - t0 ))s"
