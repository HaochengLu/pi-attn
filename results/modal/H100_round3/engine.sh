#!/bin/bash
# Round 3 integrated engine validation on H100: llm_infer_rust built from the merged tree (61ff596 + merged.diff).
# 1. H100 reference rollouts replayed bit for bit; detcheck (tokens and log-probabilities).
# 2. PI (--deterministic) vs FlashInfer mode, interleaved FI/PI, two rounds, fresh server per pass:
#    config B (eval/baselines.sh: --max-running-req 32, 131072 tokens): Table 3 workloads (num 64, repeats 2) and
#             Table 4 shapes (num 32, repeats 3, as eval/baselines.sh);
#    config A (--max-running-req 256, 262144 tokens): the high-concurrency rows (num 64, repeats 2).
set -u
t0=$(date +%s); el() { echo "$(( $(date +%s) - t0 ))s"; }
export HF_ENDPOINT=https://huggingface.co
R=/pi; E=$R/env
cd $R && git apply /work/merged.diff && echo "patch applied" || exit 1
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
cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(el))"
grep -E "^error|error\[|error:" -A6 /work/build.log | head -40
echo "C7511 lines in build: $(grep -c C7511 /work/build.log)"
grep C7511 /work/build.log | grep -o "in the function '[A-Za-z0-9_]*" | cut -c1-110 | sort | uniq -c
wait
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd $R
M=$R/models/Qwen3-1.7B; port=18097
serve() {  # serve <flashinfer|deterministic> <A|B> <log>
  local margs="--attention-backend flashinfer"; [ $1 = deterministic ] && margs="--deterministic"
  local cfg="--max-running-req 32 --max-total-tokens 131072"; [ $2 = A ] && cfg="--max-running-req 256 --max-total-tokens 262144"
  scripts/serve.sh $port $M $margs $cfg --max-seq-len 8192 --chunked-prefill-size 2048 --log-dir $R/logs > $3 2>&1 &
  pid=$!
  for i in $(seq 1 450); do curl -s -m 2 localhost:$port/health >/dev/null && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 2; done
  return 1
}
stop() { kill $pid 2>/dev/null; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null; bash $R/scripts/stop_all.sh >/dev/null 2>&1; sleep 5; }
# --- 1. bit identity: reference rollouts, then detcheck, on the merged PI engine
ok=0
if serve deterministic B /work/out/check.server.log; then
  python eval/xgpu_rollout.py --url http://127.0.0.1:$port --ref results/modal/H100_sm90/H100_prechange.rollout.gen.json --out /work/out/rollout.gen.json 2>&1 | tail -5
  rc=${PIPESTATUS[0]}; echo "xgpu_rollout exit=$rc ($(el))"; [ $rc = 0 ] && ok=1
  python detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $M --api openai --flush flood:270000 --logprobs \
    --out /work/out/detcheck.json > /work/out/detcheck.log 2>&1; echo "detcheck exit=$? ($(el))"
  cut -c1-330 /work/out/detcheck.log | tail -12
  grep -h "tile choice" /work/out/check.server.log | sed 's/\x1b\[[0-9;]*m//g; s/^.*PI GEMM/PI GEMM/' | cut -c1-400 | head -8
else echo "PI server failed to start"; tail -30 /work/out/check.server.log; fi
stop
if [ $ok != 1 ]; then echo "ROLLOUTS NOT IDENTICAL: throughput runs skipped"; tail -40 /work/out/check.server.log | cut -c1-300; echo "total $(el)"; exit 2; fi
# --- 2. throughput / TTFT, interleaved
T3="64:32:1 64:32:8 64:32:32 512:256:8 2048:128:8 1024:512:32"
T4="64:256:1 512:256:8 512:256:32 2048:256:16 4096:128:4"
HC="512:256:64 512:256:128 512:256:256 2048:128:64"
for rnd in 1 2; do
  for cfg in B A; do for mode in flashinfer deterministic; do
    ts=$(date +%s)
    if serve $mode $cfg /work/out/$mode.$cfg.$rnd.server.log; then
      up=$(( $(date +%s) - ts ))
      if [ $cfg = B ]; then
        python eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 64 --repeats 2 --workloads $T3 --out /work/out/$mode.T3.$rnd.json > /work/out/$mode.T3.$rnd.log 2>&1; r1=$?
        python eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 32 --repeats 3 --workloads $T4 --out /work/out/$mode.T4.$rnd.json > /work/out/$mode.T4.$rnd.log 2>&1; r2=$?
      else
        python eval/loadgen.py --url http://127.0.0.1:$port --temperature 0 --num 64 --repeats 2 --workloads $HC --out /work/out/$mode.HC.$rnd.json > /work/out/$mode.HC.$rnd.log 2>&1; r1=$?; r2=0
      fi
      echo "$mode cfg $cfg round $rnd: up after ${up}s, loadgen exit=$r1/$r2 ($(el))"
    else echo "$mode cfg $cfg round $rnd: server failed"; tail -20 /work/out/$mode.$cfg.$rnd.server.log | cut -c1-300; fi
    stop
  done; done
done
python - <<'PY'
import glob, json, statistics as st
for tab in ("T3", "T4", "HC"):
    d = {}
    for m in ("flashinfer", "deterministic"):
        for f in sorted(glob.glob(f"/work/out/{m}.{tab}.*.json")):
            for r in json.load(open(f)):
                e = d.setdefault((r["in"], r["out"], r["concurrency"]), {}).setdefault(m, {"tok": [], "ttft": [], "tpot": [], "p99": []})
                e["tok"] += r["out_tok_s_all"]; e["ttft"].append(r["ttft_ms"]); e["tpot"].append(r["tpot_ms"]); e["p99"].append(r["p99_ttft_ms"])
    for k, v in d.items():
        if len(v) < 2: print(tab, k, "INCOMPLETE", v); continue
        f, p = v["flashinfer"], v["deterministic"]
        ft, pt = st.median(f["tok"]), st.median(p["tok"]); fT, pT = st.mean(f["ttft"]), st.mean(p["ttft"])
        print(f"ENG {tab} {k[0]}/{k[1]}/c{k[2]}: out tok/s FI {ft:.0f} PI {pt:.0f} ({(pt/ft-1)*100:+.1f}%) | TTFT ms FI {fT:.1f} PI {pT:.1f} ({(pT/fT-1)*100:+.1f}%) "
              f"| tpot ms FI {st.mean(f['tpot']):.3f} PI {st.mean(p['tpot']):.3f} | all tok/s FI {f['tok']} PI {p['tok']} | TTFT per server FI {[round(x,1) for x in f['ttft']]} PI {[round(x,1) for x in p['ttft']]}")
PY
for f in /work/out/deterministic.*.server.log; do echo "$(basename $f): error|panic lines $(grep -ci 'error\|panic' $f)"; done
echo "total $(el)"
