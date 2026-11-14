#!/bin/bash
# Round 4 integrated validation of the merged tree (61ff596 + merged.diff). $1 = H100 | classA.
# H100, one sandbox: the engine builds on the CPUs while the kernel suites run on the GPU (two streams); then the
#   kernel benches alone on the GPU; then the engine: reference rollouts, detcheck, PI vs FlashInfer mode interleaved
#   (the round-3 protocol, results/modal/H100_round3/engine.sh). Any bit-identity failure stops the run.
# classA (A10G): test_invariance, test_directed, xgpu_hashes against the A800 reference.
set -u
t0=$(date +%s); el() { echo "$(( $(date +%s) - t0 ))s"; }
FAIL=0; bad() { echo "BIT-IDENTITY/SUITE FAILURE: $*"; FAIL=1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
R=/pi; E=$R/env
cd $R && git apply /work/merged.diff && echo "patch applied (HEAD $(git rev-parse --short HEAD) + merged.diff)" || exit 1
if [ "$1" = H100 ]; then
  ( export HF_ENDPOINT=https://huggingface.co
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
    cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(el))" | tee /work/cargo.done
    wait ) &
  engpid=$!
fi
# ---------------- kernel suites
cd $R/kernels
python -c "import torch, flashinfer; print('torch', torch.__version__, 'flashinfer', flashinfer.__version__, 'cc', torch.cuda.get_device_capability())"
python -c "import test_invariance as T; print('attn ext WGMMA', T.ext.WGMMA, 'VARIANTS', T.VARIANTS, 'DECODE_VARIANTS', T.DECODE_VARIANTS)" 2>&1 | tail -3; echo "attn ext build exit=${PIPESTATUS[0]} ($(el))"
REF=$R/results/modal/H100_sm90/hashes_H100.json; [ "$1" = classA ] && REF=$R/results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json
( # stream 2: the GEMM extension
  python ../eval/xgpu_hashes.py > /work/hashes_new.json 2> /work/hashes.err; echo "xgpu_hashes exit=$? ($(el))" > /work/s2.log
  python - $REF >> /work/s2.log 2>&1 <<'PY'
import json, sys
a = json.load(open("/work/hashes_new.json")); b = json.load(open(sys.argv[1]))
print("device", a["device"], "vs reference", b["device"], sys.argv[1])
for kind in ("gemm", "attention"):
    pi = sum(a[kind][c]["pi"] == b[kind][c]["pi"] for c in b[kind])
    full = sum(a[kind][c] == b[kind][c] for c in b[kind])
    print(f"HASHES {kind}: PI identical {pi}/{len(b[kind])}; PI and vendor both identical {full}/{len(b[kind])}; cases in new {len(a[kind])}")
    for c in b[kind]:
        if a[kind][c]["pi"] != b[kind][c]["pi"]: print("  DIFFERENT", kind, c, a[kind][c]["pi"], "ref", b[kind][c]["pi"])
PY
  if [ "$1" = H100 ]; then
    timeout 1200 python test_gemm.py > /work/test_gemm.log 2>&1; echo "test_gemm exit=$? ($(el))" >> /work/s2.log
    grep -E "INVARIANCE|Error|error" /work/test_gemm.log | cut -c1-600 >> /work/s2.log
  fi ) &
s2=$!
[ "$1" = classA ] && wait $s2   # 24 GB: one extension at a time
python test_invariance.py > /work/inv.log 2>&1; rc=$?; echo "test_invariance exit=$rc ($(el))"; tail -3 /work/inv.log | cut -c1-300; [ $rc = 0 ] || bad test_invariance
python test_directed.py > /work/dir.log 2>&1; rc=$?; echo "test_directed exit=$rc ($(el))"; tail -2 /work/dir.log | cut -c1-300; [ $rc = 0 ] || bad test_directed
if [ "$1" = H100 ]; then
  python test_matrix.py > /work/mat.log 2>&1; rc=$?; echo "test_matrix exit=$rc ($(el))"; tail -2 /work/mat.log | cut -c1-300; [ $rc = 0 ] || bad test_matrix
  python test_mutants.py > /work/mut.log 2>&1; rc=$?; echo "test_mutants exit=$rc ($(el))"; grep -E "mutant|UNCAUGHT|control|rror" /work/mut.log | cut -c1-300; [ $rc = 0 ] || bad test_mutants
fi
wait $s2; cat /work/s2.log
grep -q "xgpu_hashes exit=0" /work/s2.log || bad xgpu_hashes
grep -q "HASHES gemm: PI identical 108/108" /work/s2.log || bad "gemm hashes"
grep -q "HASHES attention: PI identical 5/5" /work/s2.log || bad "attention hashes"
echo "=== HASHES_JSON_BEGIN"; python -c "import json;print(json.dumps(json.load(open('/work/hashes_new.json'))))"; echo "=== HASHES_JSON_END"
if [ "$1" = classA ]; then echo "class A FAIL=$FAIL"; echo "total $(el)"; exit $FAIL; fi
grep -q "test_gemm exit=0" /work/s2.log || bad test_gemm
python /work/dec_r4_bench.py eq > /work/eq.log 2>&1; grep -E "EQ|MISMATCH|rror|Trace" /work/eq.log | cut -c1-300
[ "$(grep -c 'EQ .* 0 mismatches' /work/eq.log)" = 2 ] || bad "decode EQ"
python /work/attn_hashes_prefill.py > /work/ah1.log 2>&1; grep -E "HASH|hashes|rror" /work/ah1.log | sed 's/^/PREFILL-VARIANTS /' | cut -c1-420
grep -q "hashes_H100.json: True" /work/ah1.log || bad "attn_hashes (prefill variants)"
python /work/attn_hashes_decode.py > /work/ah2.log 2>&1; grep -E "HASH|hashes|rror" /work/ah2.log | sed 's/^/DECODE-VARIANTS /' | cut -c1-420
grep -q "hashes_H100.json: True" /work/ah2.log || bad "attn_hashes (decode variants)"
echo "kernel suites done ($(el))"
wait $engpid; cat /work/cargo.done
# ---------------- kernel benches, alone on the GPU (the engine build has finished)
timeout 900 python /work/cold_table_r4.py hash > /work/cold.log 2>&1; echo "cold_table exit=$? ($(el))"; grep -E "^(TABLE|HASHX)|Error|error" /work/cold.log | cut -c1-1200
grep -q "hashes differing from hashes_H100.json: 0 \[\]" /work/cold.log || bad "HASHX"
echo "FAIL=$FAIL after the kernel suites"
if [ $FAIL != 0 ]; then echo "STOP: bit identity failed; benches and engine skipped"; echo "total $(el)"; exit 3; fi
python /work/dec_grid.py 2>&1 | grep -E "DEC|rror|Trace"; echo "dec bench done ($(el))"
PI_PRE_VARIANTS=5,6,7,0 python -u /work/pre_bench.py 2>&1 | grep -E "PRE|rror"; echo "pre bench done ($(el))"
PI_PRE_ENGINE=1 PI_PRE_VARIANTS=5,6,7,0 python -u /work/pre_bench.py 2>&1 | grep -E "PRE|rror" | sed 's/^/ENGSHAPE /'; echo "pre bench (engine shapes) done ($(el))"
# ---------------- engine
grep -E "^error|error\[|error:" -A6 /work/build.log | head -40
echo "C7511 lines in build: $(grep -c C7511 /work/build.log)"
grep C7511 /work/build.log | grep -o "in the function '[A-Za-z0-9_]*" | cut -c1-110 | sort | uniq -c
source $R/env.sh
export FLASHINFER_CUDA_ARCH=90 HF_ENDPOINT=https://huggingface.co
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
ok=0
if serve deterministic B /work/out/check.server.log; then
  python eval/xgpu_rollout.py --url http://127.0.0.1:$port --ref results/modal/H100_sm90/H100_prechange.rollout.gen.json --out /work/out/rollout.gen.json 2>&1 | tail -5
  rc=${PIPESTATUS[0]}; echo "xgpu_rollout exit=$rc ($(el))"; [ $rc = 0 ] && ok=1
  python detcheck/detcheck.py --url http://127.0.0.1:$port --tokenizer $M --api openai --flush flood:270000 --logprobs \
    --out /work/out/detcheck.json > /work/out/detcheck.log 2>&1; drc=$?; echo "detcheck exit=$drc ($(el))"; [ $drc = 0 ] || ok=0
  cut -c1-330 /work/out/detcheck.log | tail -12
  sed 's/\x1b\[[0-9;]*m//g' /work/out/check.server.log | grep "tile choice" | sed 's/^.*PI GEMM/PI GEMM/' | cut -c1-420 | head -8
  sed 's/\x1b\[[0-9;]*m//g' /work/out/check.server.log | grep -i "decode" | cut -c1-300 | head -6
else echo "PI server failed to start"; tail -30 /work/out/check.server.log; fi
stop
if [ $ok != 1 ]; then echo "STOP: ROLLOUTS OR DETCHECK FAILED: throughput runs skipped"; tail -40 /work/out/check.server.log | cut -c1-300; echo "total $(el)"; exit 2; fi
summary() {
python - <<'PY'
import glob, json, statistics as st
for tab in ("T3", "T4", "HC"):
    d = {}
    for m in ("flashinfer", "deterministic"):
        for f in sorted(glob.glob(f"/work/out/{m}.{tab}.*.json")):
            for r in json.load(open(f)):
                e = d.setdefault((r["in"], r["out"], r["concurrency"]), {}).setdefault(m, {"tok": [], "ttft": [], "tpot": []})
                e["tok"] += r["out_tok_s_all"]; e["ttft"].append(r["ttft_ms"]); e["tpot"].append(r["tpot_ms"])
    for k, v in d.items():
        if len(v) < 2: print(tab, k, "INCOMPLETE", v); continue
        f, p = v["flashinfer"], v["deterministic"]
        ft, pt = st.median(f["tok"]), st.median(p["tok"]); fT, pT = st.mean(f["ttft"]), st.mean(p["ttft"])
        print(f"ENG {tab} {k[0]}/{k[1]}/c{k[2]}: out tok/s FI {ft:.0f} PI {pt:.0f} ({(pt/ft-1)*100:+.1f}%) | TTFT ms FI {fT:.1f} PI {pT:.1f} ({(pT/fT-1)*100:+.1f}%) "
              f"| tpot ms FI {st.mean(f['tpot']):.3f} PI {st.mean(p['tpot']):.3f} | all tok/s FI {f['tok']} PI {p['tok']} | TTFT per server FI {[round(x,1) for x in f['ttft']]} PI {[round(x,1) for x in p['ttft']]}")
PY
}
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
  echo "--- summary after round $rnd"; summary
done
for f in /work/out/deterministic.*.server.log; do echo "$(basename $f): error|panic lines $(grep -ci 'error\|panic' $f)"; done
echo "total $(el)"
