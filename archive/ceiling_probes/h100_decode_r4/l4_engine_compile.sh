set -u
t0=$(date +%s)
R=/pi; E=$R/env
cd $R && git apply /work/r4.diff && echo "patch applied"
# torch extension for sm_90a (compile only on this GPU), in the background
(cd $R/kernels && PI_ATTN_ARCH=90 python -c "import build; build.build(verbose=True)" > /work/ext.log 2>&1; echo "ext build exit=$? ($(( $(date +%s) - t0 ))s)"; grep -E "error|rror:" -B2 -A6 /work/ext.log | head -60) &
mkdir -p $E/bin $E/gcc13/bin $E/stdcxx $E/syslibs/lib $E/tmp $E/cache
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
source $R/env.sh
export FLASHINFER_CUDA_ARCH=90
cd $R/llm_infer_rust && cargo build --release > /work/build.log 2>&1; echo "cargo build exit=$? ($(( $(date +%s) - t0 ))s)"
grep -E "^(error|warning: unused)|error\[|error:" -A12 /work/build.log | head -120
cargo check --release --tests > /work/check.log 2>&1; echo "cargo check --tests exit=$? ($(( $(date +%s) - t0 ))s)"
grep -E "^error|error\[|error:" -A12 /work/check.log | head -80
wait
echo "done ($(( $(date +%s) - t0 ))s)"
