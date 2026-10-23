#!/usr/bin/env bash
# Reproducible, self-contained environment for the PI-Attn project.
# Everything lives under $ROOT/env (ROOT = this directory); nothing is installed into the
# host system. Re-running is idempotent. After it finishes: `source env.sh`.
#
# Python packages are installed from the exact lock files in requirements/ (engine venv,
# SGLang and vLLM baseline venvs). Download sources default to the mirrors used on the
# original dev box (China); override them for other networks, e.g.
#   CUDA_URL=https://developer.download.nvidia.com/compute/cuda \
#   CONDA_CHANNEL=conda-forge PYPI_INDEX=https://pypi.org/simple \
#   UV_PYTHON_INSTALL_MIRROR= HF_ENDPOINT=https://huggingface.co bash setup_env.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV=$ROOT/env
CUDA_VER=12.6.3
CUDA_RUN=cuda_12.6.3_560.35.05_linux.run
PY_VER=3.12.13
RUST_TOOLCHAIN=1.95.0
UV_VER=0.12.3
MICROMAMBA_VER=2.9.0
CUDA_URL=${CUDA_URL:-https://developer.download.nvidia.cn/compute/cuda}
CONDA_CHANNEL=${CONDA_CHANNEL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}
PYPI_INDEX=${PYPI_INDEX:-https://mirrors.ivolces.com/pypi/simple/}
UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR-https://registry.npmmirror.com/-/binary/python-build-standalone}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p $ENV/downloads $ENV/tmp $ENV/bin $ENV/cache
export TMPDIR=$ENV/tmp PATH=$ENV/bin:$PATH XDG_CACHE_HOME=$ENV/cache XDG_CONFIG_HOME=$ENV/config

# 1. CUDA toolkit (toolkit only, no driver) into $ENV/cuda-12.6
if [[ ! -x $ENV/cuda-12.6/bin/nvcc ]]; then
  [[ -f $ENV/downloads/$CUDA_RUN ]] || curl -L --retry 5 -o $ENV/downloads/$CUDA_RUN \
    $CUDA_URL/$CUDA_VER/local_installers/$CUDA_RUN
  sh $ENV/downloads/$CUDA_RUN --silent --toolkit --toolkitpath=$ENV/cuda-12.6 \
    --no-opengl-libs --no-man-page --no-drm --override --tmpdir=$ENV/tmp
fi

# 1b. Host compiler: CUDA 12.6 nvcc supports gcc <= 13; the dev box has gcc 14.
#     gcc/g++ 13 from conda-forge (via a local micromamba) into $ENV/gcc13.
if [[ ! -x $ENV/gcc13/bin/x86_64-conda-linux-gnu-g++ ]]; then
  mkdir -p $ENV/mamba
  [[ -f $ENV/downloads/micromamba.tar.bz2 ]] || curl -sL -o $ENV/downloads/micromamba.tar.bz2 \
    https://micro.mamba.pm/api/micromamba/linux-64/$MICROMAMBA_VER
  tar -xjf $ENV/downloads/micromamba.tar.bz2 -C $ENV/mamba bin/micromamba
  HOME=$ENV/mamba MAMBA_ROOT_PREFIX=$ENV/mamba $ENV/mamba/bin/micromamba create -y -p $ENV/gcc13 --override-channels \
    -c $CONDA_CHANNEL gcc_linux-64=13.4.0 gxx_linux-64=13.4.0 sysroot_linux-64=2.28
fi
# glibc 2.28 headers, so builds also run on older hosts (the A800 box is Ubuntu 20.04, glibc
# 2.31); with the default 2.39 sysroot libstdc++ references __libc_single_threaded (GLIBC_2.32).
ls $ENV/gcc13/conda-meta | grep -q '^sysroot_linux-64-2.28' || HOME=$ENV/mamba MAMBA_ROOT_PREFIX=$ENV/mamba \
  $ENV/mamba/bin/micromamba install -y -p $ENV/gcc13 --override-channels -c $CONDA_CHANNEL sysroot_linux-64=2.28

mkdir -p $ENV/stdcxx
ln -sf ../gcc13/lib/libstdc++.so.6 $ENV/stdcxx/libstdc++.so.6
ln -sf ../gcc13/lib/libgcc_s.so.1 $ENV/stdcxx/libgcc_s.so.1

# 1c. libnuma (needed by the SGLang baseline's sgl_kernel; absent on the dev box) into $ENV/syslibs.
[[ -f $ENV/syslibs/lib/libnuma.so.1 ]] || HOME=$ENV/mamba MAMBA_ROOT_PREFIX=$ENV/mamba $ENV/mamba/bin/micromamba create -y \
  -p $ENV/syslibs --override-channels -c $CONDA_CHANNEL libnuma=2.0.18

# 2. Rust toolchain into $ENV/rustup and $ENV/cargo
export RUSTUP_HOME=$ENV/rustup CARGO_HOME=$ENV/cargo
if [[ ! -x $CARGO_HOME/bin/cargo ]]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal \
    --default-toolchain $RUST_TOOLCHAIN
fi

# 3. uv, Python and the three venvs, each from its lock file.
#    torch cu126 (engine) and cu128 (baselines) run on driver >= 525 via CUDA 12
#    minor-version compatibility; cu13 wheels do not run on driver 535.
#    Pins that are not arbitrary: lm-eval 0.4.13 (0.4.9.1 breaks with transformers 5);
#    sglang 0.5.5.post3 (has --enable-deterministic-inference; 0.5.3 pins a yanked
#    transformers); vllm 0.11.2 (VLLM_BATCH_INVARIANT first ships after 0.11.0).
if ! uv --version 2>/dev/null | grep -q "^uv $UV_VER"; then
  curl -LsSf https://astral.sh/uv/$UV_VER/install.sh | env UV_INSTALL_DIR=$ENV/bin UV_NO_MODIFY_PATH=1 sh
fi
export UV_PYTHON_INSTALL_DIR=$ENV/python UV_CACHE_DIR=$ENV/uv-cache UV_PYTHON_INSTALL_MIRROR
for v in venv:engine:cu126 venv-sglang:sglang:cu128 venv-vllm:vllm:cu128; do
  IFS=: read -r dir lock cu <<< "$v"
  [[ -x $ENV/$dir/bin/python ]] || uv venv --python $PY_VER $ENV/$dir
  uv pip install --python $ENV/$dir/bin/python --index-url $PYPI_INDEX \
    --extra-index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/$cu \
    --index-strategy unsafe-best-match -r $ROOT/requirements/$lock.txt
done

# 3b. nvcc wrapper for FlashInfer JIT: drops fatbin compression flags
#     (--compress-mode=size), whose output driver 535 rejects as an invalid image.
cat > $ENV/bin/nvcc-nocompress <<'EON'
#!/usr/bin/env bash
args=()
for a in "$@"; do
  case "$a" in --compress-mode=*|-Xfatbin=-compress-all) ;; *) args+=("$a") ;; esac
done
exec "$CUDA_HOME/bin/nvcc" "${args[@]}"
EON
chmod +x $ENV/bin/nvcc-nocompress

# 4. env.sh: locates ROOT from its own path, so the checkout can live anywhere.
cat > $ROOT/env.sh <<EOS
# source this file to activate the PI-Attn environment (generated by setup_env.sh)
export ROOT=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)
export CUDA_HOME=\$ROOT/env/cuda-12.6
export PATH=\$ROOT/env/bin:\$ROOT/env/cuda-12.6/bin:\$ROOT/env/cargo/bin:\$ROOT/env/venv/bin:\$PATH
export RUSTUP_HOME=\$ROOT/env/rustup CARGO_HOME=\$ROOT/env/cargo
export VIRTUAL_ENV=\$ROOT/env/venv
export LIBTORCH_USE_PYTORCH=1
export UV_PYTHON_INSTALL_DIR=\$ROOT/env/python UV_CACHE_DIR=\$ROOT/env/uv-cache
export UV_PYTHON_INSTALL_MIRROR=\${UV_PYTHON_INSTALL_MIRROR-$UV_PYTHON_INSTALL_MIRROR}
export TORCH_LIB=\$(ls -d \$ROOT/env/venv/lib/python*/site-packages/torch/lib)
# gcc 13's libstdc++/libgcc_s (env/stdcxx) for JIT-built FlashInfer ops, which need GLIBCXX_3.4.32;
# older hosts (Ubuntu 20.04 on the A800 box) ship an older libstdc++.
export LD_LIBRARY_PATH=\$TORCH_LIB:\$ROOT/env/cuda-12.6/lib64:\$ROOT/env/stdcxx\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export HF_ENDPOINT=\${HF_ENDPOINT:-$HF_ENDPOINT} HF_HOME=\$ROOT/env/hf-home
export TMPDIR=\$ROOT/env/tmp
# Every cache stays inside the project, never in \$HOME (the dev box's home is not shared):
# pip/sglang/vllm/tvm-ffi (XDG), torch extensions, Triton, CUDA JIT, ModelScope.
export XDG_CACHE_HOME=\$ROOT/env/cache XDG_CONFIG_HOME=\$ROOT/env/config
export TORCH_EXTENSIONS_DIR=\$ROOT/env/cache/torch_extensions TRITON_CACHE_DIR=\$ROOT/env/cache/triton
export CUDA_CACHE_PATH=\$ROOT/env/cache/nv MODELSCOPE_CACHE=\$ROOT/env/cache/modelscope
export VLLM_CACHE_ROOT=\$ROOT/env/cache/vllm
export CC=\$ROOT/env/gcc13/bin/x86_64-conda-linux-gnu-gcc CXX=\$ROOT/env/gcc13/bin/x86_64-conda-linux-gnu-g++
export NVCC_CCBIN=\$CXX CUDAHOSTCXX=\$CXX
export FLASHINFER_NVCC=\$ROOT/env/bin/nvcc-nocompress FLASHINFER_WORKSPACE_BASE=\$ROOT/env/flashinfer-home
# Host-level CXXFLAGS/CFLAGS (the dev box exports CXXFLAGS=-std=c++17) would override the
# engine build.rs C++20 flags; builds must not depend on them.
unset CXXFLAGS CFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
EOS
source $ROOT/env.sh

# 5. Models: Qwen3 0.6B/1.7B/4B/8B from ModelScope (downloaded 2026-10-20 from master),
#    checked against the recorded file hashes.
for m in Qwen3-0.6B Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  [[ -f $ROOT/models/$m/config.json ]] || $ENV/venv/bin/modelscope download --model Qwen/$m --local_dir $ROOT/models/$m
done
(cd $ROOT/models && sha256sum --quiet -c $ROOT/requirements/models.sha256)

# 6. Record what was installed. Every line is generated, so re-running never loses one.
py() { $ENV/$1/bin/python -c "import importlib.metadata as m; print(' '.join(p + ' ' + m.version(p) for p in '$2'.split()))"; }
{
  echo "# Recorded by setup_env.sh ($(date -u +%FT%TZ), host $(hostname))"
  echo "gpu/driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "nvcc: $(nvcc --version | tail -1)"
  echo "rustc: $(rustc --version)"; echo "cargo: $(cargo --version)"
  echo "uv: $(uv --version)"; echo "micromamba: $($ENV/mamba/bin/micromamba --version)"
  echo "host gcc: $(gcc --version | head -1)"; echo "build g++: $($CXX --version | head -1)"
  echo "libnuma: $(ls $ENV/syslibs/conda-meta | grep libnuma)"
  echo "engine venv ($($ENV/venv/bin/python --version)): $(py venv 'torch flashinfer-python transformers lm-eval')"
  echo "sglang venv: $(py venv-sglang 'sglang torch flashinfer-python sgl-kernel transformers')"
  echo "vllm venv: $(py venv-vllm 'vllm torch flashinfer-python transformers')"
  echo "full package lists: requirements/{engine,sglang,vllm}.txt"
} > $ROOT/VERSIONS.txt
cat $ROOT/VERSIONS.txt
