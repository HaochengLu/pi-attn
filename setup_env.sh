#!/usr/bin/env bash
# Reproducible, self-contained environment for the PI-Attn project.
# Everything lives under $ROOT/env; nothing is installed into the host system.
# Re-running is idempotent. After it finishes: `source $ROOT/env.sh`.
set -euo pipefail
ROOT=/uniparser/users/luhaocheng/pi_attn
ENV=$ROOT/env
CUDA_VER=12.6.3
CUDA_RUN=cuda_12.6.3_560.35.05_linux.run
PY_VER=3.12
TORCH_VER=2.13.0+cu126
TORCH_INDEX=https://download.pytorch.org/whl/cu126
FLASHINFER_VER=0.6.18
RUST_TOOLCHAIN=1.95.0
mkdir -p $ENV/downloads $ENV/tmp
export TMPDIR=$ENV/tmp

# 1. CUDA toolkit (toolkit only, no driver) into $ENV/cuda-12.6
if [[ ! -x $ENV/cuda-12.6/bin/nvcc ]]; then
  [[ -f $ENV/downloads/$CUDA_RUN ]] || curl -L --retry 5 -o $ENV/downloads/$CUDA_RUN \
    https://developer.download.nvidia.cn/compute/cuda/$CUDA_VER/local_installers/$CUDA_RUN
  sh $ENV/downloads/$CUDA_RUN --silent --toolkit --toolkitpath=$ENV/cuda-12.6 \
    --no-opengl-libs --no-man-page --no-drm --override --tmpdir=$ENV/tmp
fi

# 1b. Host compiler: CUDA 12.6 nvcc supports gcc <= 13; the host has gcc 14.
#     gcc/g++ 13 from conda-forge (via a local micromamba) into $ENV/gcc13.
if [[ ! -x $ENV/gcc13/bin/x86_64-conda-linux-gnu-g++ ]]; then
  mkdir -p $ENV/mamba
  [[ -f $ENV/downloads/micromamba.tar.bz2 ]] || curl -sL -o $ENV/downloads/micromamba.tar.bz2 \
    https://micro.mamba.pm/api/micromamba/linux-64/latest
  tar -xjf $ENV/downloads/micromamba.tar.bz2 -C $ENV/mamba bin/micromamba
  MAMBA_ROOT_PREFIX=$ENV/mamba $ENV/mamba/bin/micromamba create -y -p $ENV/gcc13 --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge gcc_linux-64=13 gxx_linux-64=13
fi

# 2. Rust toolchain into $ENV/rustup and $ENV/cargo
export RUSTUP_HOME=$ENV/rustup CARGO_HOME=$ENV/cargo
if [[ ! -x $CARGO_HOME/bin/cargo ]]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal \
    --default-toolchain $RUST_TOOLCHAIN
fi

# 3. Python + engine venv (torch cu126 runs on driver >= 525 via CUDA 12 minor-version compat)
export UV_PYTHON_INSTALL_DIR=$ENV/python UV_CACHE_DIR=$ENV/uv-cache
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
if [[ ! -x $ENV/venv/bin/python ]]; then
  uv venv --python $PY_VER $ENV/venv
fi
uv pip install --python $ENV/venv/bin/python \
  "torch==$TORCH_VER" --index-url $TORCH_INDEX --extra-index-url https://mirrors.ivolces.com/pypi/simple/ --index-strategy unsafe-best-match
uv pip install --python $ENV/venv/bin/python --index-url https://mirrors.ivolces.com/pypi/simple/ \
  --extra-index-url https://pypi.org/simple --extra-index-url $TORCH_INDEX --index-strategy unsafe-best-match \
  "torch==$TORCH_VER" "flashinfer-python==$FLASHINFER_VER" numpy pytest "openai>=1.0,<3.0" "requests>=2.34.2" \
  "transformers" "safetensors" "huggingface_hub" "modelscope" ninja

# 3b. nvcc wrapper for FlashInfer JIT: drops fatbin compression flags
#     (--compress-mode=size), whose output driver 535 rejects as an invalid image.
mkdir -p $ENV/bin
cat > $ENV/bin/nvcc-nocompress <<'EON'
#!/usr/bin/env bash
args=()
for a in "$@"; do
  case "$a" in --compress-mode=*|-Xfatbin=-compress-all) ;; *) args+=("$a") ;; esac
done
exec "$CUDA_HOME/bin/nvcc" "${args[@]}"
EON
chmod +x $ENV/bin/nvcc-nocompress

cat > $ROOT/env.sh <<EOS
# source this file to activate the PI-Attn environment
export ROOT=$ROOT
export CUDA_HOME=$ENV/cuda-12.6
export PATH=$ENV/cuda-12.6/bin:$ENV/cargo/bin:$ENV/venv/bin:\$PATH
export RUSTUP_HOME=$ENV/rustup CARGO_HOME=$ENV/cargo
export VIRTUAL_ENV=$ENV/venv
export LIBTORCH_USE_PYTORCH=1
export UV_PYTHON_INSTALL_DIR=$ENV/python UV_CACHE_DIR=$ENV/uv-cache
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
export TORCH_LIB=\$(ls -d $ENV/venv/lib/python*/site-packages/torch/lib)
export LD_LIBRARY_PATH=\$TORCH_LIB:$ENV/cuda-12.6/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export HF_ENDPOINT=https://hf-mirror.com
export TMPDIR=$ENV/tmp
export CC=$ENV/gcc13/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/gcc13/bin/x86_64-conda-linux-gnu-g++
export NVCC_CCBIN=\$CXX CUDAHOSTCXX=\$CXX
export FLASHINFER_NVCC=$ENV/bin/nvcc-nocompress FLASHINFER_WORKSPACE_BASE=$ENV/flashinfer-home
# Host-level CXXFLAGS/CFLAGS (this dev box exports CXXFLAGS=-std=c++17) would override the
# engine build.rs C++20 flags; builds must not depend on them.
unset CXXFLAGS CFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
EOS
source $ROOT/env.sh
{
  echo "# Recorded versions ($(date -u +%FT%TZ), host $(hostname))"
  echo "driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
  echo "nvcc: $(nvcc --version | tail -1)"
  echo "rustc: $(rustc --version)"; echo "cargo: $(cargo --version)"
  echo "python: $($ENV/venv/bin/python --version)"
  $ENV/venv/bin/python -c "import torch,flashinfer;print('torch',torch.__version__,'cuda',torch.version.cuda);print('flashinfer',flashinfer.__version__)"
  echo "host gcc: $(gcc --version | head -1)"; echo "build g++: $($CXX --version | head -1)"
} | tee $ROOT/VERSIONS.txt
