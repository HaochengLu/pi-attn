# source this file to activate the PI-Attn environment
export ROOT=/uniparser/users/luhaocheng/pi_attn
export CUDA_HOME=/uniparser/users/luhaocheng/pi_attn/env/cuda-12.6
export PATH=/uniparser/users/luhaocheng/pi_attn/env/cuda-12.6/bin:/uniparser/users/luhaocheng/pi_attn/env/cargo/bin:/uniparser/users/luhaocheng/pi_attn/env/venv/bin:$PATH
export RUSTUP_HOME=/uniparser/users/luhaocheng/pi_attn/env/rustup CARGO_HOME=/uniparser/users/luhaocheng/pi_attn/env/cargo
export VIRTUAL_ENV=/uniparser/users/luhaocheng/pi_attn/env/venv
export LIBTORCH_USE_PYTORCH=1
export UV_PYTHON_INSTALL_DIR=/uniparser/users/luhaocheng/pi_attn/env/python UV_CACHE_DIR=/uniparser/users/luhaocheng/pi_attn/env/uv-cache
export UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone
export TORCH_LIB=$(ls -d /uniparser/users/luhaocheng/pi_attn/env/venv/lib/python*/site-packages/torch/lib)
export LD_LIBRARY_PATH=$TORCH_LIB:/uniparser/users/luhaocheng/pi_attn/env/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/uniparser/users/luhaocheng/pi_attn/env/hf-home
export TMPDIR=/uniparser/users/luhaocheng/pi_attn/env/tmp
export CC=/uniparser/users/luhaocheng/pi_attn/env/gcc13/bin/x86_64-conda-linux-gnu-gcc CXX=/uniparser/users/luhaocheng/pi_attn/env/gcc13/bin/x86_64-conda-linux-gnu-g++
export NVCC_CCBIN=$CXX CUDAHOSTCXX=$CXX
export FLASHINFER_NVCC=/uniparser/users/luhaocheng/pi_attn/env/bin/nvcc-nocompress FLASHINFER_WORKSPACE_BASE=/uniparser/users/luhaocheng/pi_attn/env/flashinfer-home
# Host-level CXXFLAGS/CFLAGS (this dev box exports CXXFLAGS=-std=c++17) would override the
# engine build.rs C++20 flags; builds must not depend on them.
unset CXXFLAGS CFLAGS NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS
