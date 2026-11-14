"""JIT-build the PI-Attn torch extension. Arch list via PI_ATTN_ARCH (default: the local GPU).

A compute-capability 9.0 GPU (Hopper) builds for sm_90a with PI_SM90A defined, which adds the wgmma kernels
(gemm_sm90.cuh, pi_attn_sm90.cuh; the mma.sync kernels are compiled as before). They need CUTLASS's sm90
headers (shipped with the FlashInfer wheel) and, for the TMA descriptors, libcuda (cuTensorMapEncodeTiled).
Probe builds next to the shipped one: PI_ATTN_SUFFIX names another extension, PI_ATTN_CFLAGS adds nvcc flags."""
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

HERE = os.path.dirname(os.path.abspath(__file__))


def cuda_build_flags(arch=None):
    """(arch, extra_cuda_cflags, extra_include_paths, extra_ldflags) of every kernel extension of this GPU."""
    major, minor = torch.cuda.get_device_capability()
    arch = arch or f"{major}{minor}"
    cflags, includes, ldflags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"], [], []
    if arch == "90":
        arch = "90a"
        cflags.append("-DPI_SM90A=1")
        includes.append(os.path.join(os.path.dirname(__import__("flashinfer").__file__), "data", "cutlass", "include"))
        # The link-time stub carries libcuda.so.1's soname; the driver's library is the one loaded at run time.
        ldflags += [f"-L{os.path.join(CUDA_HOME, 'lib64', 'stubs')}", "-lcuda"]
    cflags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return arch, cflags, includes, ldflags


def build(verbose=False):
    arch, cflags, includes, ldflags = cuda_build_flags(os.environ.get("PI_ATTN_ARCH"))
    return load(
        name=f"pi_attn_ext_sm{arch}" + os.environ.get("PI_ATTN_SUFFIX", ""),
        sources=[os.path.join(HERE, "pi_attn_torch.cu")],
        extra_include_paths=[HERE] + includes,
        extra_cuda_cflags=cflags + ["-lineinfo"] + os.environ.get("PI_ATTN_CFLAGS", "").split(),
        extra_ldflags=ldflags,
        build_directory=os.environ.get("PI_ATTN_BUILD_DIR") or None,
        verbose=verbose,
    )
