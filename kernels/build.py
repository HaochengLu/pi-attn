"""JIT-build the PI-Attn torch extension. Arch list via PI_ATTN_ARCH (default: the local GPU)."""
import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def build(verbose=False):
    major, minor = torch.cuda.get_device_capability()
    arch = os.environ.get("PI_ATTN_ARCH", f"{major}{minor}")
    return load(
        name=f"pi_attn_ext_sm{arch}",
        sources=[os.path.join(HERE, "pi_attn_torch.cu")],
        extra_include_paths=[HERE],
        extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                           "--expt-relaxed-constexpr", "-lineinfo"],
        build_directory=os.environ.get("PI_ATTN_BUILD_DIR") or None,
        verbose=verbose,
    )
