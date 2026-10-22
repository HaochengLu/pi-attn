import os, torch, flashinfer
from torch.utils.cpp_extension import load
KD = "/uniparser/users/luhaocheng/pi_attn/kernels"
CUT = os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include")
os.makedirs("/tmp/occ_build", exist_ok=True)
m = load(name="occp", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "occ_probe.cu")], extra_include_paths=[KD, CUT],
         extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89"], build_directory="/tmp/occ_build")
torch.zeros(1, device="cuda")
print(m.run())
