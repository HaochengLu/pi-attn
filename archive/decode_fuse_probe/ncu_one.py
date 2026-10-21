import math, os, sys, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../kernels"))
import bench_attention as B
var = int(sys.argv[1])
ext = load(name=f"pi_fuse_probe_x{var}" if var in (6, 2) else (f"pi_fuse_probe_y{var}" if var in (7, 8) else f"pi_fuse_probe_w{var}"), sources=[os.path.join(HERE, "pi_attn_torch.cu")], extra_include_paths=[HERE],
           extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr", f"-DVAR={var}"])
b, kv, hq = 32, 4096, 16
k, v, indptr, indices, last, lens = B.make_cache(b, kv, 8, True)
q = torch.randn(b, hq, 128, device="cuda", dtype=torch.bfloat16)
rows = b * 8
wo = torch.empty(rows, hq, 128, device="cuda"); wm = torch.empty(rows, hq, 2, device="cuda")
cnt = torch.zeros(rows, hq, dtype=torch.int32, device="cuda")
for _ in range(5): ext.decode(q, k, v, indptr, indices, lens, wo, wm, cnt, kv, 1 / math.sqrt(128))
torch.cuda.synchronize()
