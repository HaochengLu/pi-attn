import math, os, sys, torch
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../kernels"))
import bench_attention as B
res = {}
for var in [11, 12]:
    ext = load(name=f"pi_fuse_probe_z{var}", sources=[os.path.join(HERE, "pi_attn_torch.cu")], extra_include_paths=[HERE],
               extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr", f"-DVAR={var}"])
    for (b, kv, hq, shared) in [(8, 2048, 16, False), (8, 4096, 16, True), (32, 4096, 16, True), (32, 1024, 16, False), (1, 8192, 32, False)]:
        k, v, indptr, indices, last, lens = B.make_cache(b, kv, 8, shared)
        q = torch.randn(b, hq, 128, device="cuda", dtype=torch.bfloat16)
        rows = b * -(-kv // 512)
        wo = torch.empty(rows, hq, 128, device="cuda"); wm = torch.empty(rows, hq, 2, device="cuda")
        cnt = torch.zeros(rows, hq, dtype=torch.int32, device="cuda")
        t = B.timeit(lambda: ext.decode(q, k, v, indptr, indices, lens, wo, wm, cnt, kv, 1 / math.sqrt(128)))
        print(f"var{var} b{b} kv{kv} hq{hq} shared={shared}: {t}", flush=True)
