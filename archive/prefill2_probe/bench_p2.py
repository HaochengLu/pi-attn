"""PERF PROBE: prefill2 vs FlashInfer vs current PI (planner) — time and max error vs FI."""
import math, os, sys
import torch, flashinfer
from torch.utils.cpp_extension import load
K = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, K)
from bench_attention import PAGE, D, dev, make_cache, timeit, ext
H = os.path.dirname(os.path.abspath(__file__))
p2 = load(name="pi_p2", sources=[os.path.join(H, "p2_ext.cu")], extra_include_paths=[K, H],
          extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr",
                             "-Xptxas=-v"], build_directory="/tmp/pi_p2_build", verbose=False)
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in [(1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (1, 512, 8192), (4, 2048, 2048), (1, 128, 4096), (1, 128, 16384)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        ref = w.run(q, (k, v)).float()
        fi = timeit(lambda: w.run(q, (k, v)), iters=20)
        cur = timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), 0, 0), iters=20)
        r = []
        for sp, gr in [(1, 2), (1, 1), (4, 2), (4, 1), (8, 2), (8, 1)]:
            f = lambda: p2.prefill2(q, k, v, qo, indptr, indices, lens, q_len, kv_len, 1 / math.sqrt(D), sp, gr)
            err = (f().float() - ref).abs().max().item()
            r.append(f"s{sp}g{gr} {timeit(f, iters=20) / fi:.2f}{'' if err < 0.02 else f' ERR {err:.3f}'}")
        print(f"hq{hq} b{batch} q{q_len} p{prefix}: fi {fi:7.1f} cur {cur / fi:.2f} | " + " ".join(r), flush=True)
