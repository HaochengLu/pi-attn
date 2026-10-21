"""PERF PROBE: prefill MODE 0 (current) / 1 (online in group) / 2 (one accumulator) vs FlashInfer."""
import math, os, sys
import torch, flashinfer
from torch.utils.cpp_extension import load
K = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, K)
from bench_attention import PAGE, D, dev, make_cache, timeit
ext = load(name="pi_probe", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_prefill.cu")],
           extra_include_paths=[K], extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89",
           "--expt-relaxed-constexpr"], build_directory="/tmp/pi_probe_build")
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, q_len, prefix in [(1, 8192, 0), (8, 1024, 0)]:
        kv_len = q_len + prefix
        k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)), iters=20)
        ref = w.run(q, (k, v)).float()
        r = []
        for cfg in (2,):
            for mode in (0, 3, 5, 6, 7):
                f = lambda: ext.probe(q, k, v, qo, indptr, indices, lens, q_len, 1 / math.sqrt(D), cfg, mode)
                err = (f().float() - ref).abs().max().item()
                r.append(f"m{mode} {timeit(f, iters=20) / fi:.2f}")
        print(f"hq{hq} b{batch} q{q_len} p{prefix}: fi {fi:7.1f}us | " + " ".join(r), flush=True)
