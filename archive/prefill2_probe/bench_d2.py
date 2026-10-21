"""PERF PROBE: decode2 vs FlashInfer vs current engine decode (8 warps, 64-key chunks)."""
import math, os, sys, torch, flashinfer
from torch.utils.cpp_extension import load
K = "/uniparser/users/luhaocheng/pi_attn/kernels"
sys.path.insert(0, K)
from bench_decode_variants import cache, D, C, ext, timeit
H = os.path.dirname(os.path.abspath(__file__))
p2 = load(name="pi_p2", sources=[os.path.join(H, "p2_ext.cu")], extra_include_paths=[K, H],
          extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_89,code=sm_89", "--expt-relaxed-constexpr"],
          build_directory="/tmp/pi_p2_build")
dev = "cuda"
for hq, hkv in [(16, 8), (32, 8)]:
    for batch, kv_len, shared in [(1, 1024, False), (8, 1024, False), (32, 1024, False), (1, 32768, False), (2, 16384, False),
                                  (8, 4096, True), (8, 16384, False), (32, 4096, True), (32, 8192, False), (128, 2048, False)]:
        k, v, indptr, indices, last, lens = cache(batch, kv_len, hkv, shared)
        q = torch.randn(batch, hq, D, device=dev, dtype=torch.bfloat16)
        groups = -(-kv_len // 512)
        ws_o = torch.empty(batch * groups, hq, D, device=dev); ws_ml = torch.empty(batch * groups, hq, 2, device=dev)
        cur = timeit(lambda: ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, kv_len, 1 / math.sqrt(D), 8))
        d2 = timeit(lambda: p2.decode2(q, k, v, indptr, indices, lens, kv_len, 1 / math.sqrt(D)))
        ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, D, k.shape[1], q_data_type=torch.bfloat16)
        fi = timeit(lambda: w.run(q, (k, v)))
        print(f"hq{hq} b{batch} {kv_len}{' S' if shared else '  '} fi {fi:7.1f} | cur {cur:7.1f} | d2 {d2:7.1f} ({d2 / fi:.2f})", flush=True)
