"""Eager (no CUDA graph) prefill launches of two builds, as the engine issues them between other kernels: wall time
per launch including the host side, median of many, with a GEMM between launches. usage: r6_eager.py treeA treeB"""
import math, os, sys, time, statistics
import torch
from torch.utils.cpp_extension import load
sys.path.insert(0, os.getcwd())
from build import cuda_build_flags
def build(tree):
    src = os.path.join(os.environ["R6"], tree, "kernels")
    arch, cflags, includes, ldflags = cuda_build_flags(None)
    return load(name="r6eq_" + tree.replace("/", "_"), sources=[os.path.join(src, "pi_attn_torch.cu")], extra_include_paths=[src] + includes,
                extra_cuda_cflags=cflags + ["-lineinfo"], extra_ldflags=ldflags)
exts = {t: build(t) for t in sys.argv[1:3]}
dev, PAGE, D, hkv = "cuda", 16, 128, 8
w = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
for hq in (16, 32):
    for batch, q_len, prefix in [(1, 64, 0), (1, 16, 0), (1, 512, 0), (4, 512, 0), (1, 2048, 0), (8, 512, 0)]:
        kv_len = q_len + prefix; pages = (kv_len + PAGE - 1) // PAGE; n = batch * pages + 1
        layers = [(torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16), torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)) for _ in range(28)]
        perm = (torch.randperm(n - 1, device=dev).int() + 1).contiguous()
        indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages
        lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
        q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        res = {}
        for rnd in range(5):
            for name, e in exts.items():
                ts = []
                for rep in range(40):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    for k, v in layers:
                        x = w @ w  # another kernel between attention launches
                        e.prefill(q, k, v, qo, indptr, perm, lens, q_len, kv_len, 1 / math.sqrt(D), 0, 0)
                    torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / len(layers) * 1e6)
                res.setdefault(name, []).append(statistics.median(ts))
        m = {k: statistics.median(v) for k, v in res.items()}
        a, b = list(m)
        print(f"EAGER hq={hq} b={batch} q={q_len:4d}: {a} {m[a]:7.1f} us per layer (GEMM + prefill) | {b} {m[b]:7.1f} | {b}/{a} {m[b]/m[a]:.3f} ; delta {m[b]-m[a]:+.1f} us", flush=True)
