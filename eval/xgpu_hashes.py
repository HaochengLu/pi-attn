"""Cross-GPU bitwise comparison at kernel level. Inputs are drawn on the CPU from fixed
seeds, so every GPU computes on identical bits; the script prints one SHA-256 per output.
Run it on two GPUs and diff the outputs (eval/xgpu_compare.py).
PI: the row-invariant GEMM (reference id 1 and CUTLASS id 13, which every engine kernel
must match bitwise on one GPU, test_gemm.py) on every Qwen3 projection at M = 1, 8, 128,
1024, and PI-Attn prefill, extend and decode. For contrast the same products through
torch/cuBLAS and FlashInfer's decode, whose kernel choice depends on the GPU.
Usage (from kernels/): python ../eval/xgpu_hashes.py > out.json"""
import hashlib
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import bench_attention as b  # noqa: E402  (PI attention extension, paged-cache helpers)
import test_gemm as tg  # noqa: E402  (PI GEMM extension and the projection shapes)

dev = "cuda"


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def cpu_randn(seed, *shape, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to(dev)


out = {"device": torch.cuda.get_device_name(0), "gemm": {}, "attention": {}}
for i, (name, (N, K)) in enumerate(tg.SHAPES.items()):
    W = cpu_randn(1000 + i, N, K, scale=K ** -0.5)
    X = cpu_randn(2000 + i, 1024, K)
    for M in (1, 8, 128, 1024):
        Xm = X[:M].contiguous()
        y1, y13 = tg.ext.linear(Xm, W, 1, False), tg.ext.linear(Xm, W, 13, False)
        assert torch.equal(y1.view(torch.int16), y13.view(torch.int16)), (name, M)
        out["gemm"][f"{name} M={M}"] = {"pi": h(y1), "cublas": h(Xm @ W.t())}
    del W, X

HQ, HKV, D, PAGE = 32, 8, 128, b.PAGE
for name, batch, q_len, prefix in (("prefill b1 q512", 1, 512, 0), ("prefill b4 q2048", 4, 2048, 0),
                                   ("extend b1 q128 prefix 4096", 1, 128, 4096), ("decode b8 kv 2048", 8, 1, 2047),
                                   ("decode b1 kv 32768", 1, 1, 32767)):
    kv = q_len + prefix
    pages = -(-kv // PAGE)
    k = cpu_randn(3000, batch * pages + 1, PAGE, HKV, D)
    v = cpu_randn(3001, batch * pages + 1, PAGE, HKV, D)
    q = cpu_randn(3002, batch * q_len, HQ, D)
    indices = torch.arange(1, batch * pages + 1, device=dev, dtype=torch.int32)
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages
    lens = torch.full((batch,), kv, device=dev, dtype=torch.int32)
    scale = 1 / math.sqrt(D)
    if q_len == 1:
        rows = batch * -(-kv // b.ext.GROUP_KEYS)
        ws_o = torch.empty(rows, HQ, D, device=dev)
        ws_ml = torch.empty(rows, HQ, 2 * b.ext.DECODE_WARPS, device=dev)
        cnt = torch.zeros(rows, HQ, b.ext.DECODE_WARPS, dtype=torch.int32, device=dev)
        pi = b.ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, kv, scale)
        last = torch.full((batch,), (kv - 1) % PAGE + 1, dtype=torch.int32)
        w = b.flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(indptr.cpu(), indices, last, HQ, HKV, D, PAGE, q_data_type=torch.bfloat16)
        ref = w.run(q, (k, v))
    else:
        qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
        pi = b.ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv, scale, 0)
        last = torch.full((batch,), (kv - 1) % PAGE + 1, dtype=torch.int32)
        w = b.flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
        w.plan(qo.cpu(), indptr.cpu(), indices, last, HQ, HKV, D, PAGE, causal=True, q_data_type=torch.bfloat16)
        ref = w.run(q, (k, v))
    out["attention"][name] = {"pi": h(pi), "flashinfer": h(ref)}
json.dump(out, sys.stdout, indent=1)
