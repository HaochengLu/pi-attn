"""Probe 20: decode attention at the shapes of the 0.6B 512/256/c128 run (batch 128, 16 query
and 8 KV heads, 512-768 keys), where the ablation puts the slowdown in PI attention.
Run from kernels/: python ../archive/ceiling_probes/c128_decode/probe.py"""
import math
import sys

import torch

sys.path.insert(0, ".")
import bench_attention as b  # noqa: E402

for kv in (512, 513, 576, 640, 704, 768, 1024):
    r = b.bench_decode(128, kv, 16, 8)
    print(f"b=128 hq=16 kv={kv:4d}: " + " ".join(f"{k}={v}" for k, v in r.items())
          + f" | pi/best_fi={r['pi'] / b.best_fi(r):.2f}", flush=True)

# The engine sizes the decode grid from the page-table stride (max_seq_len = 8192 in the
# c128 run: 16 fold groups), not from the batch's longest request. Same lengths, grid of 16.
print("grid sized for 8192 keys, as in the engine's decode graph:")
for batch, kv in ((128, 640), (128, 768), (64, 640), (32, 640), (1, 320)):
    k, v, indptr, indices, last, lens = b.make_cache(batch, kv, 8)
    q = torch.randn(batch, 16, b.D, device=b.dev, dtype=torch.bfloat16)
    res = {}
    for name, mx in (("own", kv), ("8192", 8192)):
        rows = batch * -(-mx // b.ext.GROUP_KEYS)
        ws_o = torch.empty(rows, 16, b.D, device=b.dev)
        ws_ml = torch.empty(rows, 16, 2 * b.ext.DECODE_WARPS, device=b.dev)
        cnt = torch.zeros(rows, 16, b.ext.DECODE_WARPS, dtype=torch.int32, device=b.dev)
        res[name] = b.timeit(lambda: b.ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, mx, 1 / math.sqrt(b.D)))
    print(f"b={batch:3d} kv={kv}: grid for own length {res['own']} us, grid for 8192 {res['8192']} us", flush=True)

# The engine's requests have different lengths and mostly contiguous pages.
print("mixed lengths 512-767, pages contiguous or shuffled:")
import flashinfer  # noqa: E402

for contiguous in (True, False):
    torch.manual_seed(0)
    batch, hq, hkv = 128, 16, 8
    lens_l = [512 + (i * 2) for i in range(batch)]  # 512..766: one decode run's spread
    pages = [(n + b.PAGE - 1) // b.PAGE for n in lens_l]
    n = sum(pages) + 1
    k = torch.randn(n, b.PAGE, hkv, b.D, device=b.dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    order = torch.arange(1, n, device=b.dev) if contiguous else torch.randperm(n - 1, device=b.dev) + 1
    indices = order.int().contiguous()
    indptr = torch.tensor([0] + list(__import__("itertools").accumulate(pages)), device=b.dev, dtype=torch.int32)
    last = torch.tensor([(x - 1) % b.PAGE + 1 for x in lens_l], device=b.dev, dtype=torch.int32)
    lens = torch.tensor(lens_l, device=b.dev, dtype=torch.int32)
    q = torch.randn(batch, hq, b.D, device=b.dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=b.dev, dtype=torch.uint8)
    res = {}
    for name, tc in (("fi", False), ("fi_tc", True)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, b.D, b.PAGE, q_data_type=torch.bfloat16)
        res[name] = b.timeit(lambda: w.run(q, (k, v)))
    mx = max(lens_l)
    rows = batch * -(-mx // b.ext.GROUP_KEYS)
    ws_o = torch.empty(rows, hq, b.D, device=b.dev)
    ws_ml = torch.empty(rows, hq, 2 * b.ext.DECODE_WARPS, device=b.dev)
    cnt = torch.zeros(rows, hq, b.ext.DECODE_WARPS, dtype=torch.int32, device=b.dev)
    res["pi"] = b.timeit(lambda: b.ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, mx, 1 / math.sqrt(b.D)))
    print(f"{'contiguous' if contiguous else 'shuffled  '}: " + " ".join(f"{k}={v}" for k, v in res.items())
          + f" | pi/best_fi={res['pi'] / b.best_fi(res):.2f}", flush=True)
