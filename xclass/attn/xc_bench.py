"""Latency (us per call, CUDA-graph replay as kernels/bench_attention.py) of FlashInfer, stock PI-Attn
(XC=0 build of the same sources) and the XC programs."""
import math
import sys

import flashinfer
import torch

import xc_lib as X

dev = "cuda"
PAGE, D = X.PAGE, X.D
CALLS = 10
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), flush=True)
EXT = {"stock": X.build(0), "xc128": X.build(1, 128), "xc32": X.build(1, 32), "xc16": X.build(1, 16)}
BLK = {"stock": 0, "xc128": 128, "xc32": 32, "xc16": 16}


def timeit(fn, layers, iters=15):
    calls = -(-CALLS // layers) * layers
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i % layers)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / calls)
    ts.sort()
    return round(ts[len(ts) // 2] * 1000, 1)


def warm_gpu(seconds=2.0):
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    while True:
        for _ in range(50):
            a @ a
        e.record()
        torch.cuda.synchronize()
        if s.elapsed_time(e) > seconds * 1000:
            return


def make_cache(batch, kv_len, hkv):
    pages_per = (kv_len + PAGE - 1) // PAGE
    n = batch * pages_per + 1
    layer_bytes = 2 * n * PAGE * hkv * D * 2
    layers = min(28, -(-(160 << 20) // layer_bytes))
    raw = [(torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16),
            torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)) for _ in range(layers)]
    one = torch.ones(n, PAGE, hkv, device=dev)
    kv = {"stock": [(k, v, one) for k, v in raw]}
    for tag in ("xc128", "xc32", "xc16"):
        kv[tag] = [(X.proj_qk(k, BLK[tag]),) + X.proj_v(v) for k, v in raw]
    perm = torch.randperm(n - 1, device=dev).int() + 1
    indices = perm[: batch * pages_per].contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), (kv_len - 1) % PAGE + 1, device=dev, dtype=torch.int32)
    lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
    return raw, kv, indptr, indices, last, lens


def show(kind, desc, r):
    print(f"{kind} {desc}: " + " ".join(f"{k}={v}" for k, v in r.items())
          + " | /fi: " + " ".join(f"{t}={r[t] / r['fi']:.2f}" for t in EXT)
          + " | /stock: " + " ".join(f"{t}={r[t] / r['stock']:.3f}" for t in EXT if t != "stock"), flush=True)


def bench_prefill(batch, q_len, prefix, hq, hkv):
    kv_len = q_len + prefix
    raw, kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(256 << 20, device=dev, dtype=torch.uint8), "NHD")
    w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
    r = {"fi": timeit(lambda i: w.run(q, raw[i]), len(raw))}
    for tag, ext in EXT.items():
        qq = X.proj_qk(q, BLK[tag]) if BLK[tag] else q
        r[tag] = timeit(lambda i: ext.prefill(qq, *kv[tag][i], qo, indptr, indices, lens, q_len, kv_len,
                                             1 / math.sqrt(D), 0), len(raw))
    show("prefill", f"hq={hq} hkv={hkv} b={batch} q={q_len} prefix={prefix}", r)


def bench_decode(batch, kv_len, hq, hkv):
    raw, kv, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch, hq, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    r = {}
    for name, tc in (("fi", False), ("fi_tc", True)):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
        w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, q_data_type=torch.bfloat16)
        r[name] = timeit(lambda i: w.run(q, raw[i]), len(raw))
    for tag, ext in EXT.items():
        qq = X.proj_qk(q, BLK[tag]) if BLK[tag] else q
        rows = batch * -(-kv_len // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, hq, D, device=dev)
        ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device=dev)
        cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device=dev)
        r[tag] = timeit(lambda i: ext.decode(qq, *kv[tag][i], indptr, indices, lens, ws_o, ws_ml, cnt, kv_len,
                                            1 / math.sqrt(D)), len(raw))
    show("decode ", f"hq={hq} hkv={hkv} b={batch} kv={kv_len}", r)


warm_gpu()
HQ, HKV = 32, 8
for batch, ql, pre in [(1, 512, 0), (1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (32, 512, 0), (1, 128, 4096), (1, 128, 16384)]:
    bench_prefill(batch, ql, pre, HQ, HKV)
    torch.cuda.empty_cache()
for batch, kvl in [(1, 4096), (1, 32768), (8, 2048), (64, 1024)]:
    bench_decode(batch, kvl, HQ, HKV)
    torch.cuda.empty_cache()
