"""Attention latency with fixed planners (no best-of): PI-Attn vs FlashInfer 0.6.18.

Decode baselines: FI default (nondeterministic split-KV), FI tensor-core decode, and
FI tensor-core decode with fixed_split_size=2048 (SGLang deterministic decode). Prefill: FI paged prefill, FI with
fixed_split_size=2048 (deterministic), PI planner. Times in us (median)."""
import argparse
import json
import math

import flashinfer
import torch

from build import build

ext = build()
dev = "cuda"
PAGE, D = 16, 128


def timeit(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return round(ts[len(ts) // 2] * 1000, 1)


def make_cache(batch, kv_len, hkv, shared=False):
    """Paged K/V in shuffled pages; shared: every request reads the same pages (prefix cache)."""
    pages_per = (kv_len + PAGE - 1) // PAGE
    n = (1 if shared else batch) * pages_per + 1
    k = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(n - 1, device=dev).int() + 1
    indices = (perm.repeat(batch) if shared else perm[: batch * pages_per]).contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), (kv_len - 1) % PAGE + 1, device=dev, dtype=torch.int32)
    lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
    return k, v, indptr, indices, last, lens


def fi_try(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def best_fi(r):  # fastest non-deterministic FlashInfer decode (CUDA-core or tensor-core)
    return min(v for k, v in r.items() if k in ("fi", "fi_tc") and isinstance(v, (int, float)))


def bench_decode(batch, kv_len, hq, hkv, shared=False):
    k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv, shared)
    q = torch.randn(batch, hq, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    res = {}
    for name, kw, tc in [("fi", {}, False), ("fi_tc", {}, True), ("fi_det_fixed2048", {"fixed_split_size": 2048}, True)]:
        def run():
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
            w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, q_data_type=torch.bfloat16, **kw)
            return timeit(lambda: w.run(q, (k, v)))
        res[name] = fi_try(run)
    rows = batch * -(-kv_len // ext.GROUP_KEYS)
    ws_o = torch.empty(rows, hq, D, device=dev)
    ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device=dev)
    cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device=dev)
    res["pi"] = timeit(lambda: ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, kv_len, 1 / math.sqrt(D)))
    return res


def bench_prefill(batch, q_len, prefix, hq, hkv):
    kv_len = q_len + prefix
    k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    res = {}
    for name, kw in [("fi", {}), ("fi_det_fixed2048", {"fixed_split_size": 2048})]:
        def run():
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
            w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True,
                   q_data_type=torch.bfloat16, **kw)
            return timeit(lambda: w.run(q, (k, v)), iters=20)
        res[name] = fi_try(run)
    res["pi"] = timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, kv_len,
                                           1 / math.sqrt(D), 0), iters=20)
    return res


DECODE = [(1, 1024), (1, 8192), (1, 32768), (8, 2048), (8, 16384), (32, 1024), (32, 8192), (128, 1024), (128, 4096)]
DECODE_SHARED = [(8, 4096), (32, 4096)]  # all requests read the same prefix pages
PREFILL = [(1, 512, 0), (1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (32, 512, 0), (1, 512, 8192), (1, 128, 4096),
           (1, 128, 16384), (4, 2048, 2048)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for hq, hkv in [(16, 8), (32, 8)]:
        for batch, kv in DECODE:
            r = bench_decode(batch, kv, hq, hkv)
            rows.append({"kind": "decode", "hq": hq, "hkv": hkv, "batch": batch, "kv_len": kv, **r})
            print(f"decode  hq={hq:2d} b={batch:3d} kv={kv:5d}: " + " ".join(f"{k}={v}" for k, v in r.items())
                  + f" | pi/fi={r['pi'] / r['fi']:.2f} pi/best_fi={r['pi'] / best_fi(r):.2f}", flush=True)
        for batch, kv in DECODE_SHARED:
            r = bench_decode(batch, kv, hq, hkv, shared=True)
            rows.append({"kind": "decode_shared", "hq": hq, "hkv": hkv, "batch": batch, "kv_len": kv, **r})
            print(f"decode  hq={hq:2d} b={batch:3d} kv={kv:5d} shared: " + " ".join(f"{k}={v}" for k, v in r.items())
                  + f" | pi/fi={r['pi'] / r['fi']:.2f} pi/best_fi={r['pi'] / best_fi(r):.2f}", flush=True)
        for batch, ql, pre in PREFILL:
            r = bench_prefill(batch, ql, pre, hq, hkv)
            rows.append({"kind": "prefill", "hq": hq, "hkv": hkv, "batch": batch, "q_len": ql, "prefix": pre, **r})
            print(f"prefill hq={hq:2d} b={batch} q={ql:4d} prefix={pre:5d}: " + " ".join(f"{k}={v}" for k, v in r.items())
                  + f" | pi/fi={r['pi'] / r['fi']:.2f}", flush=True)
    if a.out:
        json.dump({"device": torch.cuda.get_device_name(0), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
