"""Attention kernel latency: PI-Attn vs FlashInfer 0.6.18 (paged, bf16, head_dim 128, page 16).

Decode: FlashInfer default (dynamic split-KV), fixed_split_size (SGLang-style
deterministic decode), disable_split_kv (vLLM-style num_splits=1), and
tensor-core decode. Prefill: FlashInfer paged prefill (causal)."""
import argparse
import json
import math

import flashinfer
import torch

from build import build

ext = build()
dev = "cuda"
PAGE, D = 16, 128
C = ext.CHUNK


def timeit(fn, iters=50, warmup=10):
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
    return ts[len(ts) // 2] * 1000  # us


def make_cache(batch, kv_len, hkv):
    pages_per = (kv_len + PAGE - 1) // PAGE
    n = batch * pages_per + 1
    k = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(n - 1, device=dev).int() + 1
    indices = perm[: batch * pages_per].contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), (kv_len - 1) % PAGE + 1, device=dev, dtype=torch.int32)
    lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
    return k, v, indptr, indices, last, lens


def bench_decode(batch, kv_len, hq, hkv):
    k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch, hq, D, device=dev, dtype=torch.bfloat16)
    scale = 1 / math.sqrt(D)
    res = {}
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    for name, kw, tc in [("fi_default", {}, False), ("fi_tc", {}, True),
                         ("fi_tc_fixed2048", {"fixed_split_size": 2048}, True),
                         ("fi_nosplit", {"disable_split_kv": True}, False)]:
        try:
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=tc)
            w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, q_data_type=torch.bfloat16, **kw)
            res[name] = timeit(lambda: w.run(q, (k, v)))
        except Exception as e:  # noqa: BLE001
            res[name] = f"err: {type(e).__name__}: {str(e)[:80]}"
    max_chunks = (kv_len + C - 1) // C
    ws_o = torch.empty(batch, hq, max_chunks, D, device=dev)
    ws_ml = torch.empty(batch, hq, max_chunks, 2, device=dev)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    best = None
    big = torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 139264
    for var in ((1, 2, 3, 4, 5) if big else (1, 2, 3, 4)):
        vb = None
        for splits in [1, 2, 4, 8, 16, 32, 64, 128]:
            if splits > max_chunks:
                break
            t = timeit(lambda: ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, splits, scale, var))
            vb = t if vb is None else min(vb, t)
        res[f"pi_v{var}"] = vb
        best = vb if best is None else min(best, vb)
    res["pi_best"] = best
    return res


def bench_prefill(batch, q_len, prefix, hq, hkv):
    kv_len = q_len + prefix
    k, v, indptr, indices, last, lens = make_cache(batch, kv_len, hkv)
    q = torch.randn(batch * q_len, hq, D, device=dev, dtype=torch.bfloat16)
    qo = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * q_len
    scale = 1 / math.sqrt(D)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(qo.cpu(), indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, causal=True, q_data_type=torch.bfloat16)
    res = {"fi": timeit(lambda: w.run(q, (k, v)), iters=20)}
    res["pi"] = timeit(lambda: ext.prefill(q, k, v, qo, indptr, indices, lens, q_len, scale, 0), iters=20)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for hq, hkv in [(16, 8), (32, 8)]:
        for batch, kv_len in [(1, 1024), (1, 8192), (1, 32768), (8, 2048), (8, 16384), (32, 1024),
                              (32, 8192), (128, 1024), (128, 4096)]:
            r = bench_decode(batch, kv_len, hq, hkv)
            rows.append({"kind": "decode", "hq": hq, "hkv": hkv, "batch": batch, "kv_len": kv_len, **r})
            fmt = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if True}
            print(json.dumps(rows[-1] | {}, default=str)[:0] + f"decode hq={hq} b={batch} kv={kv_len}: {fmt}", flush=True)
        for batch, q_len, prefix in [(1, 512, 0), (1, 2048, 0), (1, 8192, 0), (8, 1024, 0), (1, 512, 8192)]:
            r = bench_prefill(batch, q_len, prefix, hq, hkv)
            rows.append({"kind": "prefill", "hq": hq, "hkv": hkv, "batch": batch, "q_len": q_len, "prefix": prefix, **r})
            print(f"prefill hq={hq} b={batch} q={q_len} prefix={prefix}: fi {r['fi']:.1f}us pi {r['pi']:.1f}us "
                  f"ratio {r['pi'] / r['fi']:.2f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
