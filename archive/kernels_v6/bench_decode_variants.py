"""Decode attention at long context, with and without shared prefix pages: FlashInfer
(default split-KV) vs every PI decode variant (1-5 planner splits, 6-9 persistent)
and PI grouped mode.
Times in us (median). Usage: python bench_decode_variants.py [--out file.json]"""
import argparse
import json
import math

import flashinfer
import torch

from bench_attention import D, PAGE, C, ext, timeit

dev = "cuda"
# Decode smem per variant: warps x tiles x one 64 x 136 bf16 tile.
TILE = 64 * 136 * 2
SMEM = {1: TILE, 2: 2 * TILE, 3: 4 * TILE, 4: 4 * TILE, 5: 8 * TILE, 6: TILE, 7: 2 * TILE, 8: 4 * TILE, 9: 8 * TILE}


def cache(batch, kv_len, hkv, shared):
    pages_per = (kv_len + PAGE - 1) // PAGE
    n = (1 if shared else batch) * pages_per + 1
    k = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(n, PAGE, hkv, D, device=dev, dtype=torch.bfloat16)
    one = torch.randperm(n - 1, device=dev).int() + 1
    indices = (one[:pages_per].repeat(batch) if shared else one[: batch * pages_per]).contiguous()
    indptr = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * pages_per
    last = torch.full((batch,), (kv_len - 1) % PAGE + 1, device=dev, dtype=torch.int32)
    lens = torch.full((batch,), kv_len, device=dev, dtype=torch.int32)
    return k, v, indptr, indices, last, lens


def bench(batch, kv_len, hq, hkv, shared):
    k, v, indptr, indices, last, lens = cache(batch, kv_len, hkv, shared)
    q = torch.randn(batch, hq, D, device=dev, dtype=torch.bfloat16)
    ws = torch.empty(256 << 20, device=dev, dtype=torch.uint8)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
    w.plan(indptr.cpu(), indices, last.cpu(), hq, hkv, D, PAGE, q_data_type=torch.bfloat16)
    res = {"batch": batch, "kv_len": kv_len, "hq": hq, "hkv": hkv, "shared": shared,
           "fi": timeit(lambda: w.run(q, (k, v)))}
    max_chunks = (kv_len + C - 1) // C
    full = batch * max_chunks
    grouped = batch * ((max_chunks + 31) // 32)
    for cap, tag in [(full, ""), (grouped, "_grouped")]:
        ws_o = torch.empty(cap, hq, D, device=dev)
        ws_ml = torch.empty(cap, hq, 2, device=dev)
        ref = None
        for var in range(1, 10):
            if SMEM[var] > torch.cuda.get_device_properties(0).shared_memory_per_block_optin:
                continue  # 4 warps x 2 tiles: big-smem GPUs only
            run = lambda: ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, max_chunks, 0, 1 / math.sqrt(D), var)
            out = run()
            ref = out if ref is None else ref
            assert torch.equal(out, ref), (var, tag)  # every variant gives the same bits
            res[f"pi_v{var}{tag}"] = timeit(run)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for hq, hkv in [(16, 8), (32, 8)]:
        for batch, kv_len, shared in [(1, 1024, False), (8, 1024, False), (32, 1024, False),
                                      (1, 32768, False), (2, 16384, False), (8, 4096, False), (8, 4096, True),
                                      (4, 8192, True), (8, 16384, False), (32, 4096, True), (32, 8192, False)]:
            r = bench(batch, kv_len, hq, hkv, shared)
            rows.append(r)
            print(json.dumps(r), flush=True)
    if a.out:
        json.dump({"device": torch.cuda.get_device_name(0), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
