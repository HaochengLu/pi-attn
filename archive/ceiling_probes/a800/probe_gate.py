"""Decode kernel choice on this GPU: decode_kernel only (deep_tiles=0) vs decode_deep_kernel
for every step (huge threshold) vs the device rule (-1, decode_deep_tiles), graph-replayed like
the engine, grid sized for GRID keys; 16 qo / 8 kv heads. tiles = batch * ceil(kv/32) * kv heads.
Times in us."""
import math, sys
import torch
from build import build
from bench_attention import make_cache, warm_gpu, D

ext = build()
GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 8192


def graph_time(fn, reps=200):
    torch.cuda.synchronize()  # calls share ws and counters: never overlap two
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps // 10):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return round(a.elapsed_time(b) * 1000 / reps, 1)


warm_gpu()
hq, hkv = 16, 8
BATCHES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (1, 2, 4, 8, 12)
KVS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else (256, 512, 640, 1024, 1536, 2048, 3072, 4096, 8192)
for batch in BATCHES:
    for kv in KVS:
        k, v, indptr, indices, last, lens = make_cache(batch, kv, hkv)
        q = torch.randn(batch, hq, D, device="cuda", dtype=torch.bfloat16)
        rows = batch * -(-GRID // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, hq, D, device="cuda")
        ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device="cuda")
        cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device="cuda")
        r, outs = {}, {}
        for name, d in (("old", 0), ("deep", 1 << 20), ("rule", -1)):
            f = lambda: ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, GRID, 1 / math.sqrt(D), d)
            outs[name] = f().clone()
            r[name] = graph_time(f)
        same = all(torch.equal(outs["old"], o) for o in outs.values())
        tiles = batch * -(-kv // 32) * hkv
        print(f"b={batch:2d} kv={kv:5d} tiles={tiles:5d}: old {r['old']:6.1f} deep {r['deep']:6.1f} rule {r['rule']:6.1f}"
              f" same={same}", flush=True)
