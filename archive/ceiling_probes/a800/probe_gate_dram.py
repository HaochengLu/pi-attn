"""probe_gate.py in the engine's memory regime: LAYERS caches of the shape (one per layer,
together far beyond L2), one graph replay = LAYERS decode calls, as one decode step of a
28-layer model. Per-call times in us: old / deep / rule as in probe_gate.py."""
import math, sys
import torch
from build import build
from bench_attention import make_cache, warm_gpu, D

ext = build()
GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
BATCHES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (1, 2, 4, 8, 12)
KVS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else (512, 640, 1024, 2048, 4096)
LAYERS = 28


def graph_time(fns, reps=10):
    torch.cuda.synchronize()
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return round(a.elapsed_time(b) * 1000 / reps / len(fns), 1)


warm_gpu()
hq, hkv = 16, 8
for batch in BATCHES:
    for kv in KVS:
        caches = [make_cache(batch, kv, hkv) for _ in range(LAYERS)]
        q = torch.randn(batch, hq, D, device="cuda", dtype=torch.bfloat16)
        rows = batch * -(-GRID // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, hq, D, device="cuda")
        ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device="cuda")
        cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device="cuda")
        r = {}
        for name, d in (("old", 0), ("deep", 1 << 20), ("rule", -1)):
            fns = [lambda c=c, d=d: ext.decode(q, c[0], c[1], c[2], c[3], c[5], ws_o, ws_ml, cnt, GRID,
                                               1 / math.sqrt(D), d) for c in caches]
            r[name] = graph_time(fns)
        mb = 2 * batch * kv * hkv * D * 2 / 1e6
        print(f"b={batch:2d} kv={kv:5d} tiles={batch * -(-kv // 32) * hkv:5d} KV/layer {mb:5.1f} MB: old {r['old']:6.1f} "
              f"deep {r['deep']:6.1f} rule {r['rule']:6.1f}", flush=True)
        del caches; torch.cuda.empty_cache()
