"""decode_kernel K/V ring depth in the engine's memory regime (28 caches per graph replay, as
probe_gate_dram.py), 16 qo / 8 kv heads, grid for GRID keys; us per call. The shipped kernel
before the ring (one K and one V slot) measured in the same harness: probe_gate_dram.txt `old`."""
import math, sys
import torch
from build import build
from bench_attention import make_cache, warm_gpu, D
from probe_gate_dram import graph_time

ext = build()
GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BATCHES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else (1, 4, 8, 16, 32)
KVS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else (512, 640, 1024, 2048, 4096)
warm_gpu()
hq, hkv = 16, 8
for batch in BATCHES:
    for kv in KVS:
        caches = [make_cache(batch, kv, hkv) for _ in range(28)]
        q = torch.randn(batch, hq, D, device="cuda", dtype=torch.bfloat16)
        rows = batch * -(-GRID // ext.GROUP_KEYS)
        ws_o = torch.empty(rows, hq, D, device="cuda")
        ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device="cuda")
        cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device="cuda")
        r, outs = {}, {}
        for ring in (1, 2, 3, 4):
            fns = [lambda c=c, ring=ring: ext.decode(q, c[0], c[1], c[2], c[3], c[5], ws_o, ws_ml, cnt, GRID,
                                                     1 / math.sqrt(D), ring) for c in caches]
            outs[ring] = fns[0]().clone()
            r[ring] = graph_time(fns)
        same = all(torch.equal(outs[1], o) for o in outs.values())
        print(f"b={batch:2d} kv={kv:5d}: ring1 {r[1]:6.1f} ring2 {r[2]:6.1f} ring3 {r[3]:6.1f} ring4 {r[4]:6.1f} same={same}", flush=True)
        del caches; torch.cuda.empty_cache()
