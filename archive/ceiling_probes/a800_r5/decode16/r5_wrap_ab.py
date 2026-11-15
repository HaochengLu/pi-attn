"""A/B of two builds of the attention extension on decode (us per layer): the default build against the one built
with PI_ATTN_SUFFIX=$1 (and PI_ATTN_CFLAGS=$2). usage: r5_wrap_ab.py <suffix> [cflags] ; PI_R5_VARS=3,1"""
import math, os, subprocess, sys
sys.path.insert(0, os.getcwd())
import torch
import bench_attention as B
from build import build
A = B.ext
os.environ["PI_ATTN_SUFFIX"] = sys.argv[1]
os.environ["PI_ATTN_CFLAGS"] = sys.argv[2] if len(sys.argv) > 2 else ""
Bx = build()
dev, D, HQ, HKV, CAP = "cuda", 128, 16, 8, 8192
VARS = [int(v) for v in os.environ.get("PI_R5_VARS", "3,1").split(",")]
VARS_B = [int(v) for v in os.environ.get("PI_R5_VARS_B", os.environ.get("PI_R5_VARS", "3,1")).split(",")]
B.warm_gpu()
GRID = os.environ.get("PI_R5_GRID", "1:320,512,1024,4200;2:192,512;4:192,640;8:192,512,640;13:192,512")
for item in GRID.split(";"):
    bs, kvs = item.split(":")
    for batch in map(int, bs.split(",")):
        for kv in map(int, kvs.split(",")):
            kvc, indptr, indices, last, lens = B.make_cache(batch, kv, HKV)
            q = torch.randn(batch, HQ, D, device=dev, dtype=torch.bfloat16)
            rows = batch * -(-CAP // A.GROUP_KEYS)
            ws_o = torch.full((rows, HQ, D), float("nan"), device=dev)
            ws_ml = torch.full((rows, HQ, 2 * A.DECODE_WARPS), float("nan"), device=dev)
            cnt = torch.zeros(rows, HQ, A.DECODE_WARPS, dtype=torch.int32, device=dev)
            def pi(e, var):
                e.set_decode_variant(var)
                t = B.timeit(lambda i: e.decode(q, *kvc[i], indptr, indices, lens, ws_o, ws_ml, cnt, CAP, 1 / math.sqrt(D), kv), len(kvc))
                e.set_decode_variant(0)
                return t
            r = {}
            for _ in range(3):
                for v in VARS: r.setdefault(f"A.v{v}", []).append(pi(A, v))
                for v in VARS_B: r.setdefault(f"B.v{v}", []).append(pi(Bx, v))
            m = {k: sorted(v)[1] for k, v in r.items()}
            print(f"AB b={batch:2d} kv={kv:5d}: " + " ".join(f"{k} {v:6.1f}" for k, v in m.items()), flush=True)
