"""Decode attention, us per layer (28-layer KV ring in one CUDA graph): FlashInfer's decode kernel, the
per-tile PI kernel (variant 1) and the two-phase kernel with 2 / 3 / 4 ring slots (variants 2 / 3 / 4).
usage (from kernels/): dec_bench.py [quick]"""
import os, sys
sys.path.insert(0, os.getcwd())
import bench_attention as B
B.warm_gpu()
quick = len(sys.argv) > 1
VARS = [int(v) for v in os.environ.get("PI_DEC_VARIANTS", "1,2").split(",")]
for batch in ((1, 8, 32) if quick else (1, 4, 8, 16, 32, 64)):
    for kv in ((96, 640, 2176) if quick else (96, 320, 640, 1280, 2176, 4200)):
        r = {}
        for v in VARS:
            B.ext.set_decode_variant(v)
            x = B.bench_decode(batch, kv, 16, 8)
            r[v] = x["pi"]; fi = x["fi"]
        B.ext.set_decode_variant(0)
        print(f"DEC b={batch:2d} kv={kv:4d} us/layer: fi {fi:6.1f} | " + " | ".join(f"v{v} {t:6.1f}" for v, t in r.items())
              + f" | best two-phase / per-tile {min(t for v, t in r.items() if v != 1) / r[1]:.2f}", flush=True)
