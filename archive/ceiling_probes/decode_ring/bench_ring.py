"""Decode copy schedule probe: decode_kernel<stages> for stages 0 (one K and one V
buffer, the engine's schedule before this probe) and K/V rings of 2, 3 and 4 tile pairs.

Bits: every schedule must equal stages 0 on every configuration (also covered by
kernels/test_invariance.py). Time: median us per launch, two ways:
  hot : the same cache every launch (L2-resident for small caches; bench_attention.py's method)
  cold: a burst over up to 28 separate caches (fewer when they would exceed 3 GB), as
        consecutive layers of one decode step read them, so each launch's K/V comes
        from DRAM once the set exceeds L2.
The engine keeps stages 0; this directory holds the probe's kernel (pi_attn_kernels.cu, with
decode_kernel<kStages> and launch_decode) and its torch binding (pi_attn_torch.cu, decode with
a trailing `stages` argument). To rerun, copy them over src/kernels/pi/pi_attn_kernels.cu and
kernels/pi_attn_torch.cu. Result (RTX 4090 D, out.txt): every schedule is bitwise identical
and none is faster; rings of 3-4 pairs are up to 1.9x slower hot and 1.26x cold at batch
16-128, where the larger shared-memory footprint lowers occupancy.
Usage (from kernels/): python ../archive/ceiling_probes/decode_ring/bench_ring.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../kernels"))
from bench_attention import D, ext, make_cache, timeit  # noqa: E402

STAGES = (0, 2, 3, 4)


def run(batch, kv_len, hq, hkv):
    cache_bytes = 2 * batch * kv_len * hkv * D * 2
    layers = max(2, min(28, (3 << 30) // cache_bytes))
    caches = [make_cache(batch, kv_len, hkv) for _ in range(layers)]
    q = torch.randn(batch, hq, D, device="cuda", dtype=torch.bfloat16)
    rows = batch * -(-kv_len // ext.GROUP_KEYS)
    ws_o = torch.empty(rows, hq, D, device="cuda")
    ws_ml = torch.empty(rows, hq, 2 * ext.DECODE_WARPS, device="cuda")
    cnt = torch.zeros(rows, hq, ext.DECODE_WARPS, dtype=torch.int32, device="cuda")

    def call(c, s):
        k, v, indptr, indices, _, lens = c
        return ext.decode(q, k, v, indptr, indices, lens, ws_o, ws_ml, cnt, kv_len, 1 / math.sqrt(D), s)

    ref = call(caches[0], 0)
    bad = [s for s in STAGES if not torch.equal(call(caches[0], s).view(torch.int16), ref.view(torch.int16))]
    hot = {s: timeit(lambda: call(caches[0], s)) for s in STAGES}
    cold = {s: timeit(lambda: [call(c, s) for c in caches], iters=10, warmup=2) / layers for s in STAGES}
    return bad, hot, cold


def main():
    print("config | hot us: " + " ".join(f"s{s}" for s in STAGES) + " | cold us/layer: "
          + " ".join(f"s{s}" for s in STAGES) + " | bitwise")
    for hq, hkv in ((16, 8), (32, 8)):
        for batch in (1, 4, 16, 64, 128):
            for kv in (64, 128, 256, 512, 1024, 4096, 16384):
                if batch * kv > 128 * 4096:
                    continue
                bad, hot, cold = run(batch, kv, hq, hkv)
                print(f"hq{hq} b{batch:3d} kv{kv:5d} | " + " ".join(f"{hot[s]:7.1f}" for s in STAGES) + " | "
                      + " ".join(f"{cold[s]:7.1f}" for s in STAGES) + f" | {'OK' if not bad else 'MISMATCH ' + str(bad)}",
                      flush=True)


if __name__ == "__main__":
    main()
