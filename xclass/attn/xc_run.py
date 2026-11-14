"""Cross-class identity hashes + accuracy vs float64 for stock PI-Attn, FlashInfer and the XC programs.
usage: python xc_run.py [quick]"""
import math
import sys

import flashinfer
import torch

import xc_lib as X

dev = "cuda"
mode = sys.argv[1] if len(sys.argv) > 1 else "all"
quick = mode == "quick"
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(), flush=True)
EXT = {"stock": X.build(0), "xc128": X.build(1, 128), "xc32": X.build(1, 32), "xc16": X.build(1, 16)}
BLK = {"xc128": 128, "xc32": 32, "xc16": 16}
HQ, HKV = 8, 2
CASES = [("gauss", 512), ("gauss", 4096), ("gauss", 16384), ("outlier", 512), ("outlier", 4096), ("outlier", 16384),
         ("sharp", 4096), ("peaky", 4096), ("peaky", 16384),
         ("adv_wide", 2049), ("adv_cancel", 2049), ("adv_huge", 2049), ("adv_scaled", 2049), ("adv_const", 2049)]
if mode == "real":
    CASES = [c for c in CASES if not c[0].startswith("adv")]
if mode == "adv":
    CASES = [c for c in CASES if c[0].startswith("adv")]
if quick:
    CASES = [("gauss", 512), ("outlier", 517), ("peaky", 1300), ("adv_wide", 517), ("adv_const", 517), ("adv_huge", 517)]


def err(out, ref):
    e = (out.double() - ref)
    fin = torch.isfinite(e)
    if not bool(fin.all()):
        return f"nonfinite({int((~fin).sum())})"
    return f"rel {float(e.norm() / ref.norm()):.3e} mean {float(e.abs().mean()):.3e} max {float(e.abs().max()):.3e}"


for name, L in CASES:
    q, k, v = X.make_data(name, L, HQ, HKV)
    print(f"CASE {name} L={L} inputs {X.sha(q)} {X.sha(k)} {X.sha(v)}", flush=True)
    ref = X.exact64(q, k, v)
    ones = torch.ones(L, HKV)
    print(f"  ACC {name} L={L} bf16(exact)     {err(ref.bfloat16(), ref)}")
    try:
        fi = flashinfer.single_prefill_with_kv_cache(q.to(dev), k.to(dev), v.to(dev), causal=True)
        print(f"  ACC {name} L={L} flashinfer      {err(fi, ref)}")
        print(f"  HASH {name} L={L} flashinfer {X.sha(fi)}")
    except Exception as e:  # noqa: BLE001
        print("  flashinfer failed", type(e).__name__, e)
    out = X.prefill_full(EXT["stock"], q, k, v, ones)
    print(f"  ACC {name} L={L} pi-stock        {err(out, ref)}")
    print(f"  HASH {name} L={L} pi-stock {X.sha(out)}")
    for tag, blk in BLK.items():
        qh, kh, vt, sv = X.project(q, k, v, blk)
        vh = vt.double() * sv[..., None].double()
        print(f"  HASH {name} L={L} {tag}-inputs {X.sha(qh)} {X.sha(kh)} {X.sha(vt)} {X.sha(sv)}")
        out = X.prefill_full(EXT[tag], qh, kh, vt, sv)
        print(f"  ACC {name} L={L} {tag:5s}           {err(out, ref)}")
        print(f"  HASH {name} L={L} {tag} {X.sha(out)}")
        if L <= 4096:
            refp = X.exact64(qh, kh, vh)
            print(f"  ACC {name} L={L} {tag:5s} fp64 of projected: all {err(refp, ref)} | QK only {err(X.exact64(qh, kh, v), ref)}"
                  f" | V only {err(X.exact64(q, k, vh), ref)} | kernel vs its own fp64 {err(out, refp)}")
        # the stock chained program on the projected inputs (same function up to fp32 rounding): is it
        # the exact partials or just the projection that gives cross-class identity?
        o2 = X.prefill_full(EXT["stock"], qh, kh, vh.bfloat16(), ones)
        print(f"  HASH {name} L={L} stock-on-{tag}-inputs {X.sha(o2)}")
        if not name.startswith("adv"):  # static per-channel power-of-two gauge of V (calibrated on this data)
            vg, c = X.gauge_v(v)
            vt, sv = X.proj_v(vg)
            og = X.prefill_full(EXT[tag], qh, kh, vt, sv).cpu().double()
            og = torch.ldexp(og, c.repeat_interleave(HQ // HKV, 0)[None]).to(dev)
            print(f"  ACC {name} L={L} {tag}+vgauge    {err(og, ref)}")
            print(f"  HASH {name} L={L} {tag}+vgauge {X.sha(og.bfloat16())}")
    sys.stdout.flush()
    del ref
    torch.cuda.empty_cache()
