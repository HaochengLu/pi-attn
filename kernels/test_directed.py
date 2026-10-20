"""Directed edge cases (from the phase-2 review): signed zeros in V, ftz underflow
with huge scores, max ties, page sizes 1/64, GQA group 16, plus input validation."""
import random
import sys

import torch

import test_invariance as T

dev = "cuda"


def run(name, hq, hkv, L, qscale, vfn=None, page=16, splits=(0, 1, 3, 16)):
    T.HQ, T.HKV, T.PAGE = hq, hkv, page
    torch.manual_seed(1)
    random.seed(1)
    pool = T.Pool(pages=(L // page + 2) * 3 + 8)
    q = (torch.randn(L, hq, T.D, device=dev) * qscale).bfloat16()
    k = torch.randn(L, hkv, T.D, device=dev).bfloat16()
    v = torch.randn(L, hkv, T.D, device=dev).bfloat16()
    if vfn:
        v = vfn(v)
    pg = pool.alloc(L)
    pool.write(pg, 0, k, v)
    T.WARPS, T.KVSPLITS = 1, 1
    ref = T.prefill(pool, [(q, pg, L)])[0]
    fails = []
    for cfg, kvs in [(2, 1), (3, 1), (2, 3), (1, 0)]:
        T.WARPS, T.KVSPLITS = cfg, kvs
        if not T.same(T.prefill(pool, [(q, pg, L)])[0], ref):
            fails.append(("cfg", cfg, kvs))
    T.WARPS, T.KVSPLITS = 0, 0
    p2 = pool.alloc(L)
    pool.write(p2, 0, k, v)
    cuts = sorted(random.sample(range(1, L), min(3, L - 1)))
    outs, s = [], 0
    for e in cuts + [L]:
        outs.append(T.prefill(pool, [(q[s:e].contiguous(), p2, e)])[0])
        s = e
    if not T.same(torch.cat(outs), ref):
        fails.append("chunked")
    probes = [p for p in sorted({0, L // 2, L - 1, 65, 127, 200, 2047, 2048}) if p < L]
    for dvar in (1, 2, 4):
        T.DVAR = dvar
        for sp in splits:
            outs = T.decode(pool, [(q[p:p + 1].contiguous(), pg, p + 1) for p in probes], sp)
            for p, o in zip(probes, outs):
                if not T.same(o[0], ref[p]):
                    fails.append(("dec", dvar, sp, p))
    f = ref.float()
    bad = torch.isnan(f).any().item() or ((f == 0) & torch.signbit(f)).any().item()
    print(f"{name}: fails={len(fails)} {fails[:4]} nan_or_negzero={bad}", flush=True)
    T.PAGE = 16
    return len(fails) + int(bad)


def vmix(v):
    v = v.clone()
    v[:, :, :64] = -0.0
    v[:, :, 64:] = -v[:, :, 64:].abs()
    return v


def main():
    n = 0
    n += run("baseline", 16, 8, 777, 1.0)
    n += run("V all -0", 16, 8, 777, 1.0, vfn=lambda v: torch.full_like(v, -0.0))
    n += run("V half -0 half negative", 16, 8, 777, 1.0, vfn=vmix)
    n += run("huge scale 40 (ftz/alpha=0)", 16, 8, 777, 40.0)
    n += run("huge scale 40, L=4500", 16, 8, 4500, 40.0)
    n += run("zero Q (max ties)", 16, 8, 777, 0.0)
    n += run("page 64", 16, 8, 777, 1.0, page=64)
    n += run("page 1", 16, 8, 300, 1.0, page=1)
    n += run("hq=16 hkv=1 (group 16)", 16, 1, 300, 1.0)
    # Validation: group > 16 must be rejected, not silently wrong.
    try:
        T.HQ, T.HKV = 32, 1
        pool = T.Pool(pages=64)
        T.prefill(pool, [(torch.randn(5, 32, T.D, device=dev).bfloat16(), pool.alloc(5), 5)])
        print("group 32: NOT rejected")
        n += 1
    except RuntimeError as e:
        print("group 32 rejected:", str(e).splitlines()[0][:80])
    print("DIRECTED FAILURES:", n)
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
