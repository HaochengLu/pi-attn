"""Variant under test (PI_NEW, default 6) vs the shipped sm_90 prefill (variant 5) and the mma.sync kernel (1): bits."""
import os, sys, random
sys.path.insert(0, os.getcwd())
import torch
import test_invariance as T
NEWS = [int(x) for x in os.environ.get("PI_NEW", "6").split(",")]
NEW = NEWS[0]
pool = T.Pool()
bad = 0
for L in [1, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 511, 512, 513, 517, 1024, 1300, 2049, 3000]:
    q, k, v = T.seq(L)
    pg = pool.alloc(L)
    pool.write(pg, 0, k, v)
    T.KVSPLITS, T.VARIANT = 1, 1
    ref = T.prefill(pool, [(q, pg, L)])[0]
    for var in [5] + NEWS:
        for sp in (1, 2, 3, 7, 0):
            T.KVSPLITS, T.VARIANT = sp, var
            out = T.prefill(pool, [(q, pg, L)])[0]
            if not T.same(out, ref):
                bad += 1
                print("MISMATCH", L, var, sp, (out.view(torch.int16) != ref.view(torch.int16)).sum().item(), flush=True)
    # chunked with prefix, new variant
    T.KVSPLITS, T.VARIANT = 0, NEW
    if L >= 3:
        cuts = sorted(random.sample(range(1, L), min(4, L - 1)))
        outs, s = [], 0
        for e in cuts + [L]:
            outs.append(T.prefill(pool, [(q[s:e].contiguous(), pg, e)])[0])
            s = e
        if not T.same(torch.cat(outs), ref):
            bad += 1
            print("MISMATCH chunk", L, cuts, flush=True)
    print("len", L, "done", flush=True)
print("QUICK mismatches:", bad, flush=True)
