"""Exhaustive invariance matrix for the PI-Attn kernels.

For each GQA geometry (Qwen3 0.6B/1.7B: 16/8, 4B/8B: 32/8, plus 8/8 and 32/4)
and query scale (normal and peaked), every way of computing a token's attention
output must give the same bits: prefill with every kernel variant (Q in smem or in
registers, 4 or 8 warps per CTA) and 1, 2, 5 or the planner's split count over fold groups (inside random
batches), chunked prefill at random cuts over a
cached prefix stored in other pages, and decode inside random batches. Also
checks run-to-run determinism and accuracy vs float64, which must be within
test_invariance.ACC_SLACK of FlashInfer's (test_invariance.accuracy).
"""
import math
import random
import sys

import torch

import test_invariance as T

ext = T.ext
dev = "cuda"


def run_geometry(hq, hkv, qscale, lens, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    T.HQ, T.HKV = hq, hkv
    pool = T.Pool(pages=sum((L + 15) // 16 for L in lens) * 2 + 64)
    seqs, pages = {}, {}
    for L in lens:
        seqs[L] = (torch.randn(L, hq, T.D, device=dev).bfloat16() * qscale,
                   torch.randn(L, hkv, T.D, device=dev).bfloat16(),
                   torch.randn(L, hkv, T.D, device=dev).bfloat16())
        pages[L] = pool.alloc(L)
        pool.write(pages[L], 0, seqs[L][1], seqs[L][2])
    fails = []
    T.KVSPLITS = 1
    ref = {L: T.prefill(pool, [(seqs[L][0], pages[L], L)])[0] for L in lens}
    for var, kvs in [(v, k) for v in (1, 2, 3, 4) for k in (1, 2, 5, 0)]:  # every variant; forced splits and the planner
        T.VARIANT, T.KVSPLITS = var, kvs
        # Split mode stores one f32 partial per (row, head, fold group): batch only the
        # requests whose workspace fits; longer ones get the extend case split mode is
        # for (their last 128 tokens over the cached prefix).
        fits = lambda L: kvs in (0, 1) or L * hq * -(-L // ext.GROUP_KEYS) * 520 <= 2 << 30
        order = [L for L in lens if fits(L)]
        random.shuffle(order)
        outs = T.prefill(pool, [(seqs[L][0], pages[L], L) for L in order]) if order else []
        for L, out in zip(order, outs):
            if not T.same(out, ref[L]):
                fails.append(("prefill-split", var, kvs, L))
        for L in [L for L in lens if not fits(L)]:
            out = T.prefill(pool, [(seqs[L][0][L - 128:].contiguous(), pages[L], L)])[0]
            if not T.same(out, ref[L][L - 128:]):
                fails.append(("extend-split", var, kvs, L))
        for L in lens:  # chunked prefill over a cached prefix in other pages
            if L < 2 or not fits(L):
                continue
            p2 = pool.alloc(L)
            pool.write(p2, 0, seqs[L][1], seqs[L][2])
            cuts = sorted(random.sample(range(1, L), min(3, L - 1)))
            outs, start = [], 0
            for end in cuts + [L]:
                outs.append(T.prefill(pool, [(seqs[L][0][start:end].contiguous(), p2, end)])[0])
                start = end
            pool.free.extend(p2)
            if not T.same(torch.cat(outs), ref[L]):
                fails.append(("chunked", var, kvs, L))
    T.VARIANT, T.KVSPLITS = 0, 0
    for _ in range(2):  # decode in random batches vs prefill rows
        probes = [(L, p) for L in lens for p in {0, L - 1, random.randrange(L)}]
        random.shuffle(probes)
        for i in range(0, len(probes), 7):
            grp = probes[i:i + 7]
            outs = T.decode(pool, [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in grp])
            for (L, p), out in zip(grp, outs):
                if not T.same(out[0], ref[L][p]):
                    fails.append(("decode", L, p))
    # run-to-run
    again = {L: T.prefill(pool, [(seqs[L][0], pages[L], L)])[0] for L in lens}
    for L in lens:
        if not T.same(again[L], ref[L]):
            fails.append(("rerun", L))
    small = [l for l in lens if l <= 4096]
    if not small:  # float64 reference is quadratic in memory; skip for long contexts
        return fails, float("nan"), float("nan")
    L = max(small)
    q, k, v = seqs[L]
    acc, mean, mx = T.accuracy(L, ref[L], q, k, v)
    return fails + acc, mean, mx


def main():
    lens = [1, 2, 63, 64, 65, 127, 128, 129, 777, 2047, 2048, 2049, 4096, 6200]
    total = []
    for hq, hkv in [(16, 8), (32, 8), (8, 8), (32, 4)]:
        for qscale in [1.0, 6.0]:
            f, em, ex = run_geometry(hq, hkv, qscale, lens, seed=hq * 100 + hkv + int(qscale))
            total += f
            print(f"hq={hq} hkv={hkv} qscale={qscale}: failures {len(f)} | err vs fp64 mean {em:.2e} max {ex:.2e} {f[:3]}", flush=True)
    # one long-context geometry
    f, em, ex = run_geometry(16, 8, 1.0, [32768, 20001], seed=7)
    total += f
    print(f"long ctx 32768/20001: failures {len(f)} {f[:3]}", flush=True)
    print("TOTAL FAILURES:", len(total))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
