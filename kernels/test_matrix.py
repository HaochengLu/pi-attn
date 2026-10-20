"""Exhaustive invariance matrix for the PI-Attn kernels.

For each GQA geometry (Qwen3 0.6B/1.7B: 16/8, 4B/8B: 32/8, plus 8/8 and 32/4)
and query scale (normal and peaked), every way of computing a token's attention
output must give the same bits: prefill configs 1-3 (and 4 where smem allows),
chunked prefill at random cuts over a cached prefix stored in other pages,
decode variants 1-4 (5 where smem allows) x split counts, inside random
batches. Also checks run-to-run determinism and accuracy vs float64.
"""
import math
import random
import sys

import torch

import test_invariance as T

ext = T.ext
dev = "cuda"
big = torch.cuda.get_device_properties(0).shared_memory_per_block_optin >= 139264
PCFGS = (1, 2, 3, 4) if big else (1, 2, 3)
DVARS = (1, 2, 3, 4, 5) if big else (1, 2, 3, 4)


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
    T.WARPS = PCFGS[0]
    ref = {L: T.prefill(pool, [(seqs[L][0], pages[L], L)])[0] for L in lens}
    for cfg in PCFGS:  # prefill configs, mixed batch
        T.WARPS = cfg
        order = lens[:]
        random.shuffle(order)
        for L, out in zip(order, T.prefill(pool, [(seqs[L][0], pages[L], L) for L in order])):
            if not T.same(out, ref[L]):
                fails.append(("prefill-cfg", cfg, L))
        for L in lens:  # chunked prefill over a cached prefix in other pages
            if L < 2:
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
                fails.append(("chunked", cfg, L))
    T.WARPS = 0
    for dvar in DVARS:  # decode variants x splits vs prefill rows
        T.DVAR = dvar
        for splits in [1, 3, 16, 128]:
            probes = [(L, p) for L in lens for p in {0, L - 1, random.randrange(L)}]
            random.shuffle(probes)
            for i in range(0, len(probes), 7):
                grp = probes[i:i + 7]
                outs = T.decode(pool, [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in grp], splits)
                for (L, p), out in zip(grp, outs):
                    if not T.same(out[0], ref[L][p]):
                        fails.append(("decode", dvar, splits, L, p))
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
    e = (ref[L].float() - T.exact(q, k, v).float()).abs()
    return fails, e.mean().item(), e.max().item()


def main():
    lens = [1, 2, 63, 64, 65, 127, 128, 129, 777, 4096]
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
