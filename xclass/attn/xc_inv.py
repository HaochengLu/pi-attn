"""Partition invariance of the XC programs (adapted from kernels/test_invariance.py): batch, chunk + prefix,
split / kernel variant, prefill row vs decode. Prints failures and the hash of the reference outputs."""
import math
import random
import sys

import torch

import xc_lib as X

dev = "cuda"
PAGE, D = X.PAGE, X.D
HQ, HKV = 16, 8
SCALE = 1 / math.sqrt(D)
COUNTERS = torch.zeros(1 << 20, dtype=torch.int32, device=dev)
same = lambda a, b: torch.equal(a.view(torch.int16), b.view(torch.int16))


class Pool:
    def __init__(self, pages=4096):
        nan = float("nan")
        self.k = torch.full((pages, PAGE, HKV, D), nan, device=dev, dtype=torch.bfloat16)
        self.v = torch.full((pages, PAGE, HKV, D), nan, device=dev, dtype=torch.bfloat16)
        self.sv = torch.full((pages, PAGE, HKV), nan, device=dev, dtype=torch.float32)
        self.free = list(range(1, pages))
        random.shuffle(self.free)

    def alloc(self, n_tokens):
        n = (n_tokens + PAGE - 1) // PAGE
        ids, self.free = self.free[:n], self.free[n:]
        return ids

    def write(self, pages, k, v, sv):
        idx = torch.tensor(pages, device=dev).repeat_interleave(PAGE)[: k.shape[0]]
        off = torch.arange(k.shape[0], device=dev) % PAGE
        self.k[idx, off], self.v[idx, off], self.sv[idx, off] = k, v, sv


def meta(page_lists, kv_lens, q_lens=None):
    indptr = [0]
    for p in page_lists:
        indptr.append(indptr[-1] + len(p))
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
    out = [t(indptr), t([i for p in page_lists for i in p]), t(kv_lens)]
    if q_lens is not None:
        qo = [0]
        for n in q_lens:
            qo.append(qo[-1] + n)
        out.append(t(qo))
    return out


def run(ext, tag, blk):
    random.seed(0)
    pool, fails = Pool(), []
    lens = [1, 15, 63, 64, 65, 200, 517, 1300, 2049]
    seqs, pages = {}, {}
    for i, L in enumerate(lens):
        q, k, v = X.make_data("outlier" if i % 2 else "sharp", L, HQ, HKV, seed=50 + i)
        qh, kh, vt, sv = X.project(q, k, v, blk)
        seqs[L] = tuple(t.to(dev) for t in (qh, kh, vt, sv))
        pages[L] = pool.alloc(L)
        pool.write(pages[L], *seqs[L][1:])

    def prefill(reqs, splits=0, variant=0):
        q = torch.cat([r[0] for r in reqs]).contiguous()
        indptr, indices, kvl, qo = meta([r[1] for r in reqs], [r[2] for r in reqs], [r[0].shape[0] for r in reqs])
        out = ext.prefill(q, pool.k, pool.v, pool.sv, qo, indptr, indices, kvl, max(r[0].shape[0] for r in reqs),
                          max(r[2] for r in reqs), SCALE, splits, variant)
        return list(out.split([r[0].shape[0] for r in reqs]))

    def decode(reqs):
        q = torch.cat([r[0] for r in reqs]).contiguous()
        indptr, indices, kvl = meta([r[1] for r in reqs], [r[2] for r in reqs])
        max_kv = max(r[2] for r in reqs)
        rows = len(reqs) * -(-max_kv // ext.GROUP_KEYS)
        ws_o = torch.full((rows, HQ, D), float("nan"), device=dev)
        ws_ml = torch.full((rows, HQ, 2 * ext.DECODE_WARPS), float("nan"), device=dev)
        out = ext.decode(q, pool.k, pool.v, pool.sv, indptr, indices, kvl, ws_o, ws_ml, COUNTERS, max_kv, SCALE)
        assert COUNTERS.count_nonzero().item() == 0
        return list(out.split(1))

    alone = {L: prefill([(seqs[L][0], pages[L], L)])[0] for L in lens}
    assert all(bool(torch.isfinite(a.float()).all()) for a in alone.values())
    print(f"INV {tag} reference hash {X.sha(torch.cat([alone[L] for L in lens]))}", flush=True)
    n = {"batch": 0, "chunk": 0, "split": 0, "path": 0}
    order = lens[:]
    random.shuffle(order)
    for L, out in zip(order, prefill([(seqs[L][0], pages[L], L) for L in order])):
        n["batch"] += 1
        if not same(out, alone[L]):
            fails.append(("batch", L))
    for L in lens:
        if L < 3:
            continue
        p2 = pool.alloc(L)
        pool.write(p2, *seqs[L][1:])
        cuts = sorted(random.sample(range(1, L), min(4, L - 1)))
        outs, start = [], 0
        for end in cuts + [L]:
            outs.append(prefill([(seqs[L][0][start:end].contiguous(), p2, end)])[0])
            start = end
        n["chunk"] += 1
        if not same(torch.cat(outs), alone[L]):
            fails.append(("chunk+prefix", L))
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    variants = [v for v in (1, 2, 3, 4) if ext.prefill_smem_bytes(v) <= smem]
    for var in variants:
        for sp in (1, 2, 3, 7):
            for L in lens:
                n["split"] += 1
                if not same(prefill([(seqs[L][0], pages[L], L)], sp, var)[0], alone[L]):
                    fails.append(("split", var, L, sp))
    for _ in range(2):
        probes = [(L, p) for L in lens for p in range(L)]
        random.shuffle(probes)
        i = 0
        while i < len(probes):
            m = random.randint(1, 64)
            group, i = probes[i:i + m], i + m
            reqs = [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in group]
            for (L, p), out in zip(group, decode(reqs)):
                n["path"] += 1
                if not same(out[0], alone[L][p]):
                    fails.append(("path", L, p))
        for L, p in probes[:8]:
            n["path"] += 1
            if not same(decode([(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1)])[0][0], alone[L][p]):
                fails.append(("path alone", L, p))
    print(f"INV {tag} variants {variants} checks {n} failed {len(fails)} {fails[:8]}", flush=True)
    return fails


if __name__ == "__main__":
    bad = 0
    for blk in (128, 32, 16):
        bad += len(run(X.build(1, blk), f"xc{blk}", blk))
    sys.exit(1 if bad else 0)
