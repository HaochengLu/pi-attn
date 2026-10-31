"""GPU invariance and accuracy tests for the PI-Attn kernels.

Six partition changes must leave every output bit unchanged:
  batch    : a request computed alone vs inside a mixed batch
  split    : prefill split over fold groups (1..many CTAs per row tile), either kernel variant
  chunk    : prefill in one call vs chunked prefill with arbitrary cut points
  prefix   : prefix KV already cached (different physical pages) vs computed now
  path     : prefill row at position p vs decode of the token at position p
  heads    : all heads in one call vs the head shards of tensor-parallel ranks (t = 2, 3, 4,
             8 over the kv heads; t = 16 replicates kv heads, so each rank's GQA group shrinks)
Accuracy against float64 attention must be within ACC_SLACK of FlashInfer's single prefill (a
production bf16 flash-attention kernel, P rounded to bf16 as here) on the same inputs. (torch
SDPA on these 3-d inputs runs its math backend, which keeps P in fp32; it is not the reference.)
kernels/test_mutants.py runs these checks on deliberately broken kernels, which must fail them.
"""
import math
import random
import sys

import flashinfer
import torch

from build import build

ext = build()
dev = "cuda"
torch.manual_seed(0)
random.seed(0)
PAGE = 16
HQ, HKV, D = 16, 8, 128
SCALE = 1 / math.sqrt(D)
KVSPLITS = int(__import__('os').environ.get('PI_KVSPLITS', '1'))
# Prefill kernel variant: 0 = the device's choice, 1 = Q in smem, 2 = Q in registers, 3 / 4 = the same
# with 8 warps per CTA. VARIANTS: those whose shared memory this GPU allows (4 does not fit sm_89).
VARIANT = 0
VARIANTS = [v for v in (1, 2, 3, 4)
            if ext.prefill_smem_bytes(v) <= torch.cuda.get_device_properties(0).shared_memory_per_block_optin]
# PI error vs float64 may exceed FlashInfer's by at most this factor (mean, max).
ACC_SLACK = (1.05, 1.25)
# Decode arrival counters, shared by every call as in the engine: each call must
# leave them zero for the next.
COUNTERS = torch.zeros(1 << 20, dtype=torch.int32, device=dev)


class Pool:
    """Paged KV pool with a shuffled free list; stale pages hold NaN garbage."""

    def __init__(self, pages=4096):
        self.k = torch.full((pages, PAGE, HKV, D), float("nan"), device=dev, dtype=torch.bfloat16)
        self.v = torch.full((pages, PAGE, HKV, D), float("nan"), device=dev, dtype=torch.bfloat16)
        self.free = list(range(1, pages))
        random.shuffle(self.free)

    def alloc(self, n_tokens):
        n = (n_tokens + PAGE - 1) // PAGE
        ids, self.free = self.free[:n], self.free[n:]
        return ids

    def write(self, pages, start, k, v):
        for i in range(k.shape[0]):
            p = start + i
            self.k[pages[p // PAGE], p % PAGE] = k[i]
            self.v[pages[p // PAGE], p % PAGE] = v[i]


def seq(L):
    s = 3.0
    return (torch.randn(L, HQ, D, device=dev).bfloat16() * s,
            torch.randn(L, HKV, D, device=dev).bfloat16(),
            torch.randn(L, HKV, D, device=dev).bfloat16())


def meta(page_lists, kv_lens, q_lens=None):
    indptr = [0]
    for p in page_lists:
        indptr.append(indptr[-1] + len(p))
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
    out = dict(kv_page_indptr=t(indptr), kv_page_indices=t([i for p in page_lists for i in p]),
               kv_lens=t(kv_lens))
    if q_lens is not None:
        qo = [0]
        for n in q_lens:
            qo.append(qo[-1] + n)
        out["qo_indptr"] = t(qo)
    return out


def prefill(pool, reqs):
    """reqs: list of (q[q_len], pages, kv_len). Returns per-request outputs."""
    q = torch.cat([r[0] for r in reqs]).contiguous()
    m = meta([r[1] for r in reqs], [r[2] for r in reqs], [r[0].shape[0] for r in reqs])
    out = ext.prefill(q, pool.k, pool.v, m["qo_indptr"], m["kv_page_indptr"], m["kv_page_indices"],
                      m["kv_lens"], max(r[0].shape[0] for r in reqs), max(r[2] for r in reqs), SCALE,
                      KVSPLITS, VARIANT)
    return list(out.split([r[0].shape[0] for r in reqs]))


def decode(pool, reqs):
    """reqs: list of (q[1], pages, kv_len)."""
    q = torch.cat([r[0] for r in reqs]).contiguous()
    m = meta([r[1] for r in reqs], [r[2] for r in reqs])
    max_kv = max(r[2] for r in reqs)
    rows = len(reqs) * -(-max_kv // ext.GROUP_KEYS)
    hq = q.shape[1]
    ws_o = torch.full((rows, hq, D), float("nan"), device=dev)
    ws_ml = torch.full((rows, hq, 2 * ext.DECODE_WARPS), float("nan"), device=dev)
    out = ext.decode(q, pool.k, pool.v, m["kv_page_indptr"], m["kv_page_indices"], m["kv_lens"],
                     ws_o, ws_ml, COUNTERS, max_kv, SCALE)
    assert COUNTERS.count_nonzero().item() == 0, "decode left arrival counters set"
    return list(out.split(1))


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def head_shards(t):
    """(q heads, kv heads) of each of t ranks: kv heads split as evenly as possible when t <= HKV,
    else q heads split evenly and each rank holds a copy of its one kv head."""
    g = HQ // HKV
    if t <= HKV:
        cuts = [round(i * HKV / t) for i in range(t + 1)]
        return [((a * g, b * g), (a, b)) for a, b in zip(cuts, cuts[1:])]
    cuts = [round(i * HQ / t) for i in range(t + 1)]
    return [((a, b), (a // g, (b - 1) // g + 1)) for a, b in zip(cuts, cuts[1:])]


def check_heads(pool, seqs, pages, alone, fails):
    lens = list(seqs)
    probes = [(L, p) for L in lens for p in {0, L // 2, L - 1}]
    for t in (2, 3, 4, 8, 16):
        for (qa, qb), (ka, kb) in head_shards(t):
            assert (qb - qa) % (kb - ka) == 0 and qa // (HQ // HKV) >= ka
            part = Pool.__new__(Pool)
            part.k, part.v = pool.k[:, :, ka:kb].contiguous(), pool.v[:, :, ka:kb].contiguous()
            # q heads [qa, qb) attend kv heads [ka, kb) in the same GQA mapping as the full call
            # only when the slice keeps whole groups or is one group's sub-range (t = 16).
            outs = prefill(part, [(seqs[L][0][:, qa:qb].contiguous(), pages[L], L) for L in lens])
            for L, out in zip(lens, outs):
                if not same(out, alone[L][:, qa:qb]):
                    fails.append(("heads prefill", t, (qa, qb), L))
            reqs = [(seqs[L][0][p:p + 1, qa:qb].contiguous(), pages[L], p + 1) for L, p in probes]
            for (L, p), out in zip(probes, decode(part, reqs)):
                if not same(out[0], alone[L][p, qa:qb]):
                    fails.append(("heads decode", t, (qa, qb), L, p))


def exact(q, k, v):
    """float64 causal GQA attention for a full sequence."""
    L = q.shape[0]
    qd, kd, vd = q.double(), k.double().repeat_interleave(HQ // HKV, 1), v.double().repeat_interleave(HQ // HKV, 1)
    s = torch.einsum("qhd,khd->hqk", qd, kd) * SCALE
    s = s.masked_fill(torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1), -math.inf)
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), vd)


def accuracy(L, out, q, k, v):
    """Error of out vs float64, against FlashInfer's on the same inputs: (failures, mean, max)."""
    ref = exact(q, k, v).float()
    e_pi = (out.float() - ref).abs()
    e_fi = (flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True).float() - ref).abs()
    print(f"L={L} hq={q.shape[1]} hkv={k.shape[1]}: PI err mean {e_pi.mean():.3e} max {e_pi.max():.3e} | "
          f"FlashInfer err mean {e_fi.mean():.3e} max {e_fi.max():.3e} | "
          f"bf16(exact) err mean {(ref.bfloat16().float() - ref).abs().mean():.3e}")
    bad = e_pi.mean() > ACC_SLACK[0] * e_fi.mean() or e_pi.max() > ACC_SLACK[1] * e_fi.max()
    return [("accuracy", L, q.shape[1], k.shape[1])] if bad else [], e_pi.mean().item(), e_pi.max().item()


def run_checks():
    """All invariance and accuracy checks; returns the failures as (check, ...) tuples."""
    global KVSPLITS, VARIANT
    pool = Pool()
    fails = []
    lens = [1, 15, 63, 64, 65, 200, 517, 1300, 2049]
    seqs = {L: seq(L) for L in lens}
    pages = {}
    for L in lens:
        pages[L] = pool.alloc(L)
        pool.write(pages[L], 0, seqs[L][1], seqs[L][2])

    # Reference: every request alone, one prefill call.
    alone = {L: prefill(pool, [(seqs[L][0], pages[L], L)])[0] for L in lens}

    # batch: all requests in one prefill batch, shuffled order.
    order = lens[:]
    random.shuffle(order)
    batched = prefill(pool, [(seqs[L][0], pages[L], L) for L in order])
    for L, out in zip(order, batched):
        if not same(out, alone[L]):
            fails.append(("batch", L))

    # chunk + prefix: re-store the same KV in different pages, prefill in pieces
    # where each piece sees earlier tokens as cached prefix.
    for L in lens:
        if L < 3:
            continue
        p2 = pool.alloc(L)
        pool.write(p2, 0, seqs[L][1], seqs[L][2])
        cuts = sorted(random.sample(range(1, L), min(4, L - 1)))
        outs, start = [], 0
        for end in cuts + [L]:
            outs.append(prefill(pool, [(seqs[L][0][start:end].contiguous(), p2, end)])[0])
            start = end
        if not same(torch.cat(outs), alone[L]):
            fails.append(("chunk+prefix", L))

    # split / variant: prefill forced over 1..7 split CTAs per row tile, with every
    # prefill kernel variant (Q in smem / registers, 4 / 8 warps), vs the planner's run.
    for VARIANT in VARIANTS:
        for KVSPLITS in (1, 2, 3, 7):
            for L in lens:
                if not same(prefill(pool, [(seqs[L][0], pages[L], L)])[0], alone[L]):
                    fails.append(("split", VARIANT, L, KVSPLITS))
    KVSPLITS, VARIANT = 0, 0

    # path + batch: decode token p (cache holds 0..p) vs prefill row p, for every position of
    # every request (a one-ulp fp32 difference reaches the bf16 output only rarely, so the check
    # needs volume: kernels/test_mutants.py), in two random batchings of random sizes.
    for _ in range(2):
        probes = [(L, p) for L in lens for p in range(L)]
        random.shuffle(probes)
        i = 0
        while i < len(probes):
            n = random.randint(1, 64)
            group, i = probes[i:i + n], i + n
            reqs = [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in group]
            for (L, p), out in zip(group, decode(pool, reqs)):
                if not same(out[0], alone[L][p]):
                    fails.append(("path", L, p))
        for L, p in probes[:8]:  # alone: a single request runs another grid order (decode_grid)
            if not same(decode(pool, [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1)])[0][0], alone[L][p]):
                fails.append(("path alone", L, p))

    check_heads(pool, seqs, pages, alone, fails)

    # Accuracy vs float64, against FlashInfer's.
    for L in [517, 2049]:
        q, k, v = seqs[L]
        fails += accuracy(L, alone[L], q, k, v)[0]
    return fails


def main():
    fails = run_checks()
    print("checks failed:", len(fails), fails[:10])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
