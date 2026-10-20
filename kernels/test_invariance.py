"""GPU invariance and accuracy tests for the PI-Attn kernels.

Five partition changes must leave every output bit unchanged:
  batch    : a request computed alone vs inside a mixed batch
  split    : decode with 1..many split CTAs
  chunk    : prefill in one call vs chunked prefill with arbitrary cut points
  prefix   : prefix KV already cached (different physical pages) vs computed now
  path     : prefill row at position p vs decode of the token at position p
Accuracy is compared against float64 attention and torch SDPA (bf16).
"""
import math
import random
import sys

import torch

from build import build

ext = build()
C = ext.CHUNK
dev = "cuda"
torch.manual_seed(0)
random.seed(0)
PAGE = 16
HQ, HKV, D = 16, 8, 128
SCALE = 1 / math.sqrt(D)
WARPS = int(__import__('os').environ.get('PI_CFG', '0'))
DVAR = int(__import__('os').environ.get('PI_DVAR', '0'))


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
                      m["kv_lens"], max(r[0].shape[0] for r in reqs), SCALE, WARPS)
    return list(out.split([r[0].shape[0] for r in reqs]))


def decode(pool, reqs, splits):
    """reqs: list of (q[1], pages, kv_len)."""
    q = torch.cat([r[0] for r in reqs]).contiguous()
    m = meta([r[1] for r in reqs], [r[2] for r in reqs])
    max_chunks = max((r[2] + C - 1) // C for r in reqs)
    ws_o = torch.full((len(reqs), HQ, max_chunks, D), float("nan"), device=dev)
    ws_ml = torch.full((len(reqs), HQ, max_chunks, 2), float("nan"), device=dev)
    out = ext.decode(q, pool.k, pool.v, m["kv_page_indptr"], m["kv_page_indices"], m["kv_lens"],
                     ws_o, ws_ml, splits, SCALE, DVAR)
    return list(out.split(1))


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def exact(q, k, v):
    """float64 causal GQA attention for a full sequence."""
    L = q.shape[0]
    qd, kd, vd = q.double(), k.double().repeat_interleave(HQ // HKV, 1), v.double().repeat_interleave(HQ // HKV, 1)
    s = torch.einsum("qhd,khd->hqk", qd, kd) * SCALE
    s = s.masked_fill(torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1), -math.inf)
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), vd)


def main():
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

    # path + split + batch: decode token p (cache holds 0..p) vs prefill row p.
    for splits in [1, 2, 3, 7, 64]:
        probes = [(L, p) for L in lens for p in sorted({0, L // 2, L - 1, random.randrange(L)})]
        random.shuffle(probes)
        for i in range(0, len(probes), 9):
            group = probes[i:i + 9]
            outs = decode(pool, [(seqs[L][0][p:p + 1].contiguous(), pages[L], p + 1) for L, p in group], splits)
            for (L, p), out in zip(group, outs):
                if not same(out[0], alone[L][p]):
                    fails.append(("path/split", L, p, splits))

    # Accuracy vs float64 and vs bf16 SDPA.
    for L in [517, 2049]:
        q, k, v = seqs[L]
        ref = exact(q, k, v).float()
        sd = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True,
            enable_gqa=True).transpose(0, 1).float()
        e_pi = (alone[L].float() - ref).abs()
        e_sd = (sd - ref).abs()
        print(f"L={L}: PI err mean {e_pi.mean():.3e} max {e_pi.max():.3e} | "
              f"SDPA err mean {e_sd.mean():.3e} max {e_sd.max():.3e}")

    print("checks failed:", len(fails), fails[:10])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
