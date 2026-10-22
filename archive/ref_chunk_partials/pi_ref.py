"""Bit-level reference model of Partition-Invariant Attention (PI-Attn).

Every floating point operation is an explicit elementwise fp32 op in a fixed
order, so results depend only on the operands and the op sequence, never on
tensor shapes or library reduction strategies. This models a kernel whose
per-element MMA accumulation order is fixed (to be verified on hardware
separately).

Contract: KV positions are cut into chunks of C keys at absolute positions
[cC, (c+1)C). A chunk partial (m_c, l_c, o_c) is a pure function of
(q, K_c, V_c, mask_c). Outputs are produced by one canonical sequential merge
of the partials in ascending chunk order. Prefill (register running merge) and
decode (split-KV partials + merge kernel) both execute that same sequence.
"""
import math

import torch

F32 = torch.float32
LOG2E = 1.4426950408889634


def bf16(x):
    return x.to(torch.bfloat16).to(F32)


def dot_fixed(q, k):
    """[R, D] x [N, D] -> [R, N], accumulating over D in index order."""
    acc = torch.zeros(q.shape[0], k.shape[0], dtype=F32)
    for d in range(q.shape[1]):
        acc = acc + q[:, None, d] * k[None, :, d]
    return acc


def pv_fixed(p, v):
    """[R, N] x [N, Dv] -> [R, Dv], accumulating over N in index order from zero."""
    acc = torch.zeros(p.shape[0], v.shape[1], dtype=F32)
    for j in range(p.shape[1]):
        acc = acc + p[:, j, None] * v[None, j, :]
    return acc


def rowsum_fixed(p):
    acc = torch.zeros(p.shape[0], dtype=F32)
    for j in range(p.shape[1]):
        acc = acc + p[:, j]
    return acc


def chunk_partial(q, k_c, v_c, valid, scale, snap):
    """valid: [R, N] bool. Masked keys contribute exact zeros (v is zeroed too,
    so garbage in unwritten cache slots can never leak as NaN)."""
    s = dot_fixed(q, k_c) * F32_(scale * LOG2E)
    s = torch.where(valid, s, torch.full_like(s, -math.inf))
    m = s.max(dim=1).values
    if snap:
        m = torch.ceil(m)  # ablation only: the top weight is then rounded to bf16 too (~2x error)
    p = torch.exp2(s - m[:, None])
    p = torch.where(valid, p, torch.zeros_like(p))
    pb = bf16(p)
    v_c = torch.where(valid.any(0)[:, None], v_c, torch.zeros_like(v_c))
    return m, rowsum_fixed(pb), pv_fixed(pb, v_c)


def F32_(x):
    return torch.tensor(x, dtype=F32)


def scale_by(x, e, snap):
    """x * 2^e when snapped (exact ldexp), x * exp2(e) otherwise."""
    if snap:
        return torch.ldexp(x, e.to(torch.int32))
    return x * torch.exp2(e)


def merge_sequential(parts, snap):
    """Canonical merge: ascending chunk order, running max, one op sequence."""
    M = l = o = None
    for m_c, l_c, o_c in parts:
        if M is None:
            M, l, o = m_c, l_c, o_c
            continue
        Mn = torch.maximum(M, m_c)
        a, b = M - Mn, m_c - Mn
        l = scale_by(l, a, snap) + scale_by(l_c, b, snap)
        o = scale_by(o, a[:, None], snap) + scale_by(o_c, b[:, None], snap)
        M = Mn
    return o / l[:, None]


def merge_global(parts, snap):
    """Alternative merge: global max first, then ascending fp32 sum. Equal to
    merge_sequential only under exact (snapped) scaling and no underflow."""
    M = torch.stack([p[0] for p in parts]).max(0).values
    l = torch.zeros_like(parts[0][1])
    o = torch.zeros_like(parts[0][2])
    for m_c, l_c, o_c in parts:
        l = l + scale_by(l_c, m_c - M, snap)
        o = o + scale_by(o_c, (m_c - M)[:, None], snap)
    return o / l[:, None]


def pi_attention(q, pos, K, V, C, scale, snap=False, splits=None, merge="seq"):
    """q: [R, D] query rows of one sequence/head at absolute positions pos [R].
    K, V: [T, D] cache up to the largest position. Causal: key j visible to
    row r iff j <= pos[r]. `splits` is how a decode kernel distributes chunks
    among CTAs; it changes only who computes each partial, not the partial."""
    T = K.shape[0]
    n_chunks = (int(pos.max()) // C) + 1
    parts = []
    for c in range(n_chunks):
        lo, hi = c * C, min((c + 1) * C, T)
        k_c = torch.zeros(C, K.shape[1], dtype=F32)
        v_c = torch.zeros(C, V.shape[1], dtype=F32)
        k_c[: hi - lo], v_c[: hi - lo] = K[lo:hi], V[lo:hi]
        keys = torch.arange(lo, lo + C)
        valid = keys[None, :] <= pos[:, None]
        valid &= (keys < T)[None, :]
        parts.append(chunk_partial(q, k_c, v_c, valid, scale, snap))
    # Rows whose position precedes a chunk entirely get m=-inf there; drop
    # them per row by construction: only chunks <= pos//C are visible.
    out = torch.empty(q.shape[0], V.shape[1], dtype=F32)
    for r in range(q.shape[0]):
        last = int(pos[r]) // C
        row_parts = [(m[r:r + 1], l[r:r + 1], o[r:r + 1]) for m, l, o in parts[: last + 1]]
        fn = merge_sequential if merge == "seq" else merge_global
        out[r] = fn(row_parts, snap)[0]
    return out


def fa2_attention(q, pos, K, V, tile, scale, split_points=()):
    """Standard FlashAttention-2 style online softmax with running max. KV is
    processed in tiles of `tile`; `split_points` cut the KV range into splits
    each with its own running state, merged FlashDecoding style. This is the
    non-invariant baseline: P is rounded to bf16 against the running max."""
    out = torch.empty(q.shape[0], V.shape[1], dtype=F32)
    for r in range(q.shape[0]):
        n = int(pos[r]) + 1
        bounds = [0] + [s for s in split_points if 0 < s < n] + [n]
        states = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            m = torch.full((1,), -math.inf)
            l = torch.zeros(1)
            o = torch.zeros(1, V.shape[1])
            for t0 in range(a, b, tile):
                t1 = min(t0 + tile, b)
                s = dot_fixed(q[r:r + 1], K[t0:t1]) * F32_(scale * LOG2E)
                mn = torch.maximum(m, s.max(1).values)
                alpha = torch.exp2(m - mn)
                pb = bf16(torch.exp2(s - mn[:, None]))
                l = l * alpha + rowsum_fixed(pb)
                o = o * alpha[:, None] + pv_fixed(pb, V[t0:t1])
                m = mn
            states.append((m, l, o))
        M = torch.stack([s[0] for s in states]).max(0).values
        L = sum(s[1] * torch.exp2(s[0] - M) for s in states)
        O = sum(s[2] * torch.exp2(s[0] - M)[:, None] for s in states)
        out[r] = (O / L[:, None])[0]
    return out
