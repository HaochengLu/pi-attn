"""XC (cross-class exact) attention: build helper, the Q/K/V projections, data, references.

Canonical program (all rounding declared, no tensor-core rounding ever fires):
  Q^, K^ = proj_qk(Q), proj_qk(K): per row (blk = 128) or per 32- / 16-dim block, round to the
           finest power-of-two grid 2^e such that sum (x/2^e)^2 < 2^24. Cauchy-Schwarz then bounds every
           partial sum of the Q^ K^ products below 2^24 grid units: S is an exact integer in fp32.
  V~, sv = proj_v(V): per key ev = exponent of max |v| (|v| < 2^ev), V~ = RN(V 2^(11 - ev)) 2^-11, sv = 2^ev.
  kernel: S exact (blk < 128: fp32 sum of the exact block partials, ascending), online softmax as
          PI-Attn with p in fp32, d = fma(d, scale, sum p), weights W^ = bf16(RN_qw(p sv)) per 32-key
          tile (xc_weights), the tile's partial sum W^ V~ exact, o = fma(o, scale, partial), fold
          groups, o / d -> bf16.
  gauge_v: optional exact reparametrisation V_kj -> V_kj 2^-c_j (per kv head and channel, a calibrated
          constant; the output is multiplied back by 2^c_j, exactly), so static outlier channels do not
          cost the other channels their bits in the per-key grid.
"""
import hashlib
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
D, PAGE = 128, 16
BQK, BV, EMIN, EMAX = 12, 11, -40, 40


def build(xc, qb=128, verbose=False):
    from torch.utils.cpp_extension import load
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    return load(name=f"xc_ext_{xc}_{qb}_sm{arch}", sources=[os.path.join(HERE, "pi_attn_torch.cu")],
                extra_include_paths=[HERE],
                extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                   "--expt-relaxed-constexpr", f"-DXC={xc}", f"-DXC_QB={qb}"],
                verbose=verbose)


def sha(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _to_bf16_exact(x):
    b = x.to(torch.bfloat16)
    assert bool((b.double() == x).all()), "projection left the bf16 grid"
    return b


def proj_qk(x, blk):
    """x [..., 128] bf16 -> bf16 on the exactness grid (see module doc). Exact float64/integer ops only."""
    xd = x.double()
    xb = xd.reshape(*xd.shape[:-1], D // blk, blk)
    mx = xb.abs().amax(-1, keepdim=True)
    assert bool(torch.isfinite(mx).all()) and float(mx.max()) < 2.0 ** EMAX, "outside the declared range"
    e = torch.clamp(torch.frexp(mx)[1] - BQK, min=EMIN)  # max / 2^e in [2^11, 2^12)
    for _ in range(6):
        xi = torch.round(torch.ldexp(xb, -e))
        ok = (xi * xi).sum(-1, keepdim=True) < 2.0 ** (2 * BQK)
        if bool(ok.all()):
            break
        e = torch.where(ok, e, e + 1)
    assert bool(ok.all())
    return _to_bf16_exact(torch.ldexp(xi, e).reshape(xd.shape))


def proj_v(v):
    """v [..., 128] bf16 -> (V~ bf16 with |V~| <= 1 on the 2^-BV grid, sv f32 = 2^ev [...])."""
    vd = v.double()
    mx = vd.abs().amax(-1, keepdim=True)
    assert bool(torch.isfinite(mx).all()) and float(mx.max()) < 2.0 ** EMAX, "outside the declared range"
    ev = torch.clamp(torch.frexp(mx)[1], min=EMIN)
    vt = torch.round(torch.ldexp(vd, BV - ev)) * 2.0 ** -BV
    return _to_bf16_exact(vt), torch.ldexp(torch.ones_like(mx), ev).squeeze(-1).float()


def gauge_v(v):
    """v [L, hkv, 128] -> (v 2^-c exactly, c [hkv, 128] int): c = round(log2(channel RMS / median channel RMS))."""
    rms = v.double().pow(2).mean(0).sqrt().clamp(min=2.0 ** -30)
    c = torch.round(torch.log2(rms / rms.median(-1, keepdim=True)[0])).int()
    return torch.ldexp(v.double(), -c).to(torch.bfloat16), c


def project(q, k, v, blk):
    kt, (vt, sv) = proj_qk(k, blk), proj_v(v)
    return proj_qk(q, blk), kt, vt, sv


def gauss(gen, *shape):
    """Irwin-Hall N(0, 1) from integers only: the same bits on every host CPU."""
    acc = torch.zeros(shape, dtype=torch.int32)
    for _ in range(12):
        acc += torch.randint(0, 65536, shape, generator=gen, dtype=torch.int32)
    return (acc - 393210).float() / 65536.0


def make_data(name, L, hq, hkv, seed=0):
    """(q [L, hq, 128], k, v [L, hkv, 128]) bf16 on the CPU."""
    gen = torch.Generator().manual_seed(1000 + seed)
    q, k, v = gauss(gen, L, hq, D), gauss(gen, L, hkv, D), gauss(gen, L, hkv, D)
    grp = hq // hkv
    if name == "gauss":
        pass
    elif name == "sharp":
        q *= 4
    elif name == "outlier":
        q *= 2
        for c in (5, 37, 70, 101):
            q[..., c] *= 12
            k[..., c] *= 12
        for c in (3, 64):
            v[..., c] *= 16
    elif name == "peaky":
        tgt = (torch.rand(L, generator=gen) * torch.arange(1, L + 1)).long().clamp(max=L - 1)
        tgt = torch.minimum(tgt, torch.arange(L))
        q = 0.9 * k[tgt].repeat_interleave(grp, 1) + 0.5 * q
    elif name == "adv_wide":
        for t in (q, k, v):
            t *= torch.ldexp(torch.ones(()), torch.randint(-10, 11, t.shape, generator=gen)).float()
    elif name == "adv_cancel":
        k[..., 1::2] = k[..., 0::2]
        q[..., 1::2] = -q[..., 0::2]
        q[..., 126:] = 0.25 * gauss(gen, L, hq, 2)
        v[1::2] = -v[0::2][: v[1::2].shape[0]]
    elif name == "adv_huge":
        for t in (q, k, v):
            t *= torch.where(torch.rand(t.shape, generator=gen) < 0.005, 2.0 ** 20, 1.0)
        k[::97] = 0
        v[::89] = 0
        q[::101] = 0
    elif name == "adv_scaled":
        q *= 2.0 ** -30
        k *= 2.0 ** 25
        v *= 2.0 ** -35
    elif name == "adv_const":
        q[:] = 1.0
        k[:] = 1.0
        v[:] = 1.0
        v[1::3] = -1.0
        v[2::7, :, ::5] = 255.0 / 256
    else:
        raise ValueError(name)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def to_pages(k, v, sv, dev="cuda"):
    """Identity paging (page 0 is NaN garbage and unused): k/v [L, hkv, 128], sv [L, hkv]."""
    L, hkv = k.shape[0], k.shape[1]
    n = -(-L // PAGE)
    kc = torch.full(((n + 1) * PAGE, hkv, D), float("nan"), dtype=torch.bfloat16, device=dev)
    vc = torch.full(((n + 1) * PAGE, hkv, D), float("nan"), dtype=torch.bfloat16, device=dev)
    sc = torch.full(((n + 1) * PAGE, hkv), float("nan"), dtype=torch.float32, device=dev)
    kc[PAGE:PAGE + L], vc[PAGE:PAGE + L], sc[PAGE:PAGE + L] = k.to(dev), v.to(dev), sv.to(dev)
    return (kc.view(n + 1, PAGE, hkv, D), vc.view(n + 1, PAGE, hkv, D), sc.view(n + 1, PAGE, hkv),
            torch.arange(1, n + 1, device=dev, dtype=torch.int32))


def prefill_full(ext, q, k, v, sv, variant=0, splits=0, dev="cuda"):
    """One request, causal prefill of the whole sequence."""
    L = q.shape[0]
    kc, vc, sc, idx = to_pages(k, v, sv, dev)
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
    return ext.prefill(q.to(dev).contiguous(), kc, vc, sc, t([0, L]), t([0, idx.numel()]), idx, t([L]), L, L,
                       1 / math.sqrt(D), splits, variant)


def exact64(q, k, v, dev="cuda", chunk=1024):
    """float64 causal GQA attention, [L, hq, 128] float64 (on dev)."""
    L, hq, hkv = q.shape[0], q.shape[1], k.shape[1]
    grp = hq // hkv
    out = torch.empty(L, hq, D, dtype=torch.float64, device=dev)
    kd, vd = k.to(dev).double(), v.to(dev).double()
    pos = torch.arange(L, device=dev)
    for h in range(hq):
        kh, vh = kd[:, h // grp], vd[:, h // grp]
        for r0 in range(0, L, chunk):
            r1 = min(L, r0 + chunk)
            s = (q[r0:r1, h].to(dev).double() @ kh.T) / math.sqrt(D)
            s.masked_fill_(pos[None, :] > pos[r0:r1, None], -math.inf)
            out[r0:r1, h] = torch.softmax(s, -1) @ vh
    return out
