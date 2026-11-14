#!/usr/bin/env python3
"""Bit-exact models of the bf16 x bf16 -> fp32 tensor-core step (mma.sync m16n8k16 / wgmma m64n8k16),
calibrated on L4 (sm_89, class A) and H100 (sm_90, class H).  Pure Python integers, no dependencies.

One fused group over terms t_0 = c (fp32) and t_i = a_i * b_i (exact 16-bit products):
    E    = max exponent over the non-zero terms; exponent of c is floor(log2|c|), exponent of a product is
           e(a_i) + e(b_i)  (NOT renormalised: the product significand lies in [1, 4))
    unit = 2^(E - F)
    S    = sum_i sign_i * floor(|t_i| / unit)          (each term truncated toward zero, then an exact sum)
    out  = S * unit truncated toward zero to 24 significant bits (fp32); a zero result is +0
class A (Ampere / Ada):  F = 24, a k16 step is TWO chained groups: k 0..7, then k 8..15 on the fp32 result
class H (Hopper):        F = 25, a k16 step is ONE group of 16 products
NaN / Inf: any NaN operand, Inf * 0, or Inf - Inf -> 0x7FFFFFFF; otherwise +-Inf.  bf16 subnormal inputs are
honoured (not flushed); fp32 subnormal results are truncated at 2^-149.
"""
import sys

def _term_c(c):
    s, E, f = c >> 31, (c >> 23) & 255, c & 0x7FFFFF
    return (s, f, -149, -126) if E == 0 else (s, f | 0x800000, E - 150, E - 127)   # sign, mag, lsb exponent, exponent

def _term_p(a, b):
    s = (a >> 15) ^ (b >> 15)
    Ea, Eb, ma, mb = (a >> 7) & 255, (b >> 7) & 255, a & 127, b & 127
    ea = -126 if Ea == 0 else Ea - 127
    eb = -126 if Eb == 0 else Eb - 127
    if Ea: ma |= 128
    if Eb: mb |= 128
    return (s, ma * mb, ea + eb - 14, ea + eb)

def _rz_f32(sign, mag, lsb):
    if mag == 0: return 0
    e = mag.bit_length() - 1 + lsb
    q = max(e - 23, -149)
    m = mag << (lsb - q) if lsb >= q else mag >> (q - lsb)
    if m == 0: return 0
    if m < (1 << 23): return (sign << 31) | m
    E = q + 23 + 127
    return (sign << 31) | 0x7F800000 if E >= 255 else (sign << 31) | (E << 23) | (m & 0x7FFFFF)

def group(F, c, a, b):
    terms = [_term_c(c)]
    nan, inf = False, 0
    if (c >> 23) & 255 == 255:
        if c & 0x7FFFFF: nan = True
        else: inf |= 2 if c >> 31 else 1
        terms = []
    for x, y in zip(a, b):
        sx, sy = (x >> 7) & 255 == 255, (y >> 7) & 255 == 255
        if sx or sy:
            if (sx and x & 127) or (sy and y & 127) or not x & 0x7FFF or not y & 0x7FFF: nan = True
            else: inf |= 2 if ((x ^ y) >> 15) & 1 else 1
            continue
        terms.append(_term_p(x, y))
    if nan or inf == 3: return 0x7FFFFFFF
    if inf: return 0x7F800000 if inf == 1 else 0xFF800000
    nz = [t for t in terms if t[1]]
    if not nz:
        return 0x80000000 if terms and all(t[0] for t in terms) else 0
    unit = max(t[3] for t in nz) - F
    S = 0
    for s, m, lsb, _ in nz:
        v = m << (lsb - unit) if lsb >= unit else m >> (unit - lsb)
        S += -v if s else v
    return _rz_f32(1 if S < 0 else 0, abs(S), unit)

def mma_k16(cls, c, a, b):
    """cls 'A' (Ampere/Ada) or 'H' (Hopper). c: fp32 bits; a, b: 16 bf16 bit patterns each."""
    if cls == 'H': return group(25, c, a, b)
    return group(24, group(24, c, a[:8], b[:8]), a[8:], b[8:])

if __name__ == '__main__':
    n = bad = 0
    for line in open(sys.argv[1] if len(sys.argv) > 1 else 'vectors_mma.txt'):
        w = line.split(); c, gA, gH = (int(x, 16) for x in w[1:4]); v = [int(x, 16) for x in w[4:]]
        n += 1; bad += mma_k16('A', c, v[:16], v[16:]) != gA; bad += mma_k16('H', c, v[:16], v[16:]) != gH
    print(f"tc_models.py: {n} measured cases x 2 classes, {bad} mismatches vs GPU (L4, H100)")
    sys.exit(bad != 0)
