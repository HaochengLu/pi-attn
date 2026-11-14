#!/usr/bin/env python3
"""XC numerical ABI -- executable CPU reference (pure Python, exact integer / rational arithmetic).

C = XC(SA, SB, sA, sB)(A, B):  A [M][K] bf16, B [K][N] bf16, K % 16 == 0, sA + sB <= 4.
  slice : per (row, k16 block) of A and per (column, k16 block) of B: E = max exponent field; group := 0 if E < 80;
          U0 = E - s; slice0 = RNE(x / 2^(U0-134)) * 2^(U0-134); slice1 = RNE(residual / 2^(U1-134)) * 2^(U1-134),
          U1 = U0 - 8 - s.  (RNE on the magnitude; bf16 subnormals -> 0; inputs must be finite and < 2^50.)
  step  : d = exact sum of the 16 products of one slice pair  (always an fp32 value: asserted)
  fold  : acc = RNE_fp32(acc + d), order: block ascending, A-slice ascending, B-slice ascending; acc starts at +0.
  tri   : slice pair (1, 1) skipped.
XR(sA, sB), sA + sB <= 10: groups of 8, slice 0 only, d = RZ_24(exact sum of 8 products), same fold.
Values are (Fraction-free) integers: a slice element is (n, U) meaning n * 2^(U - 134).
"""
import sys
from fractions import Fraction

FLUSH, MAXE = 80, 177

def rne_shift(m, sh):               # round-to-nearest-even of m / 2^sh, m >= 0
    if sh <= 0: return m << -sh
    q, rem, half = m >> sh, m & ((1 << sh) - 1), 1 << (sh - 1)
    return q + (1 if rem > half or (rem == half and q & 1) else 0)

def slice_group(xs, s):
    """xs: bf16 bit patterns. Returns two lists of exact values (Fractions): slice 0 and slice 1."""
    E = max((x >> 7) & 255 for x in xs)
    if E < FLUSH: return [Fraction(0)] * len(xs), [Fraction(0)] * len(xs)
    assert E < MAXE, "outside the ABI domain"
    U0 = E - s; U1 = U0 - 8 - s
    s0, s1 = [], []
    for x in xs:
        Ef, sign = (x >> 7) & 255, -1 if x >> 15 else 1
        if Ef == 0: s0.append(Fraction(0)); s1.append(Fraction(0)); continue
        sig = 128 | (x & 127)
        val = Fraction(sig) * Fraction(2) ** (Ef - 134)
        n0 = rne_shift(sig, U0 - Ef) if Ef < U0 else None
        v0 = val if n0 is None else Fraction(n0) * Fraction(2) ** (U0 - 134)
        r = val - v0                                   # exact residual (signed, relative to the magnitude)
        # residual as integer multiple of 2^(Ef-134): |r_int| <= 255; round to unit 2^(U1-134)
        r_int = r / (Fraction(2) ** (Ef - 134)); assert r_int.denominator == 1
        ri = abs(int(r_int)); rs = -1 if r_int < 0 else 1
        v1 = Fraction(ri) * Fraction(2) ** (Ef - 134) if Ef >= U1 else Fraction(rne_shift(ri, U1 - Ef)) * Fraction(2) ** (U1 - 134)
        s0.append(sign * v0); s1.append(sign * rs * v1)
    return s0, s1

def f32_round(v, mode):             # exact Fraction -> fp32 value (as Fraction), mode 'RNE' or 'RZ'; normal range only
    if v == 0: return Fraction(0)
    a = abs(v); e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a: e -= 1
    assert -126 <= e <= 126, "exponent outside ABI range"
    q = Fraction(2) ** (e - 23); m = a / q
    n = m.numerator // m.denominator; rem = m - n
    if mode == 'RNE' and (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1)): n += 1
    return (n * q) if v > 0 else -(n * q)

def f32_bits(v):
    if v == 0: return 0
    a = abs(v); e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a: e -= 1
    m = a / Fraction(2) ** (e - 23); assert m.denominator == 1 and (1 << 23) <= m.numerator < (1 << 24)
    return ((1 << 31) if v < 0 else 0) | ((e + 127) << 23) | (m.numerator & 0x7FFFFF)

def xc_gemm(A, B, SA, SB, sA, sB, mode=0):
    """A: M rows of K bf16 ints; B: N columns of K bf16 ints. Returns M x N fp32 bit patterns."""
    M, N, K = len(A), len(B), len(A[0]); G = 8 if mode == 2 else 16
    As = [[slice_group(A[i][g:g + G], sA) for g in range(0, K, G)] for i in range(M)]
    Bs = [[slice_group(B[n][g:g + G], sB) for g in range(0, K, G)] for n in range(N)]
    out = []
    for i in range(M):
        row = []
        for n in range(N):
            acc = Fraction(0)
            for g in range(K // G):
                pairs = [(0, 0)] if mode == 2 else [(j, l) for j in range(SA) for l in range(SB) if not (mode == 1 and j == 1 and l == 1)]
                for j, l in pairs:
                    d = sum(x * y for x, y in zip(As[i][g][j], Bs[n][g][l]))
                    if mode == 2: d = f32_round(d, 'RZ')
                    else: assert f32_round(d, 'RZ') == d, "XC theorem violated: fused sum is not an fp32 value"
                    acc = f32_round(acc + d, 'RNE')
            row.append(f32_bits(acc))
        out.append(row)
    return out

if __name__ == '__main__':
    X = W = None; T = 0; n = bad = 0
    for line in open(sys.argv[1] if len(sys.argv) > 1 else 'vectors_xc.txt'):
        w = line.split()
        if w[0] == 'DATA': T = int(w[2])
        elif w[0] == 'X': X = [int(v, 16) for v in w[1:]]
        elif w[0] == 'W': W = [int(v, 16) for v in w[1:]]
        elif w[0] == 'PROG':
            SA, SB, sA, sB, mode = map(int, w[1:6]); want = [int(v, 16) for v in w[6:]]
            rows = range(0, 64, 9)   # 8 of the 64 rows (pure-Python speed)
            A = [[X[t * 1024 + r * 16 + k] for t in range(T) for k in range(16)] for r in rows]
            Bc = [[W[t * 128 + c * 16 + k] for t in range(T) for k in range(16)] for c in range(8)]
            got = xc_gemm(A, Bc, SA, SB, sA, sB, mode)
            for ri, r in enumerate(rows):
                for c in range(8): n += 1; bad += got[ri][c] != want[r * 8 + c]
    print(f"xc_ref.py: {n} outputs checked against the C++ reference (== GPU on L4 and H100), {bad} mismatches")
    sys.exit(bad != 0)
