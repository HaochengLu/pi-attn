#!/usr/bin/env python3
"""Independent check of the XCP reference (xcp.h) with the round-2 exact-rational reference (theory/xc_ref.py):
slicing and fp32 rounding come from xc_ref.py; the fold (512-k segments, TRI / TRIF / P21 / P11) is restated here."""
import sys
from fractions import Fraction
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else '../../round2/theory')
from xc_ref import slice_group, f32_round, f32_bits
PROGS = [(2, 2, 2, 2, 'tri'), (2, 2, 2, 2, 'trif'), (2, 1, 0, 4, 'p21'), (1, 1, 2, 2, 'p11')]
def gemm(X, W, M, N, K, prog, segb=32):
    SA, SB, sA, sB, kind = PROGS[prog]; nb = K // 16; out = []
    for m in range(M):
        xs = [slice_group(X[m * K + b * 16:m * K + b * 16 + 16], sA) for b in range(nb)]
        for n in range(N):
            ws = [slice_group(W[n * K + b * 16:n * K + b * 16 + 16], sB) for b in range(nb)]
            tot = Fraction(0)
            for s0 in range(0, nb, segb):
                p = Fraction(0)
                for b in range(s0, min(nb, s0 + segb)):
                    d = lambda j, l: sum(a * c for a, c in zip(xs[b][j], ws[b][l]))
                    steps = {'tri': [d(0, 0), d(0, 1), d(1, 0)], 'trif': [d(0, 0), d(0, 1) + d(1, 0)], 'p21': [d(0, 0), d(1, 0)], 'p11': [d(0, 0)]}[kind]
                    for v in steps:
                        assert f32_round(v, 'RZ') == v, "step is not an fp32 value"
                        p = f32_round(p + v, 'RNE')
                tot = f32_round(tot + p, 'RNE')
            out.append(f32_bits(tot))
    return out
n = bad = 0
for line in open(sys.argv[1] if len(sys.argv) > 1 else 'vectors_xcp.txt'):
    w = line.split()
    if w[0] == 'DATA': M, N, K = map(int, w[1:4])
    elif w[0] == 'X': X = [int(v, 16) for v in w[1:]]
    elif w[0] == 'W': W = [int(v, 16) for v in w[1:]]
    elif w[0] == 'PROG':
        got = gemm(X, W, M, N, K, int(w[1])); want = [int(v, 16) for v in w[2:]]
        n += len(got); bad += sum(a != b for a, b in zip(got, want))
print(f"xcp_ref.py: {n} outputs (4 data kinds x 4 programs, K = {K}: 3 segments) vs xcp.h reference: {bad} mismatches")
sys.exit(bad != 0)
