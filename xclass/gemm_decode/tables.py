#!/usr/bin/env python3
"""Compact speed tables from xcp bench logs. usage: tables.py log [regime=COLD|HOT] [progs=TRIF,P21C,...]"""
import re, sys
log = sys.argv[1]; regime = sys.argv[2] if len(sys.argv) > 2 else 'COLD'
progs = (sys.argv[3] if len(sys.argv) > 3 else 'TRIF,P21C,P11C,SKEL').split(',')
rows = []
for l in open(log, errors='replace'):
    m = re.match(r'BENCH (.{11}) N=\s*(\d+) K=\s*(\d+) M=\s*(\d+) (HOT |COLD) cuBLAS\s+([-\d.]+) native\s+([-\d.]+) \(k(-?\d+)\) \|(.*?) ALL:(.*)', l)
    if not m: continue
    best = {g: (float(t), nm) for g, t, nm in re.findall(r' (\w+) ([\d.]+) \(([^)]*)\) x', m[9])}
    allv = dict(re.findall(r'(\S+) ([\d.]+)', m[10]))
    rows.append((m[1].strip(), int(m[2]), int(m[3]), int(m[4]), m[5].strip(), float(m[6]), float(m[7]), int(m[8]), best, allv))
gpu = next((l.strip() for l in open(log, errors='replace') if l.startswith('GPU ')), '')
print(f"{gpu} -- {regime}: us per GEMM; x = time / native PI kernel (best id), [y] = time / cuBLAS")
print(f"{'shape':12s} {'N':>5s} {'K':>5s} {'M':>4s} | {'cuBLAS':>7s} {'native':>7s} {'id':>4s} | " + " | ".join(f"{p:>22s}" for p in progs))
import math
geo = {p: {} for p in progs}
for name, N, K, M, rg, cb, nat, kid, best, allv in rows:
    if rg != regime: continue
    cells = []
    for p in progs:
        if p in best:
            t = best[p][0]; cells.append(f"{t:7.1f} x{t / nat:4.2f} [{t / cb:4.2f}]")
            geo[p].setdefault(M, []).append((t / nat, t / cb))
        else: cells.append(' ' * 22)
    print(f"{name:12s} {N:5d} {K:5d} {M:4d} | {cb:7.1f} {nat:7.1f} {'k' + str(kid):>4s} | " + " | ".join(f"{c:>22s}" for c in cells))
print("\ngeometric mean over shapes of time / native [time / cuBLAS], and worst shape:")
for p in progs:
    print(f"  {p:5s} " + "  ".join(f"M={M}: x{math.exp(sum(math.log(a) for a, _ in v) / len(v)):.2f} [{math.exp(sum(math.log(b) for _, b in v) / len(v)):.2f}] max x{max(a for a, _ in v):.2f}" for M, v in sorted(geo[p].items())))
if '--sched' in sys.argv:
    print("\nbest scheduling per (shape, M):")
    for name, N, K, M, rg, cb, nat, kid, best, allv in rows:
        if rg == regime: print(f"  {name:12s} M={M:4d} " + " ".join(f"{p}:{best[p][1]}" for p in progs if p in best))
