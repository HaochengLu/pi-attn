#!/usr/bin/env python3
"""Identity table: GPU hashes from the logs given on the command line vs the CPU reference hashes computed on this
machine (cpu_hashes_mac.txt). P21C / P11C are the P21 / P11 functions (canonical weights), so they share hashes."""
import re, sys
alias = {'P21C': 'P21', 'P11C': 'P11'}
mac = {}
for l in open('cpu_hashes_mac.txt'):
    m = re.match(r'XCPHASH case (\d+) N=(\d+) K=(\d+) kind=(\d+) (\S+) cpu=(\w+)', l)
    if m: mac[(int(m[1]), m[5])] = m[6]
logs = {}
for path in sys.argv[1:]:
    gpu = None; d = {}; cls = {}
    for l in open(path, errors='replace'):
        m = re.match(r'GPU (.*?) sm_(\d+)', l)
        if m: gpu = f"{m[1].replace('NVIDIA ', '')} (sm_{m[2]})"
        m = re.match(r'XCPHASH case (\d+) N=(\d+) K=(\d+) kind=(\d+) (\S+) gpu=(\w+) cpu=(\w+) (\w+) checks=(\d+) refused=(-?\d+) \| mismatches \[([^\]]*)\]', l)
        if m: d[(int(m[1]), m[5])] = (m[6], m[7], m[8], int(m[9]), int(m[10]), sum(int(v) > 0 for v in m[11].split()), f"N={m[2]} K={m[3]} kind={m[4]}")
        m = re.match(r'CLASSHASH case (\d+) .* PI-native-fp32=(\w+) cuBLAS-bf16=(\w+)', l)
        if m: cls[int(m[1])] = (m[2], m[3])
    logs[gpu or path] = (d, cls)
keys = sorted({k for d, _ in logs.values() for k in d})
bad = 0
print("case            program | " + " | ".join(f"{g:>22s}" for g in logs) + " |   CPU ref (this Mac)")
for k in keys:
    row = []; want = mac.get((k[0], alias.get(k[1], k[1])))
    desc = ''
    for g, (d, _) in logs.items():
        if k not in d: row.append(f"{'-':>22s}"); continue
        h, hc, ok, n, ref, mm, desc = d[k]
        good = h == hc == want and mm == 0
        bad += not good
        row.append(f"{h} {'ok ' if good else 'BAD'}{n - ref:2d}")
    print(f"{k[0]:2d} {desc:24s} {k[1]:5s} | " + " | ".join(row) + f" | {want}")
print(f"\n{'ALL IDENTICAL' if not bad else 'MISMATCHES: ' + str(bad)}: every GPU hash == CPU hash on the box == CPU hash on the Mac; the number after ok = M/scheduling checks run and bit-equal")
print("\nclass-dependent kernels for contrast (case: PI native fp32 | cuBLAS bf16):")
for c in sorted({c for _, cl in logs.values() for c in cl}):
    print(f"  case {c:2d}: " + " | ".join(f"{g}: {cl.get(c, ('-', '-'))[0]} {cl.get(c, ('-', '-'))[1]}" for g, (_, cl) in logs.items()))
