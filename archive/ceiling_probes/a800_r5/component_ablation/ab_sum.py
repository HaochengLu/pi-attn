"""Summarize abl.sh launches: per workload, the median over launches of output tok/s, TPOT and TTFT of the base
mode, and each other mode's change against it. usage: python ab_sum.py <dir> <base mode> <mode>..."""
import glob, json, os, statistics, sys
d, base, modes = sys.argv[1], sys.argv[2], sys.argv[3:]
M = [base] + modes
names = sorted({os.path.basename(f).split(f".{base}.")[0] for f in glob.glob(f"{d}/*.{base}.L*.json")})
print(f"| config | workload | {base} tok/s | TPOT ms | TTFT ms | " + " | ".join(f"{m}: tok/s, TPOT, TTFT" for m in modes) + " | SM clock MHz under load |")
print("|" + "---|" * (6 + len(modes)))
for name in names:
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{name}.{m}.L*.json"))] for m in M}
    n = min(len(runs[m]) for m in M)
    if n == 0: continue
    clocks = set()
    for f in glob.glob(f"{d}/{name}.*.clock"):
        for line in open(f):
            p = line.split(",")
            if len(p) >= 2 and int(p[1].split()[0]) > 50: clocks.add(int(p[0].strip().split()[0]))
    for i, w in enumerate(runs[base][0]):
        med = lambda m, k: statistics.median(r[i][k] for r in runs[m][:n])
        t0, p0, f0 = med(base, "out_tok_s"), med(base, "tpot_ms"), med(base, "ttft_ms")
        cells = [f"{100 * (med(m, 'out_tok_s') / t0 - 1):+.1f}%, {(med(m, 'tpot_ms') - p0) * 1e3:+.0f} us, {100 * (med(m, 'ttft_ms') / f0 - 1):+.1f}%" for m in modes]
        print(f"| {name} ({n}) | {w['in']}/{w['out']} c{w['concurrency']} | {t0:.0f} | {p0:.3f} | {f0:.1f} | " + " | ".join(cells) + f" | {min(clocks) if clocks else '?'}-{max(clocks) if clocks else '?'} |")
