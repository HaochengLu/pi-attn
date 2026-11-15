"""Summarize r6_abl.sh: per workload, median over launches of each mode's output tok/s, TPOT and TTFT and its change
against the first mode. usage: r6_abl_sum.py <dir> [mode ...]"""
import glob, json, os, statistics, sys
d = sys.argv[1]
names = sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{d}/*.L*.json")})
for name in names:
    modes = sys.argv[2:] or sorted({os.path.basename(f)[len(name) + 1:].rsplit(".L", 1)[0] for f in glob.glob(f"{d}/{name}.*.L*.json")}, key=lambda m: (m != "flashinfer", m))
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{name}.{m}.L*.json"))] for m in modes}
    runs = {m: r for m, r in runs.items() if r}
    modes = [m for m in modes if m in runs]
    n = min(len(r) for r in runs.values())
    clocks = set()
    for f in glob.glob(f"{d}/{name}.*.clock"):
        for line in open(f):
            p = line.split(",")
            if len(p) >= 2 and int(p[1].split()[0]) > 50: clocks.add(p[0].strip().split()[0])
    print(f"\n### {name} ({n} launches per mode; SM clock under load {'/'.join(sorted(clocks))} MHz)\n")
    print("| workload | metric | " + " | ".join(modes) + " |")
    print("|---|---|" + "---|" * len(modes))
    for i, w in enumerate(runs[modes[0]][0]):
        for key, label, fmt in (("out_tok_s", "out tok/s", "{:.1f}"), ("ttft_ms", "TTFT ms", "{:.1f}"), ("tpot_ms", "TPOT ms", "{:.3f}")):
            med = {m: statistics.median(r[i][key] for r in runs[m][:n]) for m in modes}
            rng = {m: (min(r[i][key] for r in runs[m][:n]), max(r[i][key] for r in runs[m][:n])) for m in modes}
            cells = [fmt.format(med[modes[0]]) + f" [{fmt.format(rng[modes[0]][0])}-{fmt.format(rng[modes[0]][1])}]"]
            cells += [f"{fmt.format(med[m])} ({100 * (med[m] / med[modes[0]] - 1):+.1f}%)" for m in modes[1:]]
            print(f"| {w['in']}/{w['out']} c{w['concurrency']}" + (f" shared {w['shared']}" if w.get('shared') else "") + f" | {label} | " + " | ".join(cells) + " |")
