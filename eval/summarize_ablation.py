"""Summarize eval/e2e_ablation.sh: median output tok/s over launches per mode, and each
mode's change relative to the original engine. Usage: python eval/summarize_ablation.py results/e2e_ablation"""
import glob, json, statistics, sys

d = sys.argv[1]
MODES = ("flashinfer", "pi_attn", "pi_gemm", "deterministic")
names = sorted({f.split("/")[-1].split(".flashinfer.")[0] for f in glob.glob(f"{d}/*.flashinfer.L*.json")})
print("| config | workload | original tok/s | PI attention only | row-invariant GEMM only | both (--deterministic) |")
print("|---|---|---|---|---|---|")
for name in names:
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{name}.{m}.L*.json"))] for m in MODES}
    for i, w in enumerate(runs["flashinfer"][0]):
        med = {m: statistics.median(r[i]["out_tok_s"] for r in runs[m]) for m in MODES if runs[m]}
        base = med["flashinfer"]
        cells = [f"{100 * (med[m] / base - 1):+.1f}%" if m in med else "" for m in MODES[1:]]
        print(f"| {name} ({len(runs['flashinfer'])} launches) | {w['in']}/{w['out']}/c{w['concurrency']} | {base:.1f} | "
              + " | ".join(cells) + " |")
