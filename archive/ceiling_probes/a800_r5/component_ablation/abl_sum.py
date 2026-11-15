"""Summarize abl.sh: per workload, median over launches of tok/s, TPOT, TTFT per mode, each mode's change vs
FlashInfer mode, and each component's share of the PI-vs-FI TPOT / throughput gap.
usage: python abl_sum.py <dir>"""
import glob, json, statistics, sys, os
d = sys.argv[1]
M = ("flashinfer", "pi_attn", "pi_gemm", "deterministic")
names = sorted({os.path.basename(f).split(".flashinfer.")[0] for f in glob.glob(f"{d}/*.flashinfer.L*.json")})
print("| config | workload | FI tok/s | attn only | GEMM only | both | sum of halves | FI TPOT ms | attn dTPOT us | GEMM dTPOT us | both dTPOT us | FI TTFT ms | attn dTTFT | GEMM dTTFT | both dTTFT | clock MHz |")
print("|" + "---|" * 16)
for name in names:
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{name}.{m}.L*.json"))] for m in M}
    n = min(len(runs[m]) for m in M)
    if n == 0: continue
    clocks = set()
    for f in glob.glob(f"{d}/{name}.*.clock"):
        for line in open(f):
            p = line.split(",")
            if len(p) >= 2 and int(p[1].split()[0]) > 50: clocks.add(p[0].strip().split()[0])
    for i, w in enumerate(runs["flashinfer"][0]):
        med = lambda m, k: statistics.median(r[i][k] for r in runs[m][:n])
        t = {m: med(m, "out_tok_s") for m in M}; p = {m: med(m, "tpot_ms") for m in M}; f = {m: med(m, "ttft_ms") for m in M}
        pc = lambda m: f"{100 * (t[m] / t['flashinfer'] - 1):+.1f}%"
        s = 100 * (t["pi_attn"] / t["flashinfer"] - 1) + 100 * (t["pi_gemm"] / t["flashinfer"] - 1)
        dp = lambda m: f"{(p[m] - p['flashinfer']) * 1e3:+.0f}"
        df = lambda m: f"{100 * (f[m] / f['flashinfer'] - 1):+.1f}%"
        print(f"| {name} ({n}) | {w['in']}/{w['out']} c{w['concurrency']} | {t['flashinfer']:.0f} | {pc('pi_attn')} | {pc('pi_gemm')} | {pc('deterministic')} | {s:+.1f}% | "
              f"{p['flashinfer']:.3f} | {dp('pi_attn')} | {dp('pi_gemm')} | {dp('deterministic')} | {f['flashinfer']:.1f} | {df('pi_attn')} | {df('pi_gemm')} | {df('deterministic')} | {'/'.join(sorted(clocks))} |")
