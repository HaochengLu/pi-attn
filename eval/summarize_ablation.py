"""Summarize eval/e2e_ablation.sh: median over launches per mode of output tok/s, time per output token (TPOT)
and time to first token (TTFT), and each mode's change relative to the original engine: throughput in %, TPOT in
microseconds per token, TTFT in %. Usage: python eval/summarize_ablation.py results/e2e_ablation"""
import glob, json, statistics, sys

d = sys.argv[1]
MODES = ("flashinfer", "pi_attn", "pi_gemm", "deterministic")
names = sorted({f.split("/")[-1].split(".flashinfer.")[0] for f in glob.glob(f"{d}/*.flashinfer.L*.json")})
print("| config | workload | original tok/s | PI attention only | row-invariant GEMM only | both (--deterministic) "
      "| original TPOT ms | TPOT: attention, GEMM, both (us) | original TTFT ms | TTFT: attention, GEMM, both |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name in names:
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{name}.{m}.L*.json"))] for m in MODES}
    for i, w in enumerate(runs["flashinfer"][0]):
        med = {k: {m: statistics.median(r[i][k] for r in runs[m]) for m in MODES if runs[m]}
               for k in ("out_tok_s", "tpot_ms", "ttft_ms")}
        t, p, f = med["out_tok_s"], med["tpot_ms"], med["ttft_ms"]
        cells = [f"{100 * (t[m] / t['flashinfer'] - 1):+.1f}%" if m in t else "" for m in MODES[1:]]
        tpot = ", ".join(f"{(p[m] - p['flashinfer']) * 1e3:+.0f}" for m in MODES[1:] if m in p)
        ttft = ", ".join(f"{100 * (f[m] / f['flashinfer'] - 1):+.1f}%" for m in MODES[1:] if m in f)
        print(f"| {name} ({len(runs['flashinfer'])} launches) | {w['in']}/{w['out']}/c{w['concurrency']} | {t['flashinfer']:.1f} | "
              + " | ".join(cells) + f" | {p['flashinfer']:.3f} | {tpot} | {f['flashinfer']:.1f} | {ttft} |")
