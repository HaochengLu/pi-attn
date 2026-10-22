"""Summarize eval/e2e_matrix.sh: per workload, the median over interleaved launches.
Usage: python eval/summarize_matrix.py results/e2e_matrix"""
import glob, json, statistics, sys

d = sys.argv[1]
print("| model | T | workload | FI tok/s | PI tok/s | Δ thpt (median) | Δ per launch | FI TPOT | PI TPOT | FI TTFT | PI TTFT |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]:
    for t in ["0", "0.8"]:
        runs = {mode: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{m}.{mode}.t{t}.L*.json"))]
                for mode in ("flashinfer", "deterministic")}
        if not runs["flashinfer"] and not runs["deterministic"]:
            continue
        if len(runs["flashinfer"]) != len(runs["deterministic"]):  # a launch failed: say so, do not drop the row
            print(f"| {m[6:]} | {t} | launches FI {len(runs['flashinfer'])} vs PI {len(runs['deterministic'])}: "
                  "incomplete, not summarized | | | | | | | | |")
            continue
        for i, x in enumerate(runs["flashinfer"][0]):
            med = {mode: {k: statistics.median(r[i][k] for r in rs) for k in ("out_tok_s", "tpot_ms", "ttft_ms")}
                   for mode, rs in runs.items()}
            a, b = med["flashinfer"], med["deterministic"]
            per = " ".join(f"{100 * (q[i]['out_tok_s'] / p[i]['out_tok_s'] - 1):+.1f}"
                           for p, q in zip(runs["flashinfer"], runs["deterministic"]))
            print(f"| {m[6:]} | {t} | {x['in']}/{x['out']} c{x['concurrency']} | {a['out_tok_s']:.0f} | {b['out_tok_s']:.0f} | "
                  f"{100 * (b['out_tok_s'] / a['out_tok_s'] - 1):+.1f}% | {per} | {a['tpot_ms']:.2f} | {b['tpot_ms']:.2f} | "
                  f"{a['ttft_ms']:.0f} | {b['ttft_ms']:.0f} |")
