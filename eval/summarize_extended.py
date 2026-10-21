"""Summarize eval/e2e_extended.sh: per workload, median and min-max over launches.
Usage: python eval/summarize_extended.py results/e2e_extended"""
import glob, json, statistics, sys

d = sys.argv[1]
for model in ["Qwen3-1.7B", "Qwen3-8B"]:
    runs = {m: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{model}.{m}.L*.json"))] for m in ("flashinfer", "deterministic")}
    if not runs["flashinfer"]:
        continue
    n = len(runs["flashinfer"])
    print(f"### {model} ({n} interleaved launches per mode)\n")
    print("| workload (in/out/conc[/shared]) | metric | FI median [min–max] | PI median [min–max] | Δ median |")
    print("|---|---|---|---|---|")
    for i, w in enumerate(runs["flashinfer"][0]):
        name = f"{w['in']}/{w['out']}/c{w['concurrency']}" + (f"/{w['shared']}" if w.get("shared") else "")
        prefill = w["out"] <= 16
        metrics = [("req_s", "req/s")] if prefill else [("out_tok_s", "out tok/s")]
        metrics.append(("ttft_ms", "TTFT ms"))
        for key, label in metrics:
            vals = {m: [r[i][key] for r in runs[m]] for m in runs}
            med = {m: statistics.median(v) for m, v in vals.items()}
            fmt = lambda v: f"{statistics.median(v):.1f} [{min(v):.1f}–{max(v):.1f}]"
            print(f"| {name} | {label} | {fmt(vals['flashinfer'])} | {fmt(vals['deterministic'])} | "
                  f"{100 * (med['deterministic'] / med['flashinfer'] - 1):+.1f}% |")
    print()
