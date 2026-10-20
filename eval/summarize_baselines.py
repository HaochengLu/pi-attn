"""Summarize eval/baselines.sh: detcheck mismatches and greedy throughput per engine.
Usage: python eval/summarize_baselines.py results/baselines/<model>"""
import json, os, sys

d = sys.argv[1]
ENG = [("rust", "llm_infer_rust (FlashInfer)"), ("rust-det", "llm_infer_rust --deterministic (PI)"),
       ("sglang", "SGLang 0.5.5"), ("sglang-det", "SGLang 0.5.5 --enable-deterministic-inference"),
       ("vllm", "vLLM 0.11.2"), ("vllm-det", "vLLM 0.11.2 VLLM_BATCH_INVARIANT=1")]
loads = {}
print("| engine | batch c4/c16 | path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |")
print("|---|---|---|---|---|---|")
for k, name in ENG:
    p = f"{d}/{k}.detcheck.json"
    if not os.path.exists(p):
        print(f"| {name} | not run | | | | |"); continue
    r = json.load(open(p))
    b = "/".join(str(x["mismatched"]) for x in r["batch"])
    s = "/".join(str(x["mismatched"]) for x in r["sample"]["rows"])
    pre = f"{r['prefix']['full_hit']}/{r['prefix']['partial_hit']} (of {r['prefix']['requests']})"
    print(f"| {name} | {b} (of 16) | {r['path']['mismatched']}/{r['path']['requests']} | {r['repeat']['unique_completions']} | {s} (of 16) | {pre} |")
print()
rows = {k: json.load(open(f"{d}/{k}.t0.json")) for k, _ in ENG if os.path.exists(f"{d}/{k}.t0.json")}
if rows:
    wl = [f"{x['in']}/{x['out']} c{x['concurrency']}" for x in next(iter(rows.values()))]
    print("| engine (greedy out tok/s) | " + " | ".join(wl) + " |")
    print("|---" * (len(wl) + 1) + "|")
    for k, name in ENG:
        if k in rows:
            print(f"| {name} | " + " | ".join(f"{x['out_tok_s']:.0f}" for x in rows[k]) + " |")
