"""Summarize eval/baselines.sh: detcheck mismatches and greedy throughput per engine.
Usage: python eval/summarize_baselines.py results/baselines/<model>"""
import json, os, sys

d = sys.argv[1]
ENG = [("rust", "llm_infer_rust (FlashInfer)"), ("rust-det", "llm_infer_rust --deterministic (PI)"),
       ("sglang", "SGLang 0.5.5"), ("sglang-det", "SGLang 0.5.5 deterministic (FlashInfer)"),
       ("sglang-det-triton", "SGLang 0.5.5 deterministic (Triton)"),
       ("vllm", "vLLM 0.11.2"), ("vllm-det", "vLLM 0.11.2 VLLM_BATCH_INVARIANT=1"),
       ("vllm17", "vLLM 0.17.1"), ("vllm17-det", "vLLM 0.17.1 VLLM_BATCH_INVARIANT=1")]
loads = {}
print("| engine | batch c4/c16 | shared prefix x32 | path | seeded path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |")
print("|---|---|---|---|---|---|---|---|")
for k, name in ENG:
    p = f"{d}/{k}.detcheck.json"
    if not os.path.exists(p):
        log = f"{d}/{k}.detcheck.log"
        stalled = os.path.exists(log) and "ReadTimeout" in open(log).read()
        print(f"| {name} | {'server stalled (request timed out after 1200 s)' if stalled else 'not run'} | | | | | | |")
        continue
    r = json.load(open(p))
    b = "/".join(str(x["mismatched"]) for x in r["batch"])
    s = "/".join(str(x["mismatched"]) for x in r["sample"]["rows"])
    pre = f"{r['prefix']['full_hit']}/{r['prefix']['partial_hit']} (of {r['prefix']['requests']})"
    sh = r.get("shared"); spth = r.get("sample_path")
    sh = f"{sh['mismatched']}/{sh['requests']}" if sh else "-"
    spth = f"{spth['mismatched']}/{spth['requests']}" if spth else "-"
    print(f"| {name} | {b} (of 16) | {sh} | {r['path']['mismatched']}/{r['path']['requests']} | {spth} | {r['repeat']['unique_completions']} | {s} (of 16) | {pre} |")
print()
rows = {k: json.load(open(f"{d}/{k}.t0.json")) for k, _ in ENG if os.path.exists(f"{d}/{k}.t0.json")}
if rows:
    wl = [f"{x['in']}/{x['out']} c{x['concurrency']}" for x in next(iter(rows.values()))]
    print("| engine (greedy out tok/s) | " + " | ".join(wl) + " |")
    print("|---" * (len(wl) + 1) + "|")
    for k, name in ENG:
        if k in rows:
            print(f"| {name} | " + " | ".join(f"{x['out_tok_s']:.0f}" for x in rows[k]) + " |")
