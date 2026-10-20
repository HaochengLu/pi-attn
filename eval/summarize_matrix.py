import json,sys,os
d=sys.argv[1]
rows=[]
print("| model | T | workload | FI tok/s | PI tok/s | Δ thpt | FI TPOT | PI TPOT | FI TTFT | PI TTFT |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in ["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B"]:
  for t in ["0","0.8"]:
    a=json.load(open(f"{d}/{m}.flashinfer.t{t}.json")); b=json.load(open(f"{d}/{m}.deterministic.t{t}.json"))
    for x,y in zip(a,b):
      w=f"{x['in']}/{x['out']} c{x['concurrency']}"
      print(f"| {m[6:]} | {t} | {w} | {x['out_tok_s']:.0f} | {y['out_tok_s']:.0f} | {100*(y['out_tok_s']/x['out_tok_s']-1):+.1f}% | {x['tpot_ms']:.2f} | {y['tpot_ms']:.2f} | {x['ttft_ms']:.0f} | {y['ttft_ms']:.0f} |")
