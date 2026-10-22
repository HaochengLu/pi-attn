"""Numerical sanity of a served engine against HF transformers fp32.

For each prompt, the engine generates greedily from client-tokenized prompt ids and
returns its completion token ids; HF fp32 then scores exactly that id sequence
(teacher forcing). We report how often the engine's token equals
the fp32 argmax, and the fp32 logit margin at disagreements (small margins mean
near-ties that bf16 inference legitimately flips).

Two phases, so the fp32 model never shares the GPU with a server:
  hf_agreement.py gen   --url http://host:port --model /path --out gen.json
  hf_agreement.py score --model /path --gens a.json b.json [--device cpu] --out report.json"""
import argparse
import json
import sys

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Write a Python function that checks whether a number is prime.\n",
    "Explain the difference between TCP and UDP in simple terms.",
    "Once upon a time in a small village,",
    "List three causes of the French Revolution:",
    "Q: What is 17 * 23? A:",
    "The quick brown fox",
    "In machine learning, overfitting means",
    "Translate to French: I would like a cup of coffee.",
    "def fibonacci(n):",
]


def generate(url, ids, n):
    r = requests.post(url + "/v1/completions", json={"prompt": ids, "max_tokens": n, "temperature": 0,
                                                     "ignore_eos": True, "return_token_ids": True}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["token_ids"]


def gen(a):
    url = a.url.rstrip("/")
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok(p, add_special_tokens=False).input_ids for p in PROMPTS]
    json.dump({"url": url, "tokens": a.tokens, "prompt_ids": prompts,
               "completion_ids": [generate(url, ids, a.tokens) for ids in prompts]}, open(a.out, "w"), indent=1)


def score(a):
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(a.device).eval()
    report = {}
    for path in a.gens:
        agree = total = 0
        margins = []
        g = json.load(open(path))
        for pids, cids in zip(g["prompt_ids"], g["completion_ids"]):
            ids = pids + cids
            with torch.no_grad():
                logits = model(torch.tensor([ids], device=a.device)).logits[0].float()
            for t in range(len(pids), len(ids)):
                row = logits[t - 1]
                top2 = row.topk(2)
                chosen = ids[t]
                total += 1
                if chosen == top2.indices[0].item():
                    agree += 1
                else:
                    margins.append((top2.values[0] - row[chosen]).item())
        report[path] = {"tokens": total, "agree_fp32_argmax": agree / max(total, 1),
                        "disagreements": len(margins),
                        "max_margin_at_disagreement": max(margins) if margins else 0.0}
        print(path, json.dumps(report[path]), flush=True)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", required=True)
    g.add_argument("--model", required=True, help="tokenizer for the prompts")
    g.add_argument("--tokens", type=int, default=128)
    g.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--gens", nargs="+", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out")
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else score(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
