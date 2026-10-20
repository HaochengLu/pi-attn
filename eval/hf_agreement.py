"""Numerical sanity of a served engine against HF transformers fp32.

For each prompt, the engine generates greedily; HF fp32 then scores the same
token sequence (teacher forcing). We report how often the engine's token equals
the fp32 argmax, and the fp32 logit margin at disagreements (small margins mean
near-ties that bf16 inference legitimately flips)."""
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


def generate(url, prompt, n):
    r = requests.post(url + "/v1/completions", json={"prompt": prompt, "max_tokens": n, "temperature": 0,
                                                     "ignore_eos": True}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--urls", nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()
    report = {}
    for url in a.urls:
        agree = total = 0
        margins = []
        for p in PROMPTS:
            text = generate(url, p, a.tokens)
            pids = tok(p, add_special_tokens=False).input_ids
            ids = tok(p + text, add_special_tokens=False).input_ids
            if ids[: len(pids)] != pids:
                continue
            with torch.no_grad():
                logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
            for t in range(len(pids), len(ids)):
                row = logits[t - 1]
                top2 = row.topk(2)
                chosen = ids[t]
                total += 1
                if chosen == top2.indices[0].item():
                    agree += 1
                else:
                    margins.append((top2.values[0] - row[chosen]).item())
        report[url] = {"tokens": total, "agree_fp32_argmax": agree / max(total, 1),
                       "disagreements": len(margins),
                       "max_margin_at_disagreement": max(margins) if margins else 0.0}
        print(url, json.dumps(report[url]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
