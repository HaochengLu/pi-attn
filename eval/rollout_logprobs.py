"""Rollout log-probabilities from the serving engine vs a trainer's teacher-forced forward.

gen:   GSM8K questions (chat template, no thinking) go to a running engine at client concurrency
       16, sampled at temperature 1 with a per-request seed (and a greedy subset); the engine returns
       each completion's token ids and their log-probabilities (`logprobs: 0`).
score: with the server stopped, two trainers score exactly those sequences (prompt + completion):
  pi  kernels/engine_forward.py: the engine's kernels in PyTorch, each sequence alone and in two
      random packings of 2-8 sequences per micro-batch. Every completion token's log-probability
      must equal the engine's bit for bit.
  hf  HF transformers, bf16, SDPA attention, log_softmax in fp32 of the bf16 logits (a common trainer
      setup), each sequence alone and in right-padded batches of 8. Reports how many tokens match the
      engine's bits, and the importance ratio exp(trainer - engine) an RL update would see.
usage: rollout_logprobs.py gen --url URL --model DIR --out gen.json [--n 64 --tokens 256]
       rollout_logprobs.py score --model DIR --gen gen.json --out report.json"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import sys

import numpy as np
import requests
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def questions(n):
    files = sorted(glob.glob(os.path.join(ROOT, "results/accuracy/*/deterministic.c1/**/samples_gsm8k_*.jsonl"), recursive=True))
    assert files, "GSM8K samples from eval/accuracy.sh are needed for the questions"
    out = []
    for line in open(files[0]):
        out.append(json.loads(line)["doc"]["question"])
        if len(out) == n:
            break
    return out


def gen(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                       enable_thinking=False) for q in questions(a.n)]
    prompts = [p if isinstance(p, list) else p["input_ids"] for p in prompts]

    def one(i):
        greedy = i % 4 == 3
        r = requests.post(a.url.rstrip("/") + "/v1/completions", json={
            "prompt": prompts[i], "max_tokens": a.tokens, "temperature": 0.0 if greedy else 1.0, "seed": 1000 + i,
            "logprobs": 0, "return_token_ids": True}, timeout=1200)
        r.raise_for_status()
        c = r.json()["choices"][0]
        return {"prompt_ids": prompts[i], "completion_ids": c["token_ids"], "greedy": greedy,
                "logprobs": c["logprobs"]["token_logprobs"]}

    with cf.ThreadPoolExecutor(16) as pool:
        rollouts = list(pool.map(one, range(len(prompts))))
    json.dump({"model": a.model, "rollouts": rollouts}, open(a.out, "w"))
    print(f"{len(rollouts)} rollouts, {sum(len(r['completion_ids']) for r in rollouts)} completion tokens")


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def packings(n, rng):
    order = list(range(n))
    rng.shuffle(order)
    out, i = [], 0
    while i < n:
        k = rng.randint(2, 8)
        out.append(order[i:i + k])
        i += k
    return out


def compare(name, rollouts, scored, report):
    """scored[i]: the trainer's f32 log-probabilities of rollout i's completion tokens."""
    eng = np.concatenate([np.asarray(r["logprobs"], dtype=np.float32) for r in rollouts])
    tr = np.concatenate([np.asarray(s, dtype=np.float32) for s in scored])
    same = int((bits(eng) == bits(tr)).sum())
    d = tr.astype(np.float64) - eng.astype(np.float64)
    ratio = np.exp(d)
    report[name] = {"tokens": int(eng.size), "bitwise_equal": same, "mean_abs_diff": float(np.abs(d).mean()),
                    "max_abs_diff": float(np.abs(d).max()), "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                    "ratio_frac_outside_1pct": float((np.abs(ratio - 1) > 0.01).mean())}
    r = report[name]
    print(f"{name:22s}: {same}/{r['tokens']} tokens bitwise equal to the engine | |dlogp| mean {r['mean_abs_diff']:.2e} "
          f"max {r['max_abs_diff']:.2e} | ratio [{r['ratio_min']:.4f}, {r['ratio_max']:.4f}], "
          f"{100 * r['ratio_frac_outside_1pct']:.2f}% of tokens off by > 1%", flush=True)


def score(a):
    sys.path.insert(0, os.path.join(ROOT, "kernels"))
    os.chdir(os.path.join(ROOT, "kernels"))
    g = json.load(open(a.gen))
    rollouts = g["rollouts"]
    seqs = [r["prompt_ids"] + r["completion_ids"] for r in rollouts]
    starts = [len(r["prompt_ids"]) for r in rollouts]
    report = {"rollouts": len(rollouts), "greedy": sum(r["greedy"] for r in rollouts)}
    rng = random.Random(0)

    from engine_forward import EngineQwen3
    model = EngineQwen3(a.model)
    alone = [model.token_logprobs([s], [st])[0].cpu().numpy() for s, st in zip(seqs, starts)]
    compare("pi alone", rollouts, alone, report)
    for p in range(2):
        scored = [None] * len(seqs)
        for group in packings(len(seqs), rng):
            for i, lp in zip(group, model.token_logprobs([seqs[i] for i in group], [starts[i] for i in group])):
                scored[i] = lp.cpu().numpy()
        compare(f"pi packing {p}", rollouts, scored, report)
    del model
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    @torch.no_grad()
    def hf_score(group):
        n = max(len(seqs[i]) for i in group)
        ids = torch.zeros(len(group), n, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for j, i in enumerate(group):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[j, :len(seqs[i])] = 1
        logp = hf(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
        out = []
        for j, i in enumerate(group):
            t = torch.tensor(seqs[i][starts[i]:], device="cuda")
            out.append(logp[j, starts[i] - 1:len(seqs[i]) - 1].gather(1, t[:, None])[:, 0].cpu().numpy())
        return out

    compare("hf alone", rollouts, [hf_score([i])[0] for i in range(len(seqs))], report)
    scored = [None] * len(seqs)
    for b in range(0, len(seqs), 8):
        group = list(range(b, min(b + 8, len(seqs))))
        for i, lp in zip(group, hf_score(group)):
            scored[i] = lp
    compare("hf batch 8", rollouts, scored, report)
    json.dump(report, open(a.out, "w"), indent=1)
    pi_ok = all(report[k]["bitwise_equal"] == report[k]["tokens"] for k in report if k.startswith("pi"))
    print("PI TRAINER == ENGINE:", pi_ok)
    return 0 if pi_ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["gen", "score"])
    ap.add_argument("--url")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    return gen(a) if a.phase == "gen" else score(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
