"""Engine-level cross-GPU check: replay another GPU's rollouts (eval/rollout_logprobs.py gen output) against
a running deterministic engine and compare, bit for bit, every completion token id and every fp32
log-probability. Same requests as the original run: prompt token ids, seed 1000 + i, temperature 0 for the
greedy subset (i % 4 == 3) and 1 otherwise, the original max_tokens, client concurrency 16.
usage: python eval/xgpu_rollout.py --url URL --ref results/a800/rollout/Qwen3-1.7B.gen.json --out new.gen.json"""
import argparse
import concurrent.futures as cf
import json
import sys

import numpy as np
import requests


def bits(x):
    return np.asarray([np.nan if v is None else v for v in x], dtype=np.float32).view(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    rs = ref["rollouts"]

    def one(i):
        r = requests.post(a.url.rstrip("/") + "/v1/completions", json={
            "prompt": rs[i]["prompt_ids"], "max_tokens": a.tokens, "temperature": 0.0 if rs[i]["greedy"] else 1.0,
            "seed": 1000 + i, "logprobs": 0, "return_token_ids": True}, timeout=1200)
        r.raise_for_status()
        c = r.json()["choices"][0]
        return {"prompt_ids": rs[i]["prompt_ids"], "completion_ids": c["token_ids"], "greedy": rs[i]["greedy"],
                "logprobs": c["logprobs"]["token_logprobs"]}

    with cf.ThreadPoolExecutor(16) as pool:
        new = list(pool.map(one, range(len(rs))))
    json.dump({"model": ref["model"], "rollouts": new}, open(a.out, "w"))
    seq_same = sum(x["completion_ids"] == y["completion_ids"] for x, y in zip(rs, new))
    tok = sum(len(x["completion_ids"]) for x in rs)
    lp_same = sum(int((bits(x["logprobs"]) == bits(y["logprobs"])).sum())
                  for x, y in zip(rs, new) if len(x["logprobs"]) == len(y["logprobs"]))
    print(f"{len(rs)} rollouts: {seq_same} with identical token ids; {lp_same}/{tok} log-probabilities bitwise equal")
    ok = seq_same == len(rs) and lp_same == tok
    print("ROLLOUTS IDENTICAL TO REFERENCE:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
