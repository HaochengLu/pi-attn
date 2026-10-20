"""Serving benchmark (streaming): output throughput, TTFT, TPOT, ITL.

Requests are token-id prompts of fixed input/output length (ignore_eos), sent
with a fixed client concurrency; token counts come from the server's usage
field, never from re-tokenized text. Each configuration runs `--repeats` times
and reports the median of each metric."""
import argparse
import concurrent.futures as cf
import json
import random
import statistics
import time

import requests


SAMPLING = {"temperature": 0}


def one(url, ids, out_len, seed=0):
    t0 = time.perf_counter()
    r = requests.post(url + "/v1/completions", json={"model": "default", "prompt": ids, "max_tokens": out_len, **SAMPLING, "seed": seed,
                                                     "ignore_eos": True, "stream": True,
                                                     "stream_options": {"include_usage": True}},
                      stream=True, timeout=1200)
    r.raise_for_status()
    stamps, usage = [], None
    for line in r.iter_lines():
        if not line.startswith(b"data:"):
            continue
        body = line[5:].strip()
        if body == b"[DONE]":
            break
        msg = json.loads(body)
        if msg.get("usage"):
            usage = msg["usage"]
        if msg.get("choices") and msg["choices"][0].get("text"):
            stamps.append(time.perf_counter())
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens", len(stamps))
    ttft = (stamps[0] - t0) if stamps else end - t0
    tpot = (end - stamps[0]) / max(n - 1, 1) if stamps else 0.0
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft": ttft, "tpot": tpot, "itl": itl, "tokens": n, "e2e": end - t0}


def run(url, in_len, out_len, conc, num, seed, shared=0):
    """shared: the first `shared` input tokens are common to every request (prefix-cache hits)."""
    rng = random.Random(seed)
    head = [rng.randrange(1000, 150000) for _ in range(shared)]
    prompts = [head + [rng.randrange(1000, 150000) for _ in range(in_len - shared)] for _ in range(num)]
    if shared:
        one(url, head + [1000], 1)  # cache the shared prefix before timing
    one(url, prompts[0][:16], 4)  # warm-up
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(lambda ip: one(url, ip[1], out_len, ip[0]), enumerate(prompts)))
    wall = time.perf_counter() - t0
    tokens = sum(r["tokens"] for r in res)
    itl = [x for r in res for x in r["itl"]]
    return {"out_tok_s": tokens / wall, "req_s": num / wall, "ttft_ms": statistics.mean(r["ttft"] for r in res) * 1e3,
            "tpot_ms": statistics.mean(r["tpot"] for r in res) * 1e3,
            "itl_ms": statistics.mean(itl) * 1e3 if itl else 0.0,
            "p99_ttft_ms": sorted(r["ttft"] for r in res)[int(0.99 * (len(res) - 1))] * 1e3,
            "e2e_ms": statistics.mean(r["e2e"] for r in res) * 1e3, "tokens": tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--workloads", nargs="+", default=["64:32:1", "64:32:8", "64:32:32", "512:256:8", "2048:128:8", "1024:512:32"],
                    help="input:output:concurrency[:shared prefix tokens]")
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: sample with top_p 0.95 and per-request seeds")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.temperature > 0:
        SAMPLING.update({"temperature": a.temperature, "top_p": 0.95})
    rows = []
    for w in a.workloads:
        i, o, c, *sh = map(int, w.split(":"))
        sh = sh[0] if sh else 0
        reps = [run(a.url.rstrip("/"), i, o, c, max(a.num, 2 * c), seed=r, shared=sh) for r in range(a.repeats)]
        med = {k: statistics.median(r[k] for r in reps) for k in reps[0]}
        med.update({"in": i, "out": o, "concurrency": c, "shared": sh, "repeats": a.repeats,
                    "out_tok_s_all": [round(r["out_tok_s"], 1) for r in reps]})
        rows.append(med)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in med.items()}), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
