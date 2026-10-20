#!/usr/bin/env python3
"""detcheck: black-box bitwise-determinism checks for LLM serving engines.

Every check sends greedy (temperature 0, ignore_eos) requests as token ids to an
OpenAI-compatible /v1/completions endpoint and compares completion token ids
against a reference run of the same request. A deterministic engine must return
identical tokens no matter how the server batches, caches or chunks work.

  batch    each prompt alone vs. inside concurrent mixed-length traffic
  prefix   same request with its prompt uncached vs. (partially) prefix-cached
  shared   many concurrent requests sharing one long cached prefix vs. each alone
  path     continuation through decode vs. re-prefilling the generated tokens
  sample_path  the same with seeded sampling: holds only if the engine keys its
           sampling stream by (seed, position), not by request start
  repeat   one prompt sent many times under varying background load
  sample   seeded temperature sampling, alone vs. under concurrent traffic
  save / compare   store greedy and seeded-sample references, then compare a
           later run against them: across server restarts and across server
           configurations (chunk size, CUDA graphs, batch limits, overlap)

Usage: detcheck.py --url http://host:port --tokenizer /path/to/model [--checks ...]
Requires the server to return completion token ids ("return_token_ids": true,
supported by vLLM and llm_infer_rust) or SGLang's native /generate (--api sglang).
"""
import argparse
import concurrent.futures as cf
import json
import random
import sys
import time

import requests


class Server:
    def __init__(self, url, api, timeout=1200):
        self.url, self.api, self.timeout = url.rstrip("/"), api, timeout

    def generate(self, ids, max_tokens, sampling=None):
        sp = sampling or {"temperature": 0}
        if self.api == "sglang":
            if "seed" in sp:  # SGLang names the per-request sampling seed sampling_seed
                sp = {**{k: v for k, v in sp.items() if k != "seed"}, "sampling_seed": sp["seed"]}
            r = requests.post(self.url + "/generate", json={
                "input_ids": ids, "sampling_params": {**sp, "max_new_tokens": max_tokens,
                                                      "ignore_eos": True}}, timeout=self.timeout)
            r.raise_for_status()
            return list(r.json()["output_ids"])
        r = requests.post(self.url + "/v1/completions", json={
            "model": self.model(), "prompt": ids, "max_tokens": max_tokens, **sp,
            "ignore_eos": True, "return_token_ids": True}, timeout=self.timeout)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        if "token_ids" not in choice:
            raise SystemExit("server does not return completion token ids (return_token_ids)")
        return list(choice["token_ids"])

    _model = None

    def model(self):
        if self._model is None:
            try:
                self._model = requests.get(self.url + "/v1/models", timeout=30).json()["data"][0]["id"]
            except Exception:  # noqa: BLE001
                self._model = "default"
        return self._model

    def flush(self, how, vocab, rng):
        """Evict cached prefixes: engine endpoint, or flood the KV pool with junk."""
        if how == "sglang":
            requests.post(self.url + "/flush_cache", timeout=60)
        elif how == "vllm":
            requests.post(self.url + "/reset_prefix_cache", timeout=60)
        elif how.startswith("flood:"):
            total = int(how.split(":")[1])
            n = max(1, total // 2048)
            with cf.ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda _: self.generate([rng.randrange(1000, vocab) for _ in range(2048)], 1), range(n)))


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def make_prompts(tok, rng, n, lengths):
    """Prompts of the requested token lengths cut from a varied text corpus."""
    base = [
        "Explain how a transformer language model generates text, step by step.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "List the main differences between TCP and UDP and when to use each.",
        "Derive the formula for the sum of a geometric series and give an example.",
        "Describe the water cycle to a ten-year-old.",
        "What are the trade-offs between paged attention and contiguous KV caches?",
        "Summarize the causes and consequences of the industrial revolution.",
        "Write a Python function that merges two sorted lists, with tests.",
    ]
    corpus = tok(" ".join(base * 400), add_special_tokens=False).input_ids
    out = []
    for i in range(n):
        L = lengths[i % len(lengths)]
        start = rng.randrange(0, len(corpus) - L - 1)
        out.append(tok(base[i % len(base)], add_special_tokens=False).input_ids + corpus[start:start + L])
    return out


def check_batch(srv, prompts, T, rng, conc, noise_vocab):
    ref = [srv.generate(p, T) for p in prompts]
    rows = []
    for c in conc:
        order = list(range(len(prompts)))
        rng.shuffle(order)
        noise = [[rng.randrange(1000, noise_vocab) for _ in range(rng.choice([16, 300, 1500]))] for _ in range(c)]
        with cf.ThreadPoolExecutor(c * 2) as ex:
            futs = {ex.submit(srv.generate, prompts[i], T): i for i in order}
            for nz in noise:
                ex.submit(srv.generate, nz, rng.choice([8, 64, 256]))
            got = {futs[f]: f.result() for f in cf.as_completed(futs)}
        div = [first_divergence(ref[i], got[i]) for i in range(len(prompts))]
        bad = [d for d in div if d is not None]
        rows.append({"concurrency": c, "requests": len(prompts), "mismatched": len(bad),
                     "first_divergence_tokens": sorted(bad)[:10]})
    return rows


def check_prefix(srv, prompts, T, rng, flush, vocab):
    rows = {"full_hit": 0, "partial_hit": 0, "requests": len(prompts)}
    for p in prompts:
        srv.flush(flush, vocab, rng)
        cold = srv.generate(p, T)                       # prompt uncached
        warm = srv.generate(p, T)                       # whole prompt cached (radix)
        if first_divergence(cold, warm) is not None:
            rows["full_hit"] += 1
        srv.flush(flush, vocab, rng)
        cut = len(p) * 2 // 3
        srv.generate(p[:cut] + [rng.randrange(1000, vocab)], 1)  # cache a shared prefix only
        part = srv.generate(p, T)
        if first_divergence(cold, part) is not None:
            rows["partial_hit"] += 1
    return rows


def check_shared(srv, prompts, T, rng, n, vocab):
    """n requests = one long shared prefix + distinct short suffixes, sent at once."""
    prefix = max(prompts, key=len)
    reqs = [prefix + [rng.randrange(1000, vocab) for _ in range(rng.randrange(1, 64))] for _ in range(n)]
    ref = [srv.generate(r, T) for r in reqs]
    with cf.ThreadPoolExecutor(n) as ex:
        got = list(ex.map(lambda r: srv.generate(r, T), reqs))
    bad = [d for d in (first_divergence(a, b) for a, b in zip(ref, got)) if d is not None]
    return {"requests": n, "prefix_tokens": len(prefix), "mismatched": len(bad),
            "first_divergence_tokens": sorted(bad)[:10]}


def check_path(srv, prompts, T, rng, flush, vocab, sampling=None):
    rows = {"requests": 0, "mismatched": 0, "first_divergence_tokens": []}
    for i, p in enumerate(prompts):
        sp = sampling(i) if sampling else None
        full = srv.generate(p, T, sp)
        for k in (1, T // 4, T // 2):
            srv.flush(flush, vocab, rng)
            cont = srv.generate(p + full[:k], T - k, sp)  # tokens k.. via prefill of the generated prefix
            rows["requests"] += 1
            d = first_divergence(full[k:], cont)
            if d is not None:
                rows["mismatched"] += 1
                rows["first_divergence_tokens"].append(k + d)
    return rows


def check_repeat(srv, prompt, T, rng, n, vocab):
    outs = set()
    with cf.ThreadPoolExecutor(32) as ex:
        futs = []
        for i in range(n):
            futs.append(ex.submit(srv.generate, prompt, T))
            for _ in range(rng.randrange(0, 4)):
                ex.submit(srv.generate, [rng.randrange(1000, vocab) for _ in range(rng.randrange(8, 800))], rng.choice([4, 32, 128]))
        for f in futs:
            outs.add(tuple(f.result()))
    return {"samples": n, "unique_completions": len(outs)}


def check_sample(srv, prompts, T, rng, conc, vocab):
    sp = lambda i: {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 1000 + i}
    ref = [srv.generate(p, T, sp(i)) for i, p in enumerate(prompts)]
    rows = []
    for c in conc:
        with cf.ThreadPoolExecutor(c * 2) as ex:
            futs = {ex.submit(srv.generate, p, T, sp(i)): i for i, p in enumerate(prompts)}
            for _ in range(c):
                ex.submit(srv.generate, [rng.randrange(1000, vocab) for _ in range(rng.choice([16, 300, 1500]))],
                          rng.choice([8, 64, 256]), {"temperature": 1.0, "seed": rng.randrange(1 << 30)})
            got = {futs[f]: f.result() for f in cf.as_completed(futs)}
        bad = [first_divergence(ref[i], got[i]) for i in range(len(prompts))]
        bad = [d for d in bad if d is not None]
        rows.append({"concurrency": c, "requests": len(prompts), "mismatched": len(bad),
                     "first_divergence_tokens": sorted(bad)[:10]})
    distinct = len({tuple(r) for r in ref})
    return {"rows": rows, "distinct_reference_samples": distinct}


def references(srv, prompts, T, conc):
    sp = lambda i: {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 7000 + i}
    with cf.ThreadPoolExecutor(conc) as ex:
        greedy = list(ex.map(lambda p: srv.generate(p, T), prompts))
        sampled = list(ex.map(lambda ip: srv.generate(ip[1], T, sp(ip[0])), enumerate(prompts)))
    return {"greedy": greedy, "sampled": sampled}


def compare_refs(saved, got):
    rows = {}
    for kind in ("greedy", "sampled"):
        div = [first_divergence(a, b) for a, b in zip(saved[kind], got[kind])]
        bad = [d for d in div if d is not None]
        rows[kind] = {"requests": len(div), "mismatched": len(bad), "first_divergence_tokens": sorted(bad)[:10]}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer path (used only to build prompts)")
    ap.add_argument("--api", choices=["openai", "sglang"], default="openai")
    ap.add_argument("--checks", nargs="+", default=["batch", "prefix", "shared", "path", "sample_path", "repeat", "sample"])
    ap.add_argument("--shared", type=int, default=32, help="concurrent requests in the shared-prefix check")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--lengths", type=int, nargs="+", default=[24, 300, 1100, 2600])
    ap.add_argument("--tokens", type=int, default=256, help="completion length")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--flush", default="flood:70000", help="sglang | vllm | flood:<tokens> | none")
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", help="write greedy/sampled references (run under --concurrency[-1])")
    ap.add_argument("--compare", help="compare against references written by --save")
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    vocab = min(len(tok), 150000)
    rng = random.Random(a.seed)
    srv = Server(a.url, a.api)
    prompts = make_prompts(tok, rng, a.prompts, a.lengths)
    report = {"url": a.url, "tokens": a.tokens, "prompts": a.prompts, "lengths": a.lengths}
    t0 = time.time()
    if "batch" in a.checks:
        report["batch"] = check_batch(srv, prompts, a.tokens, rng, a.concurrency, vocab)
        print("batch ", json.dumps(report["batch"]), flush=True)
    if "prefix" in a.checks:
        report["prefix"] = check_prefix(srv, prompts[: max(4, a.prompts // 2)], a.tokens, rng, a.flush, vocab)
        print("prefix", json.dumps(report["prefix"]), flush=True)
    if "shared" in a.checks:
        report["shared"] = check_shared(srv, prompts, a.tokens, rng, a.shared, vocab)
        print("shared", json.dumps(report["shared"]), flush=True)
    if "path" in a.checks:
        report["path"] = check_path(srv, prompts[: max(4, a.prompts // 2)], a.tokens, rng, a.flush, vocab)
        print("path  ", json.dumps(report["path"]), flush=True)
    if "sample_path" in a.checks:
        sp = lambda i: {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "seed": 3000 + i}
        report["sample_path"] = check_path(srv, prompts[: max(4, a.prompts // 2)], a.tokens, rng, a.flush, vocab, sp)
        print("spath ", json.dumps(report["sample_path"]), flush=True)
    if "repeat" in a.checks:
        report["repeat"] = check_repeat(srv, prompts[0], a.tokens, rng, a.repeat, vocab)
        print("repeat", json.dumps(report["repeat"]), flush=True)
    if "sample" in a.checks:
        report["sample"] = check_sample(srv, prompts, a.tokens, rng, a.concurrency, vocab)
        print("sample", json.dumps(report["sample"]), flush=True)
    if a.save:
        json.dump({"prompts": prompts, "tokens": a.tokens,
                   **references(srv, prompts, a.tokens, a.concurrency[-1])}, open(a.save, "w"))
        print("saved references to", a.save, flush=True)
    if a.compare:
        saved = json.load(open(a.compare))
        got = references(srv, saved["prompts"], saved["tokens"], a.concurrency[-1])
        report["compare"] = compare_refs(saved, got)
        print("compare", json.dumps(report["compare"]), flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    failures = sum(r["mismatched"] for r in report.get("batch", [])) + report.get("path", {}).get("mismatched", 0) \
        + report.get("shared", {}).get("mismatched", 0) + report.get("sample_path", {}).get("mismatched", 0) \
        + report.get("prefix", {}).get("full_hit", 0) + report.get("prefix", {}).get("partial_hit", 0) \
        + max(0, report.get("repeat", {}).get("unique_completions", 1) - 1) \
        + sum(r["mismatched"] for r in report.get("sample", {}).get("rows", [])) \
        + sum(r["mismatched"] for r in report.get("compare", {}).values())
    print("DETERMINISTIC" if failures == 0 else f"NONDETERMINISTIC ({failures} failing comparisons)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
