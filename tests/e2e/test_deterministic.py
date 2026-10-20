"""End-to-end checks for a server started with --deterministic.

Skipped unless SGLANG_E2E_DETERMINISTIC=1 (and SGLANG_E2E_BASE_URL points at it)."""
import concurrent.futures as cf
import os
import random
import unittest

import requests

BASE = os.environ.get("SGLANG_E2E_BASE_URL", "http://127.0.0.1:8000/v1")


@unittest.skipUnless(os.environ.get("SGLANG_E2E_DETERMINISTIC") == "1", "server not in --deterministic mode")
class DeterministicModeTest(unittest.TestCase):
    def complete(self, prompt, **extra):
        body = {"prompt": prompt, "max_tokens": 48, "temperature": 0, "ignore_eos": True,
                "return_token_ids": True, **extra}
        response = requests.post(BASE + "/completions", json=body, timeout=300)
        response.raise_for_status()
        return response.json()

    def test_token_ids_round_trip(self):
        out = self.complete([785, 6722, 315, 9625, 374])
        self.assertEqual(out["prompt_token_ids"], [785, 6722, 315, 9625, 374])
        self.assertEqual(len(out["choices"][0]["token_ids"]), 48)

    def test_seed_is_echoed_and_reproducible(self):
        a = self.complete("Tell me a story.", temperature=0.9, top_p=0.9)
        b = self.complete("Tell me a story.", temperature=0.9, top_p=0.9, seed=a["seed"])
        self.assertEqual(a["choices"][0]["token_ids"], b["choices"][0]["token_ids"])
        c = self.complete("Tell me a story.", temperature=0.9, top_p=0.9, seed=-5)
        self.assertEqual(c["seed"], (1 << 64) - 5)

    def test_outputs_do_not_depend_on_concurrent_traffic(self):
        rng = random.Random(0)
        prompts = [[rng.randrange(1000, 100000) for _ in range(rng.choice([7, 90, 700]))] for _ in range(8)]
        alone = [self.complete(p, temperature=0.8, seed=i)["choices"][0]["token_ids"] for i, p in enumerate(prompts)]
        with cf.ThreadPoolExecutor(16) as pool:
            noise = [pool.submit(self.complete, [rng.randrange(1000, 100000) for _ in range(300)]) for _ in range(8)]
            busy = list(pool.map(lambda ip: self.complete(ip[1], temperature=0.8, seed=ip[0])["choices"][0]["token_ids"],
                                 enumerate(prompts)))
            for future in noise:
                future.result()
        self.assertEqual(alone, busy)

    def test_decode_matches_reprefill(self):
        prompt = [rng for rng in range(2000, 2100)]
        full = self.complete(prompt)["choices"][0]["token_ids"]
        tail = self.complete(prompt + full[:20], max_tokens=28)["choices"][0]["token_ids"]
        self.assertEqual(full[20:], tail)


if __name__ == "__main__":
    unittest.main()
