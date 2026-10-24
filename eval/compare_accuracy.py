"""Item-level GSM8K comparison: accuracy per run and how many generations differ
between client concurrency 1 and 16 within each serving mode.
Usage: python eval/compare_accuracy.py results/accuracy/<model>"""
import glob, json, sys

def load(d, run):
    files = glob.glob(f"{d}/{run}/**/samples_gsm8k_*.jsonl", recursive=True)
    assert files, f"no samples for {run}"
    rows = [json.loads(l) for l in open(sorted(files)[-1])]
    return {r["doc_id"]: (r["resps"][0][0], r["exact_match"]) for r in rows if r["filter"] == "strict-match"}

if __name__ == "__main__":
    d = sys.argv[1]
    print(f"| mode | acc c=1 (strict) | acc c=16 | items | generations differing c1 vs c16 |")
    print("|---|---|---|---|---|")
    for mode in ["flashinfer", "deterministic"]:
        a, b = load(d, f"{mode}.c1"), load(d, f"{mode}.c16")
        diff = sum(a[k][0] != b[k][0] for k in a)
        acc = lambda s: 100 * sum(float(v[1]) for v in s.values()) / len(s)
        print(f"| {mode} | {acc(a):.2f} | {acc(b):.2f} | {len(a)} | {diff} |")
