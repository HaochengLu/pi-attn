"""Cross-GPU GSM8K comparison: for each serving mode and client concurrency, how many of
the generations differ between two GPUs' runs of eval/accuracy.sh on the same model.
Usage: python eval/xgpu_accuracy.py results/accuracy/<model> results/a800/accuracy/<model>"""
import sys

from compare_accuracy import load

a, b = sys.argv[1:3]
print(f"{a} vs {b}")
print("| mode | client concurrency | items | generations differing between GPUs |")
print("|---|---|---|---|")
for mode in ("flashinfer", "deterministic"):
    for c in (1, 16):
        x, y = load(a, f"{mode}.c{c}"), load(b, f"{mode}.c{c}")
        assert x.keys() == y.keys()
        print(f"| {mode} | {c} | {len(x)} | {sum(x[k][0] != y[k][0] for k in x)} |")
