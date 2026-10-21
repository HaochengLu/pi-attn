"""Group an nsys cuda_gpu_kern_sum CSV by kernel category; compare several traces.
Usage: python eval/kernel_categories.py a.csv b.csv ..."""
import csv, re, sys

CATS = [("GEMM", r"gemm|gemv|matmul|cutlass|sm\d+_xmma|ampere_|cublas|splitK|_mm_"),
        ("attention", r"attention|pi_attn|flashinfer.*(Prefill|Decode)|BatchPrefill|BatchDecode|fmha|flash"),
        ("sampling", r"sampl|softmax|topk|top_k|top_p|argmax|multinomial|sort|radix|cub::"),
        ("norm", r"norm"),
        ("reduction", r"reduce|sum_|mean")]


def cat(name):
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "other"


tables = []
for path in sys.argv[1:]:
    tot = {}
    for row in csv.DictReader(open(path)):
        t = float(row["Total Time (ns)"]) / 1e6
        tot[cat(row["Name"])] = tot.get(cat(row["Name"]), 0.0) + t
    tables.append((path.rsplit("/", 1)[-1].replace("_cuda_gpu_kern_sum.csv", ""), tot))
names = [c for c, _ in CATS] + ["other"]
print("| kernel time (ms) | " + " | ".join(t for t, _ in tables) + " |")
print("|---" * (len(tables) + 1) + "|")
for c in names + ["total"]:
    vals = [sum(t.values()) if c == "total" else t.get(c, 0.0) for _, t in tables]
    print(f"| {c} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
