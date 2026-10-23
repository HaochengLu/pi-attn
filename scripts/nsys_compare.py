"""Per-step comparison of two batch-1 greedy decode traces from scripts/nsys_decode.sh (e.g.
FlashInfer mode vs --deterministic): GPU time by kernel category, the time between kernels,
and the mean time of each projection (qkv, o, gate_up, down per layer, then the LM head).
Steps are delimited by the argmax that ends each one, in the trace's last `window` seconds.
Node tracing lengthens the gaps between kernels, so the totals are for attribution only;
step times without tracing are in results/e2e_matrix.
Usage: python scripts/nsys_compare.py <a.sqlite> <b.sqlite> [window_s]"""
import collections
import sqlite3
import sys

PROJ = ("qkv", "o", "gate_up", "down")


def category(name):
    n = name.lower()
    if "argmax" in n:
        return "argmax"
    if "gemv" in n or "gemm" in n or "cutlass" in n:
        return "projections"
    if "decode" in n or "mergestates" in n or "pi_attn" in n:
        return "attention"
    return "other kernels"


def load(path, window):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    rows = sorted(db.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"))
    t_end = max(r[1] for r in rows)
    rows = [(s, e, names[n]) for s, e, n in rows if s >= t_end - window * 1e9]
    marks = [s for s, _, n in rows if category(n) == "argmax"]
    steps, cat, proj = 0, collections.Counter(), collections.defaultdict(list)
    for a, b in zip(marks, marks[1:]):
        step = [r for r in rows if a < r[0] <= b]
        p = [e - s for s, e, n in step if category(n) == "projections"]
        if len(p) % 4 != 1:  # a prefill or a step cut by the window: not one decode step
            continue
        steps += 1
        for s, e, n in step:
            cat[category(n)] += e - s
        cat["between kernels"] += (b - a) - sum(e - s for s, e, _ in step)
        for i, t in enumerate(p[:-1]):
            proj[PROJ[i % 4]].append(t)
        proj["lm_head"].append(p[-1])
    return steps, {k: v / steps / 1e3 for k, v in cat.items()}, {k: sum(v) / len(v) / 1e3 for k, v in proj.items()}


window = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
(na, ca, pa), (nb, cb, pb) = load(sys.argv[1], window), load(sys.argv[2], window)
print(f"steps: {na} vs {nb}; us per step (a -> b, b - a)")
for k in ("projections", "attention", "other kernels", "argmax", "between kernels"):
    print(f"  {k:16s} {ca.get(k, 0):8.1f} -> {cb.get(k, 0):8.1f}  {cb.get(k, 0) - ca.get(k, 0):+7.1f}")
ta, tb = sum(ca.values()), sum(cb.values())
print(f"  {'step':16s} {ta:8.1f} -> {tb:8.1f}  {tb - ta:+7.1f}")
print("us per launch: " + "  ".join(f"{k} {pa[k]:.2f} -> {pb[k]:.2f}" for k in (*PROJ, "lm_head")))
