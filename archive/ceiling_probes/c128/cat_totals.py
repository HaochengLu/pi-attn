"""GPU time per kernel category and per kernel name over the last `window` seconds of two
traces (probe 19); the window must exclude server start-up, where the PI tuner runs. Usage: python cat_totals.py a.sqlite b.sqlite [window_s]"""
import collections, sqlite3, sys


def category(name):  # as scripts/nsys_compare.py
    n = name.lower()
    if "argmax" in n:
        return "argmax"
    if "gemv" in n or "gemm" in n or "cutlass" in n:
        return "projections"
    if "decode" in n or "mergestates" in n or "pi_attn" in n or "prefill" in n:
        return "attention"
    return "other kernels"

def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    rows = sorted(db.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"))
    gaps = [i for i in range(1, len(rows)) if rows[i][0] - rows[i - 1][1] > 1e9]
    rows = [(s, e, names[n]) for s, e, n in rows[gaps[-1]:]]
    span = (max(e for _, e, _ in rows) - min(s for s, _, _ in rows)) / 1e3
    cat, name = collections.Counter(), collections.Counter()
    for s, e, n in rows:
        cat[category(n)] += (e - s) / 1e3
        name[n[:90]] += (e - s) / 1e3
    return span, cat, name

(sa, ca, na), (sb, cb, nb) = load(sys.argv[1]), load(sys.argv[2])
print(f"span {sa:.0f} -> {sb:.0f} us ({(sb / sa - 1) * 100:+.1f}%); GPU us (a -> b)")
for k in ("projections", "attention", "other kernels", "argmax"):
    print(f"  {k:14s} {ca[k]:10.0f} -> {cb[k]:10.0f}  {cb[k] - ca[k]:+9.0f}")
print(f"  {'busy':14s} {sum(ca.values()):10.0f} -> {sum(cb.values()):10.0f}")
for n in sorted(set(na) | set(nb), key=lambda n: -max(na[n], nb[n]))[:16]:
    print(f"  {na[n]:9.0f} -> {nb[n]:9.0f}  {n}")
