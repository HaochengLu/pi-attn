"""Compare HASH / CASE lines of two xc_run logs: per method, how many cases are bit-identical."""
import sys, collections
def load(p):
    d = {}
    for l in open(p):
        t = l.split()
        if t[:1] == ["HASH"]:
            d[(t[1], t[2], t[3])] = tuple(t[4:])
        elif t[:1] == ["CASE"]:
            d[(t[1], t[2], "raw-inputs")] = tuple(t[4:])
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
assert a.keys() == b.keys(), set(a) ^ set(b)
res = collections.defaultdict(lambda: [0, 0, []])
for k in a:
    r = res[k[2]]
    r[1] += 1
    if a[k] == b[k]:
        r[0] += 1
    else:
        r[2].append(f"{k[0]} {k[1]}")
for m, (same, n, diff) in res.items():
    print(f"{m:26s} identical {same:2d}/{n:2d}" + (f"   differ: {', '.join(diff)}" if diff and same else ""))
