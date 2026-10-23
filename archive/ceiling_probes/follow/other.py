import sys, sqlite3, collections
def category(name):
    n = name.lower()
    if "argmax" in n: return "argmax"
    if "gemv" in n or "gemm" in n or "cutlass" in n: return "projections"
    if "decode" in n or "mergestates" in n or "pi_attn" in n: return "attention"
    return "other kernels"
def load(path, window=0.3):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    rows = sorted(db.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"))
    t_end = max(r[1] for r in rows)
    rows = [(s, e, names[n]) for s, e, n in rows if s >= t_end - window * 1e9]
    marks = [s for s, _, n in rows if category(n) == "argmax"]
    steps=0; c=collections.Counter(); cnt=collections.Counter()
    for a, b in zip(marks, marks[1:]):
        step = [r for r in rows if a < r[0] <= b]
        p = [1 for s, e, n in step if category(n) == "projections"]
        if len(p) % 4 != 1: continue
        steps += 1
        for s, e, n in step:
            if category(n) == "other kernels":
                k = n[:70]; c[k] += e - s; cnt[k] += 1
    return {k: (v/steps/1e3, cnt[k]/steps) for k, v in c.items()}
for m in sys.argv[1:]:
    a = load(f"results/nsys/b1_{m}_fi.sqlite"); b = load(f"results/nsys/b1_{m}_pi.sqlite")
    print("==", m)
    for k in sorted(set(a)|set(b), key=lambda k: -(a.get(k,(0,0))[0]+b.get(k,(0,0))[0])):
        ta, na = a.get(k,(0,0)); tb, nb = b.get(k,(0,0))
        print(f"  {ta:7.1f} ({na:5.1f}) -> {tb:7.1f} ({nb:5.1f})  per-launch {ta/max(na,1e-9):6.2f} -> {tb/max(nb,1e-9):6.2f}  {k}")
