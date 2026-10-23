"""Compare two xgpu_hashes.py outputs: per case, whether PI and the vendor kernels give
identical bits on both GPUs. Usage: python eval/xgpu_compare.py a.json b.json"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
print(f"{a['device']} vs {b['device']}")
for kind in ("gemm", "attention"):
    impls = list(next(iter(a[kind].values())))
    same = {i: 0 for i in impls}
    for case, ha in a[kind].items():
        hb = b[kind][case]
        print(f"{kind:9s} {case:32s} " + "  ".join(f"{i}={'same' if ha[i] == hb[i] else 'DIFF'}" for i in impls))
        for i in impls:
            same[i] += ha[i] == hb[i]
    print(f"{kind}: " + ", ".join(f"{i} identical on {same[i]}/{len(a[kind])}" for i in impls))
