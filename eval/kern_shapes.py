"""Steady-state per-(kernel, grid) median durations from an nsys report's GPU trace.
Usage: python eval/kern_shapes.py trace.csv [tail_fraction]"""
import csv, statistics, sys
rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if not r.get("Name") or r.get("GrdX", "") == "":
            continue
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], r["GrdX"], r["GrdY"], r["GrdZ"], r["BlkX"]))
rows.sort()
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
rows = rows[int(len(rows) * (1 - frac)):]
span = rows[-1][0] - rows[0][0]
g = {}
for st, d, n, x, y, z, bx in rows:
    short = n.split("(")[0].replace("void ", "")[-60:]
    g.setdefault((short, f"{x}x{y}x{z}/{bx}"), []).append(d)
tot = sum(sum(v) for v in g.values())
print(f"window {span/1e6:.1f} ms, kernel time {tot/1e6:.1f} ms")
for (n, grid), v in sorted(g.items(), key=lambda kv: -sum(kv[1]))[:22]:
    print(f"{sum(v)/1e6:8.1f} ms {len(v):7d}x med {statistics.median(v)/1e3:7.2f} us  {grid:16s} {n}")
