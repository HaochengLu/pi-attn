"""Per-decode-step GPU time by kernel from an nsys trace of scripts/nsys_decode.sh (graph nodes
traced). Takes the kernels of the last `window` seconds before the trace's last kernel, which
lie inside the second request's decode, and divides by the number of steps there: greedy
decode launches one argmax per step in both attention modes.
Usage: python scripts/nsys_step.py <trace.sqlite> [window_s]"""
import collections
import re
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
window = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
t_end = max(r[1] for r in rows)
rows = [r for r in rows if r[0] >= t_end - window * 1e9]
by = collections.defaultdict(lambda: [0, 0])
for s, e, short, full in rows:
    name = re.sub(r"\s+", " ", names[full])[:110]
    by[name][0] += e - s
    by[name][1] += 1
steps = sum(c for name, (_, c) in by.items() if "argmax" in name.lower())
if not steps:
    sys.exit("no argmax kernel in the window: not a greedy decode trace")
busy = sum(t for t, _ in by.values())
span = max(r[1] for r in rows) - min(r[0] for r in rows)
print(f"{len(rows)} kernels, {steps} steps: GPU busy {busy / steps / 1e3:.1f} us/step, wall {span / steps / 1e3:.1f} us/step")
for name, (t, c) in sorted(by.items(), key=lambda kv: -kv[1][0]):
    print(f"{t / steps / 1e3:8.2f} us/step {c / steps:6.2f}/step  {name}")
