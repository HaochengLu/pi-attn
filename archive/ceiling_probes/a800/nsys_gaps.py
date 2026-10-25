"""Idle gaps between kernels in the decode window of an nsys trace (as nsys_step.py):
total idle per step, histogram, and the kernels on both sides of the largest gaps."""
import collections, re, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
t_end = max(r[1] for r in rows)
rows = [r for r in rows if r[0] >= t_end - 0.3e9]
marks = sorted(r[0] for r in rows if "argmax" in names[r[2]].lower())
rows = sorted(r for r in rows if marks[0] < r[0] <= marks[-1])
steps = len(marks) - 1
gaps, prev_end, prev = [], rows[0][1], rows[0]
for r in rows[1:]:
    if r[0] > prev_end:
        gaps.append((r[0] - prev_end, prev[2], r[2]))
    if r[1] > prev_end:
        prev_end, prev = r[1], r
idle = sum(g for g, _, _ in gaps)
print(f"{steps} steps, idle {idle / steps / 1e3:.1f} us/step over {len(gaps) / steps:.1f} gaps/step")
for lo, hi in [(0, 2e3), (2e3, 5e3), (5e3, 20e3), (20e3, 1e9)]:
    sel = [g for g, _, _ in gaps if lo <= g < hi]
    print(f"  gaps {lo/1e3:5.0f}-{hi/1e3:5.0f} us: {len(sel) / steps:6.1f}/step, {sum(sel) / steps / 1e3:7.1f} us/step")
by = collections.defaultdict(lambda: [0, 0])
short = lambda n: re.sub(r"\s+", " ", names[n])[:60]
for g, a, b in gaps:
    k = (short(a), short(b))
    by[k][0] += g; by[k][1] += 1
for (a, b), (t, c) in sorted(by.items(), key=lambda kv: -kv[1][0])[:8]:
    print(f"  {t / steps / 1e3:6.1f} us/step {c / steps:5.2f}/step  {a}  ->  {b}")
