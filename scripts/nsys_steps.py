"""Per-decode-step GPU time by kernel from an nsys sqlite export.

Uses the last `window_s` seconds of the trace (inside the second of two identical
generation requests, after any lazy tuning) and counts decode steps as
attention-decode kernel launches / layers."""
import sqlite3
import sys

path, layers, window = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
db = sqlite3.connect(path)
end = db.execute("select max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
rows = db.execute(
    "select s.value, k.start, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName = s.id "
    "where k.start > ? and k.end < ?", (end - int((window + 0.05) * 1e9), end - int(0.05e9))).fetchall()
dec = [r for r in rows if "BatchDecodeWithPagedKVCacheKernel" in r[0] or "decode_partial_kernel" in r[0]]
steps = max(1, len(dec) // layers)
span = (max(r[1] for r in rows) - min(r[1] for r in rows)) / 1e3
agg = {}
for name, _, dur in rows:
    key = name.split("(")[0].split("<")[0].replace("void ", "")[-60:]
    if "cutlass" in name and "Kernel" in key:
        key = "cutlass " + name.split("cutlass_80_")[1][:40] if "cutlass_80_" in name else "cutlass::Kernel(pi gemm)"
    if "gemm_small_kernel" in name:
        key = "pi gemm_small " + name.split("gemm_small_kernel<")[1][:12]
    a = agg.setdefault(key, [0, 0])
    a[0] += dur
    a[1] += 1
total = sum(v[0] for v in agg.values())
print(f"{steps} decode steps in {span / 1e3:.1f} ms wall ({span / steps:.0f} us/step); GPU kernel time {total / steps / 1e3:.0f} us/step")
for name, (t, n) in sorted(agg.items(), key=lambda x: -x[1][0])[:16]:
    print(f"  {t / steps / 1e3:7.1f} us/step  {n / steps:5.1f}/step  {name}")
