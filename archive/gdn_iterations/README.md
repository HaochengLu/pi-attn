# Exact GDN decode: superseded iterations

Kept so every log in `results/a800/gdn/` maps to the code that produced it. The current implementation is
`gdn/exact_decode.py` (two kernels); `gdn/exact_decode_ref.py` is the stateless reference (called v1 below).

| File | What | Logs |
|---|---|---|
| `gdn_exact2.py` | v2: row-incremental, one kernel per slot x head, l2norm and buffer writes outside | `e6_cost_v2.log` |
| `gdn_exact3.py` | v3: v2 fused with in-kernel l2norm and buffer writes, V split over programs | `e7_cost3.log` |
| `e6_cost.py`, `gdn_bench.py`, `e7_cost3.py` | benchmarks for v1/v2 and v3 (`e6_cost_v1.log` used a recurrent baseline that also copied the state back) | as above |

`e8_cost4*.log` were produced by the two-kernel version under its development name `gdn_exact4.py`, which is
`gdn/exact_decode.py` before q was indexed by key-head group (identical for H == HV, the benchmarked case).
A v5 that kept the token's rows in registers to drop barriers was not bitwise (see `gdn/README.md`, "Traps") and
was discarded.
