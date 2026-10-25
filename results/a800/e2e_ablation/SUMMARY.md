| config | workload | original tok/s | PI attention only | row-invariant GEMM only | both (--deterministic) |
|---|---|---|---|---|---|
| 0.6B (3 launches) | 64/256/c1 | 514.0 | +3.2% | -4.7% | -1.7% |
| 0.6B (3 launches) | 512/256/c32 | 7461.8 | +1.5% | -0.5% | +1.1% |
| 0.6B-c128 (3 launches) | 512/256/c128 | 11656.3 | +1.3% | -2.9% | -1.5% |
| 1.7B (3 launches) | 64/256/c1 | 282.6 | +2.0% | +4.0% | +5.8% |
| 1.7B (3 launches) | 512/256/c32 | 5073.6 | +1.5% | -1.8% | -0.1% |
| 8B (3 launches) | 64/256/c1 | 89.9 | +4.7% | +1.2% | +6.5% |
| 8B (3 launches) | 512/256/c32 | 1813.5 | +0.5% | +0.5% | +1.2% |
