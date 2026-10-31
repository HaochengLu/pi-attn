| config | workload | original tok/s | PI attention only | row-invariant GEMM only | both (--deterministic) |
|---|---|---|---|---|---|
| 0.6B (3 launches) | 64/256/c1 | 513.5 | +3.3% | -6.1% | -2.9% |
| 0.6B (3 launches) | 512/256/c32 | 7454.7 | +1.3% | -0.8% | +0.4% |
| 0.6B-c128 (3 launches) | 512/256/c128 | 11641.0 | -0.3% | -1.4% | -1.7% |
| 1.7B (3 launches) | 64/256/c1 | 282.7 | +1.8% | +3.9% | +5.9% |
| 1.7B (3 launches) | 512/256/c32 | 5080.5 | +1.1% | -1.6% | -0.6% |
| 8B (3 launches) | 64/256/c1 | 89.9 | +4.7% | +1.4% | +6.8% |
| 8B (3 launches) | 512/256/c32 | 1813.7 | +0.5% | +0.5% | +1.1% |
