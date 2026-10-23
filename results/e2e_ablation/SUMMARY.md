| config | workload | original tok/s | PI attention only | row-invariant GEMM only | both (--deterministic) |
|---|---|---|---|---|---|
| 0.6B (3 launches) | 64/256/c1 | 522.7 | +2.0% | -0.1% | +1.9% |
| 0.6B (3 launches) | 512/256/c32 | 5727.7 | +1.5% | +2.4% | +3.3% |
| 0.6B-c128 (3 launches) | 512/256/c128 | 8371.2 | -1.3% | +0.1% | -1.4% |
| 1.7B (3 launches) | 64/256/c1 | 233.9 | +0.9% | -2.3% | -1.0% |
| 1.7B (3 launches) | 512/256/c32 | 3527.9 | +0.8% | +4.3% | +4.9% |
| 8B (3 launches) | 64/256/c1 | 60.3 | +1.1% | -1.9% | -0.6% |
| 8B (3 launches) | 512/256/c32 | 849.4 | +0.6% | +2.6% | +3.2% |
