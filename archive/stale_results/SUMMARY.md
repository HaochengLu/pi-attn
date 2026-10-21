### Qwen3-1.7B
| mode | acc c=1 (strict) | acc c=16 | items | generations differing c1 vs c16 |
|---|---|---|---|---|
| flashinfer | 69.22 | 68.92 | 1319 | 893 |
| deterministic | 69.45 | 69.45 | 1319 | 0 |
### Qwen3-8B
| mode | acc c=1 (strict) | acc c=16 | items | generations differing c1 vs c16 |
|---|---|---|---|---|
| flashinfer | 90.00 | 90.00 | 500 | 327 |
| deterministic | 90.20 | 90.20 | 500 | 0 |
