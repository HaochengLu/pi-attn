../results/accuracy/Qwen3-1.7B vs ../results/a800/accuracy/Qwen3-1.7B
| mode | client concurrency | items | generations differing between GPUs |
|---|---|---|---|
| flashinfer | 1 | 1319 | 906 |
| flashinfer | 16 | 1319 | 910 |
| deterministic | 1 | 1319 | 0 |
| deterministic | 16 | 1319 | 0 |
../results/accuracy/Qwen3-8B vs ../results/a800/accuracy/Qwen3-8B
| mode | client concurrency | items | generations differing between GPUs |
|---|---|---|---|
| flashinfer | 1 | 500 | 325 |
| flashinfer | 16 | 500 | 343 |
| deterministic | 1 | 500 | 0 |
| deterministic | 16 | 500 | 0 |
