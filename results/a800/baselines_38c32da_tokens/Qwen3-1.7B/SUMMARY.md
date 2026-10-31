| engine | batch c4/c16 | shared prefix x32 | path | seeded path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 2/1 (of 16) | 12/32 | 0/24 | 6/24 | 10 | 6/6 (of 16) | 0/0 (of 8) |
| llm_infer_rust --deterministic (PI) | 0/0 (of 16) | 0/32 | 0/24 | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 | 3/3 (of 16) | 11/32 | 1/24 | 11/24 | 3 | 6/7 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (FlashInfer) | 0/0 (of 16) | 15/32 | 1/24 | 7/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (Triton) | server stalled (request timed out after 1200 s) | | | | | | |
| vLLM 0.11.2 | 1/2 (of 16) | 11/32 | 1/24 | 7/24 | 14 | 5/5 (of 16) | 1/0 (of 8) |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 1/3 (of 16) | 11/32 | 1/24 | 7/24 | 11 | 5/6 (of 16) | 0/0 (of 8) |

| engine (greedy out tok/s) | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 283 | 1895 | 5016 | 1879 | 539 |
| llm_infer_rust --deterministic (PI) | 299 | 1785 | 4978 | 1850 | 537 |
| SGLang 0.5.5 | 253 | 1770 | 4838 | 1836 | 529 |
| SGLang 0.5.5 deterministic (FlashInfer) | 73 | 553 | 1991 | 842 | 192 |
| vLLM 0.11.2 | 236 | 1595 | 4141 | 1696 | 508 |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 244 | 1535 | 4059 | 1668 | 494 |
