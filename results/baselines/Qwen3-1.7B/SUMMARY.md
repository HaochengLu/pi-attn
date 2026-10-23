| engine | batch c4/c16 | shared prefix x32 | path | seeded path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 3/1 (of 16) | 9/32 | 1/24 | 9/24 | 14 | 6/6 (of 16) | 0/0 (of 8) |
| llm_infer_rust --deterministic (PI) | 0/0 (of 16) | 0/32 | 0/24 | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 | 2/2 (of 16) | 11/32 | 0/24 | 8/24 | 7 | 6/6 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (FlashInfer) | 0/0 (of 16) | 14/32 | 1/24 | 8/24 | 1 | 1/1 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (Triton) | server stalled (request timed out after 1200 s) | | | | | | |
| vLLM 0.11.2 | 3/3 (of 16) | 14/32 | 0/24 | 8/24 | 11 | 5/6 (of 16) | 1/1 (of 8) |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 2/3 (of 16) | 14/32 | 1/24 | 8/24 | 12 | 5/6 (of 16) | 0/0 (of 8) |

| engine (greedy out tok/s) | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 234 | 1392 | 3507 | 1283 | 378 |
| llm_infer_rust --deterministic (PI) | 230 | 1472 | 3685 | 1325 | 393 |
| SGLang 0.5.5 | 228 | 1377 | 3713 | 1283 | 379 |
| SGLang 0.5.5 deterministic (FlashInfer) | 65 | 503 | 1712 | 693 | 175 |
| vLLM 0.11.2 | 198 | 1221 | 3298 | 1205 | 366 |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 198 | 1202 | 3234 | 1192 | 363 |
