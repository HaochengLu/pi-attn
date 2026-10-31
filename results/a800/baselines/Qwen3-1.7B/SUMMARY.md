| engine | batch c4/c16 | shared prefix x32 | path | seeded path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 169 | 16/16 (of 16) | 5/6 (of 8) |
| llm_infer_rust --deterministic (PI) | 0/0 (of 16) | 0/32 | 0/24 | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 33 | 16/16 (of 16) | 8/8 (of 8) |
| SGLang 0.5.5 deterministic (FlashInfer) | 5/5 (of 16) | 30/32 | 24/24 | 24/24 | 1 | 4/6 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (Triton) | server stalled (request timed out after 1200 s) | | | | | | |
| vLLM 0.11.2 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 169 | 16/16 (of 16) | 8/6 (of 8) |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 168 | 16/16 (of 16) | 7/2 (of 8) |
| vLLM 0.17.1 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 167 | 16/16 (of 16) | 8/6 (of 8) |
| vLLM 0.17.1 VLLM_BATCH_INVARIANT=1 | 0/0 (of 16) | 0/32 | 0/24 | 8/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |

| engine (greedy out tok/s) | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 283 | 1894 | 5014 | 1878 | 539 |
| llm_infer_rust --deterministic (PI) | 298 | 1790 | 4951 | 1863 | 542 |
| SGLang 0.5.5 | 253 | 1768 | 4865 | 1837 | 530 |
| SGLang 0.5.5 deterministic (FlashInfer) | 73 | 554 | 1993 | 843 | 192 |
| vLLM 0.11.2 | 237 | 1599 | 4149 | 1694 | 508 |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 244 | 1539 | 4083 | 1671 | 495 |
| vLLM 0.17.1 | 270 | 1842 | 4890 | 1885 | 553 |
| vLLM 0.17.1 VLLM_BATCH_INVARIANT=1 | 80 | 609 | 2228 | 921 | 223 |
