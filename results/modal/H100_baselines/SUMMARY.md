| engine | batch c4/c16 | shared prefix x32 | path | seeded path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 169 | 16/16 (of 16) | 5/6 (of 8) |
| llm_infer_rust --deterministic (PI) | 0/0 (of 16) | 0/32 | 0/24 | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 50 | 16/16 (of 16) | 8/8 (of 8) |
| SGLang 0.5.5 deterministic (FlashInfer) | 0/0 (of 16) | 0/32 | 0/24 | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 deterministic (Triton) | not run | | | | | | |
| vLLM 0.11.2 | not run | | | | | | |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | not run | | | | | | |
| vLLM 0.17.1 | 16/16 (of 16) | 32/32 | 24/24 | 24/24 | 161 | 16/16 (of 16) | 8/6 (of 8) |
| vLLM 0.17.1 VLLM_BATCH_INVARIANT=1 | 0/0 (of 16) | 0/32 | 0/24 | 8/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |

| engine (greedy out tok/s) | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 443 | 3024 | 8645 | 3411 | 985 |
| llm_infer_rust --deterministic (PI) | 426 | 2661 | 8273 | 3239 | 899 |
| SGLang 0.5.5 | 407 | 2758 | 7812 | 3296 | 956 |
| SGLang 0.5.5 deterministic (FlashInfer) | 243 | 1710 | 5446 | 2311 | 577 |
| vLLM 0.17.1 | 442 | 3066 | 8738 | 3648 | 1081 |
| vLLM 0.17.1 VLLM_BATCH_INVARIANT=1 | 149 | 1168 | 4442 | 1889 | 467 |
