| engine | batch c4/c16 | path | repeat (unique/200) | seeded sample c4/c16 | prefix full/partial hit |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 1/1 (of 16) | 1/24 | 14 | 6/7 (of 16) | 0/0 (of 8) |
| llm_infer_rust --deterministic (PI) | 0/0 (of 16) | 0/24 | 1 | 0/0 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 | 1/2 (of 16) | 0/24 | 7 | 7/6 (of 16) | 0/0 (of 8) |
| SGLang 0.5.5 --enable-deterministic-inference | 0/0 (of 16) | 1/24 | 1 | 2/1 (of 16) | 0/0 (of 8) |
| vLLM 0.11.2 | 2/1 (of 16) | 0/24 | 14 | 6/6 (of 16) | 1/0 (of 8) |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 2/3 (of 16) | 1/24 | 9 | 5/6 (of 16) | 0/0 (of 8) |

| engine (greedy out tok/s) | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust (FlashInfer) | 234 | 1394 | 3522 | 1282 | 379 |
| llm_infer_rust --deterministic (PI) | 223 | 1427 | 3537 | 1242 | 370 |
| SGLang 0.5.5 | 228 | 1373 | 2492 | 1282 | 380 |
| SGLang 0.5.5 --enable-deterministic-inference | 65 | 502 | 959 | 692 | 174 |
| vLLM 0.11.2 | 198 | 1223 | 3303 | 1203 | 363 |
| vLLM 0.11.2 VLLM_BATCH_INVARIANT=1 | 198 | 1206 | 3220 | 1190 | 362 |
