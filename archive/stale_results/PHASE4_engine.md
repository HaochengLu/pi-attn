# Phase 4: deterministic mode in llm_infer_rust (RTX 4090 D, Qwen3-0.6B)

Engine branch `pi-attn` (llm_infer_rust/), commits on top of e7e2806.

## What `--deterministic` does
- Attention backend `pi` (src/models/attention/backend.rs: PiAttentionBackend) calling the C ABI in
  src/kernels/pi/pi_launch.cu. Decode writes per-chunk partials into one static workspace shared by
  eager decode and all decode CUDA graphs (stream-ordered); page table and lengths are read on the GPU,
  so graph replay needs no host planning. Prefill (incl. mixed prefill/decode batches and chunked
  prefill) uses the PI prefill kernel with a host-side plan (variant + fold-group splits).
- Every projection (QKV, O, gate/up, down, lm_head) through `linear()` uses the row-invariant GEMM
  (hand-written small-M kernels + CUTLASS tiles, all bitwise identical); tile choice is tuned lazily per
  (N, K, M bucket) outside stream capture (graph warm-ups), heuristic inside capture.
- Sampling with temperature > 0: one FlashInfer sampling launch per row with Philox (seed, offset =
  token position * 2^24); OpenAI `seed` field. Greedy uses argmax (exact).
- Other ops (RMSNorm, add-RMSNorm, QK-norm, RoPE, SiLU-mul, KV store, embedding) are row-wise kernels.
- API additions: token-id prompts, `return_token_ids` (vLLM-compatible), `seed`.
- Environment note: this dev box exports CXXFLAGS=-std=c++17 globally; env.sh unsets it.

## Correctness
- Engine e2e protocol tests (tests/e2e, 12) pass against the deterministic server.
- Rust tests: CPU 159 pass; CUDA 28/29 pass + new `pi_prefill_matches_reference_and_decode_bitwise`.
  The one failure (`segmented_prefill_matches_eager_logits_and_kv_for_live_request_boundaries`,
  FlashInfer backend, KV error 0.03125) fails identically on untouched `main` on this 4090 (pre-existing).
- Greedy agreement with HF transformers fp32 (10 prompts x 128 tokens, teacher forcing):
  FlashInfer engine 99.22% of tokens = fp32 argmax; PI engine 99.39%. All disagreements are
  near-ties (fp32 logit margin <= 0.094).

## Determinism (detcheck, black-box over HTTP; Qwen3-0.6B, chunked prefill 1024, 64K-token KV pool)
| check | original engine (FlashInfer) | --deterministic (PI) |
|---|---|---|
| batch: alone vs concurrency 4 / 16 (16 prompts x 256 tok) | 3/16, 6/16 diverge | 0/16, 0/16 |
| prefix cache: cold vs full / partial hit (8 prompts) | 0/8, 0/8 | 0/8, 0/8 |
| path: decode vs re-prefill of generated tokens (24) | 2/24 | 0/24 |
| repeat: same prompt x100 under random load | 14 distinct outputs | 1 |
| seeded sampling T=0.8 p=0.95 k=50, alone vs conc 4 / 16 | 10/16, 10/16 (seed ignored) | 0/16, 0/16 |
Prefix-hit and flood eviction confirmed from server logs (cached_tokens 1984 on repeat, 0 after flood).

## End-to-end cost (same engine, streaming client, 3-run medians; results/e2e/)
| in/out, concurrency | FlashInfer tok/s (TPOT ms) | PI tok/s (TPOT ms) | cost |
|---|---|---|---|
| 64/32, 1 | 515.0 (1.84) | 433.8 (2.20) | -15.8% |
| 64/32, 8 | 2935.7 (2.34) | 2864.8 (2.44) | -2.4% |
| 64/32, 32 | 7592.8 (2.97) | 7547.0 (3.03) | -0.6% |
| 512/256, 8 | 2669.4 (2.85) | 2567.8 (2.96) | -3.8% |
| 2048/128, 8 | 1286.0 (5.60) | 1208.4 (5.94) | -6.0% |
| 1024/512, 32 | 3903.2 (7.79) | 3672.3 (8.29) | -5.9% |
Published reference points (other hardware/models): SGLang deterministic mode avg -34%; TML vLLM
batch-invariant 2.1x slower (1.6x with improved attention).

## Known gaps / next
- Batch-1 decode costs +0.36 ms/token (more than kernel microbenchmarks predict): to profile.
- No comparison yet against SGLang deterministic / vLLM batch-invariant / Vosti on the same GPU.
- Larger models (1.7B/4B/8B) not yet run end to end; A800 not yet available.
