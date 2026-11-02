# Deterministic mode (`--deterministic`)

`--deterministic` makes every response a bitwise function of the model weights and
the request alone: the same token-id prompt, sampling parameters and seed give the
same tokens no matter which other requests share the batch, how many are in flight,
whether the prompt prefix is in the radix cache, where chunked prefill cuts the
prompt, whether a token is produced by decode or by re-prefilling, whether CUDA
graphs or eager kernels run, and which page size or batch limits the server uses.
Every other serving optimization stays on.

```bash
./scripts/run-server.sh --model-path /path/to/Qwen3 --deterministic
```

Requires CUDA, bfloat16, head_dim 128, a GQA group of at most 16 query heads per KV
head, and projection weights with `K % 64 == 0` and `N % 32 == 0` (checked at start).

## How

Nondeterminism comes from reductions whose order depends on how work is
partitioned. Deterministic mode fixes every reduction tree by problem shape and
positions only, never by batch or schedule:

- **Attention** (`pi` backend, `src/kernels/pi/`): keys are cut into 32-key tiles and
  512-key fold groups at absolute positions. Within a group each query row runs an
  online softmax over the group's tiles in ascending order, starting from the empty
  state at the group start (P rounded to bf16 against the running max, P·V
  accumulated on the tensor cores); the group results are then folded in ascending
  order. Prefill folds the groups in registers or, for long prefixes with few
  queries, splits the groups over CTAs and merges them; decode runs each group of a
  request in one CTA (the tile split across warps without changing any element's
  sequence) and merges the groups. Every path executes the same operation sequence
  per row, so batched, chunked, prefix-cached, prefilled and decoded rows agree
  bitwise.
- **Projections** (`src/kernels/pi.rs`, `src/kernels/pi/gemm_*.cuh`): the sum over `k`
  of every output element is fixed by the weight shape (N, K) alone. It runs in
  ascending 16-wide MMA steps from zero, cut into S segments whose sums are added to +0
  in ascending order. S = `gemm_segments(N, K, t_max)` with t_max = 1 in this one-GPU
  engine: 2 when N <= 1024, K >= 2048 and K % 512 == 0 (4 or 8 for N <= 512 with
  K >= 4096 and K % 1024 == 0, a shape no Qwen3 projection has), otherwise 1. Every
  kernel takes S as an argument, so a launch over a column shard of a weight computes the
  whole weight's bits: the basis of tensor-parallel degree invariance (PI-Attn
  `kernels/tp_invariance.md`). Small-M kernels, CUTLASS tiles (also with the segment
  total kept in L2 slots), deterministic stream-K and segment-parallel tiles all realise
  this order, so at startup the engine times them per shape and M bucket and keeps the
  fastest (logged as "PI GEMM tile choice") without changing results. The buckets are the CUDA-graph capture sizes, so every graph runs the tile
  timed at its own M.
- **Sampling**: greedy is `argmax`. With temperature > 0, one batched kernel samples
  row *i* from its own Philox stream `(seed, token position)` (a fork of FlashInfer's
  top-k/top-p rejection sampler whose stream does not depend on the row index). The
  OpenAI `seed` field selects the stream; without it a random seed is drawn and
  returned in the response.
- Norms, RoPE, activation, KV store and embedding are already row-wise kernels.

## Scope

Results are reproducible on the same GPU model with the same build. Different GPU
architectures or compilers may round differently. Tensor parallelism is not
supported in this engine.

## Checking

`tests/e2e/test_deterministic.py` (server started with `--deterministic`) and the
black-box checker in the PI-Attn project (`detcheck`) compare outputs across
concurrency, prefix-cache state, decode vs re-prefill, seeded sampling, restarts and
server configurations.
