# Phase 1–2: PI-Attn reference model and CUDA kernels

Dev GPU: RTX 4090 D (sm_89, 114 SMs, 99 KB smem/block opt-in, driver 535). Versions: `VERSIONS.txt`.
Environment: `source /uniparser/users/luhaocheng/pi_attn/env.sh` (built by `setup_env.sh`).

## Design (final)
- KV cut into absolute-position chunks of 64 keys. Chunk partial for a query row:
  m = chunk-local max of scaled scores, P = bf16(exp2(s - m)), l = fixed-order sum of rounded P,
  o = P·V from a zero accumulator (signed zeros canonicalized).
- Output = canonical sequential fold of partials in ascending chunk order:
  Mn = max(M, m); a = ex2(M - Mn); b = ex2(m - Mn); L = fma(L, a, l*b); O = fma(O, a, o*b); out = O / L.
  Explicit `__fmul_rn/__fmaf_rn/ex2.approx` so no kernel can contract differently.
- Prefill folds in registers (o_c produced per n-tile pair and folded immediately: 4 live registers).
  Decode: split CTAs write per-chunk partials (fp32) to a workspace; merge kernel folds them.
- All kernels share `chunk_qk`, `scores_to_p`, `chunk_pv`, `fold_*` (pi_attn.cuh); configs only
  change work distribution (warps per CTA, 16-row groups per warp sharing K/V fragments, buffers).
- Invalid KV slots zero-filled at load (cp.async src-size 0): stale/NaN cache memory never read.

## Reference-model findings (ref/)
- Invariance holds by construction (CPU bit-level model, fixed-order elementwise fp32).
- Power-of-two "snapped" max (earlier idea) is NOT needed for invariance and costs ~2x error
  for peaked attention (the dominant weight gets rounded). Dropped. Exact local max: error on par
  with or below FA2 (mean 0.9–1.5e-4 vs 1.2–2.0e-4 in the model).
- FA2-style running max changes every row's bits under any change of tile size or split points.

## GPU invariance tests (kernels/test_invariance.py, test_matrix.py) — all pass, 0 failures
Geometries hq/hkv ∈ {16/8, 32/8, 8/8, 32/4}, q scale 1 and 6 (peaked), lengths 1..4096 incl. chunk
boundaries (63/64/65/127/128/129), long contexts 32768/20001 and 16384/9999.
Bitwise-equal across: prefill configs (4w×16, 8w×16, 4w×32 rows), mixed/shuffled batches, chunked
prefill at random cuts over a cached prefix stored in different physical pages, decode variants
(1/2/4 warps, shared or split K/V tiles) × split counts {1,3,16,128}, decode vs prefill row, reruns.
Unused pool pages are NaN-filled. Accuracy vs float64: on par with torch SDPA (bf16 output rounding dominates).

## Performance on 4090 D (µs, median; FlashInfer 0.6.18 paged, bf16, page 16)
Decode (PI best variant = 1 warp/CTA, one shared 17 KB tile):
| hq/hkv | batch | kv | FI default (nondet) | FI fixed_split 2048 (det, SGLang-style) | PI-Attn (det) |
|---|---|---|---|---|---|
| 16/8 | 1 | 1K | 31.3 | 36.2 | 19.4 |
| 16/8 | 1 | 8K | 32.5 | 124.3 | 36.1 |
| 16/8 | 1 | 32K | 169.7 | 498.5 | 194.5 |
| 16/8 | 8 | 2K | 40.7 | 48.9 | 50.1 |
| 16/8 | 8 | 16K | 584.2 | 589.4 | 627.4 |
| 16/8 | 32 | 1K | 166.8 | 171.2 | 163.0 |
| 16/8 | 32 | 8K | 1132.8 | 1167.3 | 1224.5 |
| 16/8 | 128 | 1K | 581.3 | 585.0 | 619.0 |
| 16/8 | 128 | 4K | 2227.0 | 2231.8 | 2443.2 |
| 32/8 | 1 | 32K | 172.3 | 500.9 | 199.9 |
| 32/8 | 128 | 4K | 2276.9 | 2237.1 | 2603.7 |
(vLLM-style no-split FI decode equals FI default on this GPU.)

Prefill (best PI config = 8 warps × 16 rows on 4090):
| hq/hkv | batch×q | prefix | FI (nondet) | PI | ratio |
|---|---|---|---|---|---|
| 16/8 | 1×512 | 0 | 77.4 | 50.4 | 0.65 |
| 16/8 | 1×2048 | 0 | 269.6 | 323.9 | 1.20 |
| 16/8 | 1×8192 | 0 | 2548.8 | 3375.8 | 1.32 |
| 16/8 | 8×1024 | 0 | 366.6 | 489.7 | 1.34 |
| 16/8 | 1×512 | 8192 | 383.7 | 695.6 | 1.81 |
| 32/8 | 1×8192 | 0 | 4502.9 | 6345.2 | 1.41 |
| 32/8 | 1×512 | 8192 | 1003.5 | 1356.6 | 1.35 |

Notes: no GPU perf-counter permission in this container (ncu ERR_NVGPUCTRPERM), tuning is by A/B.
On 4090 resident warps/SM (latency hiding) dominates; 32-row-per-warp prefill (cfg3) is slower here
because it fits only 4 warps/SM. A800 variants (8w×32 rows, 139 KB; 4-warp double-buffered decode)
are implemented and invariance-tested in code paths but cannot run on 4090.
Long-prefix/short-query prefill cannot use split-KV cheaply: invariance needs per-chunk partials per
row, whose traffic exceeds KV reads when rows are many (inherent cost; decode avoids it: ≤16 rows/head).
