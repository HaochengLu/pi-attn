# H100 round 3: integrated validation of the merged tree (a547dd5)

Tree: 61ff596 + decode90_kernel (archive/ceiling_probes/h100_decode_cluster) + Hopper GEMM ids 103-112
(archive/ceiling_probes/h100_gemm) + warp-specialised prefill90w_kernel variants 6/7 (archive/ceiling_probes/h100_prefill).
Each sandbox checked out 61ff596 and applied `merged_kernels_engine.diff` (`git diff --binary 61ff596 a547dd5 -- kernels
llm_infer_rust eval scripts detcheck`). Three sandboxes, 2026-11-14, $5.01 in total.

| file | what |
|---|---|
| kern.sh, kern_H100.log | H100 kernel suites, mutants, hashes, kernel benches (one sandbox, $1.91) |
| engine.sh, engine_H100.log | H100 engine: rollouts, detcheck, PI vs FlashInfer mode (one sandbox, $2.87) |
| classA_A10G.log | A10G (sm_86): kern.sh classA, hashes against the A800 reference ($0.23) |
| hashes_H100_merged.json, hashes_A10G_merged.json | eval/xgpu_hashes.py output of the merged tree |

## 1. Bit identity (all pass)

H100 (kern_H100.log), attention extension built with VARIANTS [1..7], DECODE_VARIANTS [1, 2]:

    test_invariance exit=0   checks failed: 0 []
    test_directed exit=0     DIRECTED FAILURES: 0
    test_matrix exit=0       TOTAL FAILURES: 0
    test_gemm exit=0         INVARIANCE FAILURES: 0 []
    test_mutants exit=0      UNCAUGHT MUTANTS: 0 []   (merge_reversed 42, decode_fold_unfused 105, prefill_group_256 12654,
                             decode90_d_unfused 7 failed checks; both GEMM controls caught)
    HASHES gemm: PI identical 108/108; PI and vendor both identical 108/108
    HASHES attention: PI identical 5/5; PI and vendor both identical 5/5      (vs results/modal/H100_sm90/hashes_H100.json;
                             hashes_H100_merged.json equals that file as a whole)
    HASHX 1956 wgmma launches over 108 cases; hashes differing from hashes_H100.json: 0 []   (every wgmma id 90-112)
    EQ decode90_kernel vs decode_kernel: 180 comparisons, 0 mismatches

H100 engine (engine_H100.log):

    cargo build exit=0     C7511 lines in build: 0
    64 rollouts: 64 with identical token ids; 14442/14442 log-probabilities bitwise equal
    ROLLOUTS IDENTICAL TO REFERENCE: True
    detcheck exit=0  DETERMINISTIC  (batch 0/16 0/16, shared 0/32, path 0/24, spath 0/24, repeat 1 unique of 200, sample 0/16 0/16)

Class A, A10G (classA_A10G.log): VARIANTS [1, 2, 3], test_invariance `checks failed: 0 []`, `DIRECTED FAILURES: 0`,

    HASHES gemm: PI identical 108/108      (vs results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json)
    HASHES attention: PI identical 5/5

(The vendor kernels differ between A10G and A800 on 60 of 108 GEMM and 5 of 5 attention cases, as before.)

Not run: archive/ceiling_probes/h100_prefill/attn_hashes.py (the five attention hashes for each explicit prefill
variant); a script edit dropped the line. The planner's variants are covered by the 5/5 above and variants 1-7 are
compared with one another by test_invariance; the per-variant hashes are in h100_prefill/runs/10_validation.log.

## 2. Kernel benches (H100, kern_H100.log; 28 layers' KV or weights from DRAM in one CUDA graph)

Decode attention, us per layer: FlashInfer (better of its two kernels) | PI planner (v0) | ratio. v1 = decode_kernel,
v2 = decode90_kernel are in the log.

| batch | kv 96 | 320 | 640 | 1280 | 2176 | 4200 |
|---|---|---|---|---|---|---|
| 1 | 4.5 / 6.1 (1.36) | 7.9 / 9.7 (1.23) | 8.8 / 14.4 (1.64) | 9.6 / 14.5 (1.51) | 10.3 / 15.2 (1.48) | 12.3 / 17.7 (1.44) |
| 4 | 5.7 / 6.3 (1.11) | 9.4 / 10.4 (1.11) | 10.5 / 15.3 (1.46) | 12.9 / 18.2 (1.41) | 18.2 / 24.9 (1.37) | 30.3 / 32.5 (1.07) |
| 8 | 6.0 / 6.9 (1.15) | 10.5 / 11.9 (1.13) | 12.9 / 18.6 (1.44) | 20.3 / 26.0 (1.28) | 31.5 / 33.2 (1.05) | 53.3 / 54.2 (1.02) |
| 16 | 6.4 / 6.9 (1.08) | 13.2 / 15.1 (1.14) | 20.5 / 26.3 (1.28) | 35.4 / 36.4 (1.03) | 55.5 / 55.5 (1.00) | 98.2 / 103.9 (1.06) |
| 32 | 8.1 / 8.7 (1.07) | 21.7 / 20.3 (0.94) | 35.3 / 37.6 (1.07) | 62.6 / 65.7 (1.05) | 102.1 / 106.7 (1.05) | 189.2 / 199.9 (1.06) |
| 64 | 11.6 / 12.1 (1.04) | 36.3 / 33.1 (0.91) | 63.7 / 66.7 (1.05) | 118.9 / 123.3 (1.04) | 196.8 / 203.1 (1.03) | 370.3 / 391.1 (1.06) |

Batch 1 at 640-4,200 keys: 22-23 us (decode_kernel) -> 14.4-17.7 us (planner), FlashInfer 8.8-12.3 us. At batch 1-8 and
96 keys the planner picks decode90_kernel where decode_kernel is 0.2-0.8 us faster (6.1 vs 5.3 at batch 1).

Prefill attention, us per layer, hq=16 hkv=8: FlashInfer fa2 / fa3, round-2 kernel v5, PI planner v0 (ratio to the
better FlashInfer backend). hq=32 rows and v6/v7 are in the log.

| batch x q (+prefix) | fa2 | fa3 | v5 (before) | PI v0 |
|---|---|---|---|---|
| 1 x 512 | 36.2 | 15.5 | 20.0 (1.29) | 19.4 (1.25) |
| 1 x 1024 | 74.7 | 22.8 | 39.5 (1.73) | 33.3 (1.46) |
| 1 x 2048 | 107.3 | 60.8 | 120.0 (1.97) | 71.0 (1.17) |
| 1 x 4096 | 340.4 | 176.8 | 367.0 (2.08) | 225.4 (1.27) |
| 8 x 1024 | 163.9 | 118.6 | 202.4 (1.71) | 155.7 (1.31) |
| 8 x 2048 | 505.2 | 360.3 | 663.6 (1.84) | 510.6 (1.42) |
| 1 x 2048 + 2048 | 214.6 | 124.5 | 270.9 (2.18) | 185.4 (1.49) |
| 1 x 128 + 4096 | 31.5 | 64.8 | 33.9 (1.08) | 31.4 (1.00) |

Engine shapes (ENGSHAPE rows): v0 is 0.94-1.21x of FlashInfer's best except 8 x 64 + 1024 (1.51x).

GEMM, 28 layers x (qkv, o, gate_up, down), us; "before" = best of ids <= 102, "after" = best of all ids:

| M | cuBLAS | before | after |
|---|---|---|---|
| 1 | 1316 | 1434 (1.089x) | 1307 (0.993x) |
| 8 | 1326 | 1477 (1.114x) | 1318 (0.994x) |
| 16 | 1331 | 1485 (1.116x) | 1326 (0.997x) |
| 32 | 1384 | 1489 (1.077x) | 1336 (0.966x) |
| 64 | 1414 | 1518 (1.073x) | 1355 (0.958x) |
| 128 | 1439 | 1729 (1.201x) | 1513 (1.051x) |
| 192 | 1542 | 2170 (1.407x) | 1750 (1.135x) |
| 256 | 1657 | 2343 (1.414x) | 1922 (1.160x) |
| 512 | 2441 | 3051 (1.250x) | 2792 (1.144x) |
| 1024 | 4118 | 4445 (1.079x) | 4445 (1.079x) |
| 2048 | 7788 | 8420 (1.081x) | 8420 (1.081x) |
| 2304 | 8955 | 10821 (1.208x) | 9713 (1.085x) |
| 4096 | 16001 | 16524 (1.033x) | 16524 (1.033x) |
| 8192 | 33220 | 34537 (1.040x) | 33186 (0.999x) |

## 3. Engine, Qwen3-1.7B, greedy, PI (--deterministic) vs FlashInfer mode

One sandbox, FI and PI servers interleaved, two rounds, a fresh server per pass. Out tok/s is the median over all
repeats (4 for Table 3 and high concurrency, 6 for Table 4); TTFT is the mean of the two servers' means. Every repeat
is in engine_H100.log.

Table 3 workloads (--max-running-req 32, num 64, 2 repeats per server):

| in/out/conc | FI tok/s | PI tok/s | PI vs FI | FI TTFT ms | PI TTFT ms | | paper Table 3 (before) |
|---|---|---|---|---|---|---|---|
| 64/32 c1 | 432 | 449 | +3.8% | 4.7 | 4.9 | +3.8% | -1.7%, TTFT +4.2% |
| 64/32 c8 | 2791 | 2904 | +4.0% | 14.3 | 13.8 | -3.5% | -2.2%, +5.8% |
| 64/32 c32 | 7713 | 8117 | +5.2% | 23.2 | 24.1 | +3.9% | -4.8%, +20.8% |
| 512/256 c8 | 2952 | 2786 | -5.6% | 31.1 | 29.8 | -4.2% | -11.8%, +18.6% |
| 2048/128 c8 | 1963 | 1947 | -0.8% | 65.4 | 69.7 | +6.5% | -7.6%, +20.0% |
| 1024/512 c32 | 6905 | 6870 | -0.5% | 109.0 | 116.6 | +7.0% | -5.3%, +15.7% |

High concurrency (--max-running-req 256, num 64, 2 repeats per server):

| in/out/conc | FI tok/s | PI tok/s | PI vs FI | FI TTFT ms | PI TTFT ms | |
|---|---|---|---|---|---|---|
| 512/256 c64 | 12521 | 12569 | +0.4% | 89.3 | 93.3 | +4.5% |
| 512/256 c128 | 15553 | 14717 | -5.4% | 156.1 | 176.5 | +13.1% |
| 512/256 c256 | 17357 | 15642 | -9.9% | 384.3 | 398.2 | +3.6% |
| 2048/128 c64 | 3599 | 3414 | -5.1% | 379.2 | 409.4 | +8.0% |

Table 4 shapes (eval/baselines.sh settings: --max-running-req 32, num 32, 3 repeats per server), out tok/s. The
vLLM / SGLang rows are from results/modal/H100_baselines/SUMMARY.md, which is another sandbox: our FlashInfer mode
measured 443 / 3024 / 8645 / 3411 / 985 there and 434 / 2958 / 8293 / 3297 / 953 here (2-4% lower), so compare PI
with the other engines through its ratio to FlashInfer mode in the same sandbox.

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust FlashInfer mode, this run | 434 | 2958 | 8293 | 3297 | 953 |
| **PI, this run (merged tree)** | **453** | **2793** | **8425** | **3313** | **944** |
| PI / FlashInfer mode, this run | 1.046 | 0.944 | 1.016 | 1.005 | 0.990 |
| PI TTFT ms (FI) this run | 5.3 (5.2) | 30.5 (29.4) | 75.7 (72.7) | 114.4 (107.8) | 81.7 (77.0) |
| baselines sandbox: FlashInfer mode | 443 | 3024 | 8645 | 3411 | 985 |
| baselines sandbox: PI before round 3 | 426 (0.962) | 2661 (0.880) | 8273 (0.957) | 3239 (0.950) | 899 (0.913) |
| baselines sandbox: vLLM 0.17.1 | 442 | 3066 | 8738 | 3648 | 1081 |
| baselines sandbox: vLLM 0.17.1 batch-invariant | 149 | 1168 | 4442 | 1889 | 467 |
| baselines sandbox: SGLang 0.5.5 | 407 | 2758 | 7812 | 3296 | 956 |
| baselines sandbox: SGLang 0.5.5 deterministic | 243 | 1710 | 5446 | 2311 | 577 |

## What remains

- 512/256 c8 is the one low-concurrency row still clearly below FlashInfer mode (-5.6%, both tables); the decode
  attention grid at batch 4-8 and 640-1,280 keys is 1.28-1.46x there.
- At 128-256 running requests PI is 5.4-9.9% below, and 2048/128 c64 5.1% below.
- TTFT is 4-8% above FlashInfer mode on the prefill-heavy rows (13% at 512/256 c128).
