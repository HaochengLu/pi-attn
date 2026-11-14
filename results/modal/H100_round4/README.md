# H100 round 4: integrated validation of the merged tree (ffbef96)

Tree: the round-3 tree (results/modal/H100_round3) + dec2 (cb1d169: decode_plan by the step's longest request, a decode
graph per (batch, kernel) in the engine) + gemm2 (994f77b: GEMM ids 113-120, 2 x 1 CTA clusters with W multicast).
Each sandbox checked out 61ff596 and applied `merged_kernels_engine.diff` (`git diff --binary 61ff596 ffbef96 -- kernels
llm_infer_rust eval scripts detcheck`). Same protocol as round 3. Two sandboxes, 2026-11-14, $3.58 in total.

| file | what |
|---|---|
| run.sh, dec_grid.py, all_H100.log | H100, one sandbox ($3.35): kernel suites while the engine builds, kernel benches, engine |
| classA_A10G.log | A10G (sm_86): run.sh classA, hashes against the A800 reference ($0.23) |
| hashes_H100_merged.json, hashes_A10G_merged.json | eval/xgpu_hashes.py output of the merged tree |

## 1. Bit identity (all pass)

H100 kernels (VARIANTS [1..7], DECODE_VARIANTS [1, 2]):

    test_invariance exit=0   checks failed: 0 []
    test_directed exit=0     DIRECTED FAILURES: 0
    test_matrix exit=0       TOTAL FAILURES: 0
    test_gemm exit=0         INVARIANCE FAILURES: 0 []
    test_mutants exit=0      UNCAUGHT MUTANTS: 0 []   (42 / 105 / 12654 / 7 failed checks; both GEMM controls caught)
    HASHES gemm: PI identical 108/108; PI and vendor both identical 108/108
    HASHES attention: PI identical 5/5; PI and vendor both identical 5/5     (hashes_H100_merged.json equals
                             results/modal/H100_sm90/hashes_H100.json as a whole)
    HASHX 2412 launches (ids 1, 13 and every wgmma id) over 108 cases; hashes differing from hashes_H100.json: 0 []
                             (ids 113, 114: 48 launches each; 115-120: 24 each)
    EQ decode variant 2 vs decode_kernel: 432 comparisons, 0 mismatches
    EQ decode variant 0 vs decode_kernel: 432 comparisons, 0 mismatches
    attention hashes equal to results/modal/H100_sm90/hashes_H100.json: True   (h100_prefill/attn_hashes.py: prefill v0, 1,
                             2, 5, 6, 7 and v6/v7 with 1 and 3 splits; the check missed in round 3)
    attention hashes equal to results/modal/H100_sm90/hashes_H100.json: True   (h100_decode_r4/attn_hashes.py: decode v0, 1, 2)

H100 engine:

    cargo build exit=0     C7511 lines in build: 0
    64 rollouts: 64 with identical token ids; 14442/14442 log-probabilities bitwise equal
    ROLLOUTS IDENTICAL TO REFERENCE: True
    detcheck exit=0  DETERMINISTIC  (batch 0/16 0/16, shared 0/32, path 0/24, spath 0/24, repeat 1 unique of 200, sample 0/16 0/16)

Class A, A10G: test_invariance `checks failed: 0 []`, `DIRECTED FAILURES: 0`,

    HASHES gemm: PI identical 108/108      (vs results/a800/xgpu/NVIDIA_A800-SXM4-80GB.json)
    HASHES attention: PI identical 5/5

## 2. What the engine chose

The tuner picks the new ids (first PI server, `PI GEMM tile choice`):

    n=2048 k=2048   128:113 192:114 256:114 384:115 448:116 512:115 768:116
    n=2048 k=6144   128:113 192:114 256:114 384:115 448:116 512:115 640:116 768:116 1024:117
    n=4096 k=2048   128:114 256:115 448:117 512:117 640:116 768:116 1280:118 1792:119
    n=12288 k=2048  128:114 256:116 448:118 512:118 768:118 896:118 1024:118 1280:118 1536:118
    n=151936 k=2048 1024:118 1792:118     (lm_head: otherwise ids <= 112)

M <= 64 stays on 103-105, and M >= 2048 on 90/91/108/111/112 (120 is never chosen in this server's table).
The decode graphs: `capturing CUDA decode graph of another launch plan size=16 bound=512 plan=2`,
`size=8 bound=1024 plan=2` (decode90_kernel below those bounds, decode_kernel above).

## 3. Kernel benches (H100, alone on the GPU; 28 layers' KV or weights from DRAM in one CUDA graph)

Decode attention, us per layer, workspace sized for 8192 keys as the engine's: FlashInfer (better of its two
kernels) / PI planner told the longest request (ratio); [vN] = decode_plan's kernel. "r3" in the log is the round-3
engine's choice at that capacity.

| batch | kv 96 | 640 | 2176 | 4200 |
|---|---|---|---|---|
| 1 | 4.5 / 5.3 (1.18) [v1] | 8.7 / 14.6 (1.68) [v2] | 10.3 / 15.3 (1.49) [v2] | 12.3 / 17.8 (1.45) [v2] |
| 2 | 4.5 / 5.4 (1.20) [v1] | 9.5 / 14.8 (1.56) [v2] | 12.5 / 17.8 (1.42) [v2] | 17.8 / 25.0 (1.40) [v1] |
| 4 | 5.7 / 6.2 (1.09) [v1] | 10.6 / 16.0 (1.51) [v2] | 18.4 / 25.5 (1.39) [v1] | 30.4 / 33.0 (1.09) [v1] |
| 8 | 6.0 / 6.4 (1.07) [v1] | 12.8 / 18.3 (1.43) [v2] | 31.5 / 33.7 (1.07) [v1] | 53.3 / 55.1 (1.03) [v1] |
| 16 | 6.4 / 6.9 (1.08) [v1] | 20.4 / 27.3 (1.34) [v1] | 55.2 / 55.6 (1.01) [v1] | 98.4 / 103.3 (1.05) [v1] |
| 32 | 7.8 / 8.5 (1.09) | 35.1 / 37.6 (1.07) | 101.1 / 107.2 (1.06) | 188.9 / 200.2 (1.06) |
| 64 | 11.6 / 13.0 (1.12) | 63.3 / 66.4 (1.05) | 196.1 / 203.7 (1.04) | 369.7 / 392.0 (1.06) |

Against the round-3 engine at this capacity: batch 2-8 at 640 keys 21.9-23.1 -> 14.8-18.3 us, batch 2 at 2,176 keys
22.7 -> 17.8, batch 1 at 96 keys 6.2 -> 5.3. In every cell of this grid the planner's kernel is the faster of the two.

Prefill attention: unchanged code; hq=16 v0 is 0.98-1.49x of the better FlashInfer backend (1.26 / 1.44 / 1.14 / 1.26
at 1 x 512 / 1024 / 2048 / 4096, 1.33 / 1.45 at 8 x 1024 / 2048, 1.49 at 1 x 2048 + 2048); all rows in the log.

GEMM, 28 layers x (qkv, o, gate_up, down), us; "before" = best of ids <= 112, "after" = best of all ids:

| M | cuBLAS | before | after |
|---|---|---|---|
| 1 | 1315 | 1309 (0.996x) | 1309 (0.996x) |
| 8 | 1327 | 1321 (0.995x) | 1321 (0.995x) |
| 16 | 1335 | 1326 (0.993x) | 1326 (0.993x) |
| 32 | 1383 | 1337 (0.967x) | 1337 (0.967x) |
| 64 | 1411 | 1355 (0.961x) | 1355 (0.961x) |
| 128 | 1451 | 1522 (1.049x) | 1460 (1.006x) |
| 192 | 1548 | 1844 (1.191x) | 1713 (1.106x) |
| 256 | 1654 | 1931 (1.168x) | 1772 (1.071x) |
| 512 | 2446 | 2792 (1.142x) | 2592 (1.060x) |
| 1024 | 4032 | 4416 (1.095x) | 4349 (1.079x) |
| 2048 | 7897 | 8365 (1.059x) | 8287 (1.049x) |
| 2304 | 8790 | 9575 (1.089x) | 9453 (1.075x) |
| 4096 | 15472 | 16175 (1.045x) | 16100 (1.041x) |
| 8192 | 31962 | 33469 (1.047x) | 33469 (1.047x) |

## 4. Engine, Qwen3-1.7B, greedy, PI (--deterministic) vs FlashInfer mode

One sandbox, FI and PI servers interleaved, two rounds, a fresh server per pass. Out tok/s is the median over all
repeats (4 for Table 3 and high concurrency, 6 for Table 4); TTFT is the mean of the two servers' means. Every repeat
is in all_H100.log. This sandbox's FlashInfer mode is 1-4% faster than round 3's, so compare rounds by the ratio.

Table 3 workloads (--max-running-req 32, num 64, 2 repeats per server):

| in/out/conc | FI tok/s | PI tok/s | PI vs FI | FI TTFT ms | PI TTFT ms | | round 3 | paper Table 3 |
|---|---|---|---|---|---|---|---|---|
| 64/32 c1 | 438 | 459 | +4.7% | 4.1 | 4.3 | +5.7% | +3.8% | -1.7% |
| 64/32 c8 | 2889 | 2996 | +3.7% | 13.2 | 12.1 | -8.8% | +4.0% | -2.2% |
| 64/32 c32 | 8656 | 8697 | +0.5% | 19.6 | 22.1 | +13.1% | +5.2% | -4.8% |
| 512/256 c8 | 2999 | 2944 | -1.8% | 27.5 | 29.5 | +7.2% | -5.6% | -11.8% |
| 2048/128 c8 | 2016 | 1994 | -1.1% | 60.1 | 64.9 | +7.9% | -0.8% | -7.6% |
| 1024/512 c32 | 7021 | 6986 | -0.5% | 99.6 | 106.8 | +7.3% | -0.5% | -5.3% |

64/32 c32: the repeats are bimodal in both modes (FI 8539-8791, PI 8522-8933); round 3's +5.2% had the same spread.

High concurrency (--max-running-req 256, num 64, 2 repeats per server):

| in/out/conc | FI tok/s | PI tok/s | PI vs FI | FI TTFT ms | PI TTFT ms | | round 3 |
|---|---|---|---|---|---|---|---|
| 512/256 c64 | 12691 | 12754 | +0.5% | 86.4 | 91.1 | +5.4% | +0.4% |
| 512/256 c128 | 16041 | 15359 | -4.3% | 158.5 | 173.0 | +9.2% | -5.4% |
| 512/256 c256 | 17144 | 15765 | -8.0% | 471.8 | 403.0 | -14.6% | -9.9% |
| 2048/128 c64 | 3770 | 3599 | -4.5% | 370.6 | 387.0 | +4.4% | -5.1% |

512/256 c256: FlashInfer mode's first repeat of round 1 was an outlier (14226 tok/s, TTFT 576 ms; the other three
17127-17415, TTFT 367 ms in round 2), so its TTFT mean is inflated; against round 2 alone PI's TTFT is +4%.

Table 4 shapes (eval/baselines.sh settings: --max-running-req 32, num 32, 3 repeats per server), out tok/s. The
vLLM / SGLang rows are from results/modal/H100_baselines/SUMMARY.md, another sandbox (FlashInfer mode 443 / 3024 /
8645 / 3411 / 985 there, 436 / 2998 / 8537 / 3375 / 974 here, 1-2% lower).

| engine | 64/256 c1 | 512/256 c8 | 512/256 c32 | 2048/256 c16 | 4096/128 c4 |
|---|---|---|---|---|---|
| llm_infer_rust FlashInfer mode, this run | 436 | 2998 | 8537 | 3375 | 974 |
| **PI, this run (round-4 tree)** | **455** | **2946** | **8624** | **3400** | **966** |
| PI / FlashInfer mode, this run | 1.044 | 0.983 | 1.010 | 1.007 | 0.992 |
| PI / FlashInfer mode, round 3 | 1.046 | 0.944 | 1.016 | 1.005 | 0.990 |
| PI TTFT ms (FI) this run | 4.9 (4.7) | 27.9 (26.5) | 66.3 (60.2) | 105.5 (99.8) | 76.0 (71.6) |
| baselines sandbox: FlashInfer mode | 443 | 3024 | 8645 | 3411 | 985 |
| baselines sandbox: PI before round 3 | 426 (0.962) | 2661 (0.880) | 8273 (0.957) | 3239 (0.950) | 899 (0.913) |
| baselines sandbox: vLLM 0.17.1 | 442 | 3066 | 8738 | 3648 | 1081 |
| baselines sandbox: vLLM 0.17.1 batch-invariant | 149 | 1168 | 4442 | 1889 | 467 |
| baselines sandbox: SGLang 0.5.5 | 407 | 2758 | 7812 | 3296 | 956 |
| baselines sandbox: SGLang 0.5.5 deterministic | 243 | 1710 | 5446 | 2311 | 577 |

## What remains

- Up to 64 running requests PI is within -1.8% .. +4.7% of FlashInfer mode in throughput.
- At 128-256 running requests PI is 4.3-8.0% below, and 2048/128 c64 4.5% below.
- TTFT is 4-10% above FlashInfer mode on most rows (prefill attention 1.14-1.49x at 512 or more query rows).
- Decode attention at 640-2,176 keys and batch 1-8 is still 1.39-1.68x FlashInfer's.
