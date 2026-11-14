# H100 projection GEMM: what bounds it below M ~ 1024, and the tiles that follow (2026-11-14)

Question (critic, `midm.py`): on H100 the wgmma GEMM was 1.08-1.42x cuBLAS at M = 64..512 and 1.25x at M = 2304
(2048-token chunk plus decode rows), and >= half of the end-to-end gap to FlashInfer mode. All runs: Qwen3-1.7B
projections, 28 weight copies in one CUDA graph, every launch streaming its weight from DRAM; cuBLAS in the same
process. Sandboxes differ by up to 10% in absolute time, so only numbers of one run are compared.

| run | probe | result | verdict |
|---|---|---|---|
| B (`run_b.py`, `runB.diff`, `runB.log`) | register-only K chain: wgmma on fixed shared-memory operands, no loads (`PI_GEMM90_PROBE`), and an mma.sync chain on registers | wgmma 20.8 ns per k16 step for one warp group at n <= 64 (384 steps = 8.0 us), 23-25 ns for two warp groups at n16 / n32; an SM saturates at 3.6e12 MAC/s (128x256: 144 ns per step); mma.sync m16n8k16 13.3 ns per step | the K chain is not the M <= 64 bound (down: 16.4-18.5 us measured, chain 8.0 us) |
| B | tiles sized for one wave (128xN, 192xN, 64x96, 64x192; 5 stages or fewer) | M = 2304: 1.22 -> 1.09x (192x192: 132 tiles); M = 128..512 only 1.19-1.34x; same tile with fewer stages is slower (64x16 M = 64 down: 8 stages 18.2 us, 128x16 with 5 stages 24.3 us) | wave fit matters at 2304; below, stage depth matters more |
| C (`run_c.py`, `runC.diff`, `runC.log`) | deeper stages (1024 columns in flight per CTA for 64-row tiles, 448-768 for 128 / 192-row tiles) | time follows the unique W bytes in flight, N x columns x 2: down M = 64: 2.1 MB 18.5 us, 3.1 MB 16.5 us, 4.2 MB 13.4 us (cuBLAS 14.8). M = 8: 1.165 -> 1.04x, M = 64: 1.13 -> 0.99x, M = 128: 1.19 -> 1.05x, M = 256: 1.44 -> 1.19x | adopted (ids 103..112) |
| C | the three idle producer warps touch the tile's W lines ahead of the TMA stream (blocking byte loads, or `prefetch.global.L2`) | no gain on any tile (64x16 deep, o M = 64: 6.7 / 7.0 / 7.3 us without / loads / prefetch) | rejected |
| D (`run_d.py`, `runD_split_pipelines.diff`, `runD.log`) | separate pipelines: X in 3-4 stages, W in 8-48 stages (up to 3072 columns), one producer thread issuing W stage g and X stage g - lead | bit-identical, but 0.8 us per K stage whatever the tile (o M = 8: 25.9 us against 5.8): an X load issued behind deep W loads completes only after them (TMA loads of a CTA complete in issue order), so X sees DRAM latency with 2-3 stages of lookahead | rejected: X and W must share stages, which caps the columns in flight at 227 KB / (2 (BM + BN)) |

Shipped: `gemm_sm90.cuh` ids 103..112 (three consumer warp groups for BM = 192), `gemm_dispatch.cuh`, `test_gemm.py`.
`cold_table.py` is the before / after table (`final.log`), and the SHA-256 of every wgmma id on the inputs of
`eval/xgpu_hashes.py` against `results/modal/H100_sm90/hashes_H100.json`.

What remains: M = 192..512 at 1.13-1.16x (per SM, shared memory holds 227 KB of stages: at M = 256, N = 2048 the
best split is 64x64 tiles with 768 columns in flight, 3.1 MB device-wide); M >= 1024 at 1.05-1.09x (untouched
128x256 / 128x128 tiles, a fixed 2-4 us plus ~5%).

Files: each run's kernel state is its `.diff` against 61ff596 (`probe90.cu` is the probe extension as of run C,
`probe90_d.cu` and `probe_common.py` as of run D; run B called `linear90(x, w, id, probe)` and listed its tiles as
4-tuples).
