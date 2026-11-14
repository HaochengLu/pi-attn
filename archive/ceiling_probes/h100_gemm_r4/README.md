# H100 projection GEMM, round 4: W multicast in 2 x 1 clusters; X in registers (2026-11-14)

Question: after round 3 (`../h100_gemm`) the wgmma GEMM was 1.13-1.16x cuBLAS at M = 192..512 and 1.08x at
1024..2048, attributed to the 227 KB of shared memory that X and W stages share. Two levers were open: X in
registers (wgmma RS, shared memory for W only) and TMA multicast in CTA clusters. All runs: Qwen3-1.7B
projections, 28 weight copies in one CUDA graph, every launch streaming its weight from DRAM, cuBLAS in the same
process; only numbers of one run are compared. $1.56 in total (two L4 compiles $0.19, two H100 probes $0.71,
the validation $0.65).

| run | probe | result | verdict |
|---|---|---|---|
| 1 (`probe1.log`) | CTA clusters of the shipped tiles (`tiles_r4.py`, ids 200..225): 2 x 1 and 4 x 1 (one W stage multicast to the CTAs of 2 / 4 row tiles), 1 x 2 and 1 x 4 (X multicast to column tiles), 2 x 2 | bit-identical (812 launches, `BITCHECK`). 2 x 1 wins wherever the output has two or more row tiles: M = 256 x 28 layers 1.180 -> 1.079x cuBLAS, M = 512 1.147 -> 1.060x, M = 192 1.185 -> 1.106x, M = 128 1.042 -> 1.001x; 1-4% on gate_up from M = 256. 4 x 1, 1 x 2 and 2 x 2 are slower than no cluster (o, M = 256: 64x64 tile 8.9 us, 2 x 1 7.8, 1 x 2 9.5, 4 x 1 12.4, 2 x 2 12.6) | 2 x 1 adopted (ids 113..120) |
| 1 | X in registers (`gemm_sm90_rs.cuh`, ids 300..304): wgmma RS, shared memory holds W stages only (1536-1664 columns in flight instead of 512-768); every consumer thread loads its X registers of the next K stage from global memory while the current stage's steps run | bit-identical to the SS kernels (`BITCHECK`), but 0.7-0.8 us per K stage (o, M = 256: 21.4 us with 16 stages of 128 columns, 29.0 us with 32 stages of 64; shipped 8.9): the latency run D of round 3 saw for X behind deep W loads is not TMA issue order, a `ld.global` one stage ahead of a CTA that streams W waits as long | rejected |
| 1 | the same kernels with no X loads at all (mode 1, ids `300c`..: the registers keep a constant): the ceiling of any scheme that keeps X out of shared memory | at M = 256, with X costing nothing: o 7.7 us with 6 W stages, 8.3 with 13 (2 x 1 cluster 7.8, cuBLAS 6.7); down 16.4 with 6, 17.6 with 13 (cluster 17.6, cuBLAS 17.5); qkv 11.4 (cluster 10.9); gate_up 27.4 (cluster 27.1). Twice the W stages is never faster, and the best case is 7% below the cluster | above M = 128 neither X's shared memory nor the W columns in flight is the bound |
| 2 (`probe2.log`) | further 2 x 1 tiles: 64x128 (8 stages), and 128-column stages for 128x128, 128x64, 64x128 | within 2% of the tiles of run 1 or slower (qkv, M = 192: 64x128 11.0 us against 11.2 for 192x32) | not shipped |

Shipped: `Sm90Gemm<..., CX, CY>` in `gemm_sm90.cuh` (the cluster walk, `cudaLaunchKernelEx`), ids 113..120 =
2 x 1 clusters of the tiles of 104, 105, 106, 107, 91, 108, 90, 111; `gemm_dispatch.cuh`, `test_gemm.py`. A
cluster id refuses M that fits one row tile; the engine's M buckets end at row tile sizes, so an id tuned at
a bucket's top runs the whole bucket.

Validation (`final.sh`, `final.log`; one H100 sandbox, the tree applied to 61ff596):

    HASHX 2412 launches (ids 1, 13 and every wgmma id) over 108 cases; hashes differing from hashes_H100.json: 0 []
    test_gemm exit=0    INVARIANCE FAILURES: 0 []

Cold table, us for 28 layers x (qkv, o, gate_up, down); before = best of ids <= 112, after = best of all ids:

| M | cuBLAS | before | after |
|---|---|---|---|
| 128 | 1447 | 1527 (1.055x) | 1454 (1.005x) |
| 192 | 1540 | 1828 (1.187x) | 1707 (1.109x) |
| 256 | 1653 | 1945 (1.177x) | 1777 (1.075x) |
| 512 | 2405 | 2807 (1.167x) | 2599 (1.081x) |
| 1024 | 4011 | 4414 (1.100x) | 4332 (1.080x) |
| 2048 | 7764 | 8161 (1.051x) | 8147 (1.049x) |
| 2304 | 8549 | 9453 (1.106x) | 9233 (1.080x) |
| 4096 | 15572 | 16240 (1.043x) | 16145 (1.037x) |

What remains: M = 192 at 1.11x (qkv 1.20, gate_up 1.11: one 192-row tile has no second row tile to share W with,
and three 64-row tiles in 2 x 1 clusters leave one CTA in four without output; a 3 x 1 cluster does not build
on CUTLASS's collective, which divides the tile shape by the cluster shape), o at M = 256 (1.16) and qkv /
gate_up at M = 512 (1.15-1.17), 1.04-1.08x from M = 1024. None of it is the contract (the tiling, the cluster and
which CTA computes an element are free; only each element's K order is fixed, and every probe here kept it).
The engine was not run.

Files: `probe_r4.cu` / `probe_r4.py` / `tiles_r4.py` / `probe.sh` build against the shipped `gemm_sm90.cuh`
(`R4_SET=2` for run 2); `cold_table_r4.py` is `../h100_gemm/cold_table.py` with ids 1 and 13 added to the hashes
and "before" = ids <= 112; `compile_L4_*.log` are the sm_90a compile checks on an L4 (the first fails on a
3 x 1 cluster of a 64-row tile).
