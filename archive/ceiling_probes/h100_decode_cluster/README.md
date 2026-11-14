# H100 decode attention: what bounds `decode_kernel`, and the designs measured on the way to `decode90_kernel`

Everything here was measured on an H100 (Modal, CUDA 12.6) against pi-attn @ 61ff596, Qwen3 shapes (16 query heads, 8 kv heads, head dim 128, page 16), in the engine's regime: one CUDA graph over a ring of per-layer K/V caches, microseconds per layer (`dec90_bench.py`, which uses `kernels/bench_attention.py`'s timer). FlashInfer is its CUDA-core decode (`fi`). Each `pN_*.diff` is the whole patch of that step against 61ff596, next to the log that judged it; the variant ids in a log are those of its diff.

What shipped is in `kernels/pi_attn_sm90.cuh` (`decode90_kernel`). Its validation is in two logs: `validate_rule_batch2_H100.log` (`validate.sh`: test_invariance, xgpu_hashes against `results/modal/H100_sm90/hashes_H100.json`, the grids, the mutant) and `final_validate_H100.log` (`validate2.sh`: test_invariance and the grid under the shipped planner rule). The table at the end is from the second.

## What the probes established

| Fact | Evidence |
|---|---|
| `decode_kernel` is not bound by memory. With the K/V of one layer only (L2-resident) it takes 20.3 µs per layer at 640 keys, 22.2 µs from DRAM. | `hopprobe_H100.log` (`L2RES` lines), `l2res.py` |
| One CTA takes in a fold group's 256 KB in 24.3 µs with one 8 KB tile load in flight per wait, 10.2 µs with 4, 6.6 µs with 16. More copy warps do not help. `cp.async.bulk` per key (256 B) is slower (25 µs); a contiguous 8 KB bulk copy, which the paged layout does not offer, takes 3.7 µs. | `copyprobe_H100.log`, `copy_probe.cu` |
| The kernel's own work is the larger part. With a group's tiles in flight and three waits, 4 warps still take 6.0 µs for S of 16 tiles, 2.0 µs for P, 3.6 µs for the o chain and 2.2 µs to write out (512 keys). | `p6_deep_few_waits_stages_H100.log` (`stop` lines) |
| A hand-off between two CTAs of a cluster costs 0.45 µs with an mbarrier (blocking `try_wait` or a `test_wait` spin alike), 0.86 µs with a polled flag; a cluster barrier 0.45 µs; 7 CTAs push 16 KB each into an eighth in 2.7 µs. | `hopprobe_H100.log`, `hop_probe.cu`, `clprobe_H100.log` |
| Clusters do not pack: 320 clusters of 8 CTAs that each spin 10 µs take 55 to 81 µs, 320 plain CTAs 11 µs. | `clprobe_H100.log`, `cl_probe.cu` |
| A remote mbarrier arrival needs `.release.cluster` and the wait `.acquire.cluster`. With the default (CTA) scope the receiver read stale shared memory: 115 of 416 comparisons differed. | `p2_ring_cta_token_H100.log` against `p3_*` |
| 20 warps (640 threads, 102 registers each) fit the group body, but not the body inside a loop over groups: ptxas spills (184 to 200 bytes of stack) and 640 keys go from 12.9 to 18.9 µs. 16 warps (128 registers) do not spill. | `ptxas_stack_*.log`, `p11_*`, `p12_*`, `p14_*` |

## Designs, batch 1, 640 keys (FlashInfer 9.2 µs, `decode_kernel` 21.9 µs)

| Step | Design | µs per layer | Verdict |
|---|---|---|---|
| p1 | Cluster of 2 / 4 / 8 CTAs; each loads the K of its key tiles and a head-dim slice of V for all 512 keys; maxima and P through distributed shared memory; two cluster barriers | 18.2 / 15.8 / 14.3 | A dim slice of V touches every key's cache line in every CTA. Batch 32 is 3 times slower. |
| p2 | Cluster as a ring: CTA c owns key tiles, (d, o) handed from CTA to CTA, one mbarrier per CTA | 14.5 / 12.1 / 11.6 / 14.1 (2 / 4 / 8 / 16) | Wrong bits for 16 rows per kv head (memory ordering, see above). |
| p3 | The ring with per-warp hand-off, cluster-scope release / acquire; also 8 copy warps | 13.0 / 14.1 / 19.3 (4 / 8 / 16); 12.7 with 8 copy warps on 4 CTAs | Correct. Each hand-off costs 0.45 µs or more. |
| p4 | The ring with a `test_wait` spin and maxima exchanged through mbarriers instead of a cluster barrier | 14.7 / 17.1 / 27.8 (4 / 8 / 16) | Slower. |
| p5 | One CTA or a cluster of 2 / 4, every tile in flight, one cp.async group and one wait per tile | 23.2 / 16.7 / 14.9 | 32 waits cost what the loads saved. |
| p6 | The same with three waits per CTA, and stage timing | 22.4 / 16.2 / 14.9 | The stages above: the time is the 4 warps' own work. |
| p7 | One CTA, 20 warps: 4 tiles of S at once, the o chain on 16 warps | 14.9 | First gain from warps alone. |
| p8 | Running max on one warp, P only for own tiles, o chain on 8 warps | 13.8 | |
| p9 | K of all tiles in the first cp.async group, S in one round | 12.9 | Best single-group body; grid in `p9_*_grid_H100.log`. |
| p10 | o chain on 4 warps, rescale skipped when every scale is 1 | 13.2 | Slower than 8 warps. |
| p11, p12 | A CTA runs several fold groups in turn (loop inlined / body not inlined), 20 warps | 18.9 / 21.3 | Register spills. |
| p14 | The same on 16 warps | 14.2 | No spills. This is the shipped kernel. |

Every step from p3 on gave `decode_kernel`'s bits in `dec90_bench.py eq` (0 mismatches in every pass: mixed lengths, 2 or 16 rows per kv head, workspaces sized for longer requests).

## Why a CTA runs several groups

`decode90_kernel` needs 220 KB of shared memory, so one CTA per SM. The engine captures a decode graph per batch size with the page table at `--max-seq-len`, so a grid with one CTA per possible fold group is mostly CTAs that return at once, and with one CTA per SM those wait behind the working ones. The shipped grid has `decode90_slots` CTAs per (request, kv head), as many as fit one wave, and CTA j runs the request's groups j, j + slots, and so on. With the workspace sized for 13 times the keys the times do not move (`SLACK13` lines of `validate_rule_batch2_H100.log`, whose rule took batch 1 at any capacity: 14.6 µs at 640 keys against 14.4).

## Where it runs

It is faster than `decode_kernel` only while SMs would otherwise idle. With a CTA per fold group it is 9 to 36% faster from 192 keys on for 1 to 8 requests and 5 to 15% slower at 96 keys; once CTAs run several groups in turn it is 10 to 49% slower (`GRID2` lines of `validate_rule_batch2_H100.log`: batch 1 at 9000 keys, batch 2 at 4200 and 9000, batch 4 from 2176), and with 16 requests it does not gain (`p14_*_grid_H100.log`). A launch knows its batch and its capacity in fold groups, not its lengths, so the planner (`decode90_plan`) takes it when every group the launch can hold gets a CTA of its own in one wave: batch x kv heads x max(capacity, 2) <= SMs. In the engine the capacity is `--max-seq-len`: at 8192 that is batch 1 only. `validate_rule_batch2_H100.log` is the full validation under an earlier rule (batch x kv heads x 8 <= SMs), which showed the regressions above; `final_validate_H100.log` is the shipped rule.

## What remains above FlashInfer

At batch 1 a full group takes about 14 µs against FlashInfer's 9 to 10. The remaining time is the group's own instruction stream on one SM (issue, S, P, o chain) plus 2 µs of partial write, arrival and fold when a request has more than one group. Spreading a group over SMs costs 0.45 µs per hand-off and loses the packing, which is what p1 to p6 measured. Larger batches at short lengths would also gain (batch 8 at 640 keys: 18.2 against 22.8 µs), but a per-batch rule cannot separate them from long requests, where the kernel is up to 40% slower; that needs the engine to choose between two captured graphs by the lengths it knows on the host.

## Shipped kernel under the shipped rule (`final_validate_H100.log`), µs per layer

`v0` is the planner's choice; where it equals `v1` the launch ran `decode_kernel`. The grid's capacity is the request length, so the rule takes batch 1 everywhere, batch 4 up to 1280 keys and batch 8 up to 640.

| batch | keys | FlashInfer | `decode_kernel` | planner |
|---|---|---|---|---|
| 1 | 96 | 4.4 | 5.3 | 6.1 |
| 1 | 320 | 8.6 | 13.4 | 9.7 |
| 1 | 640 | 9.2 | 22.1 | 14.1 |
| 1 | 1280 | 9.5 | 21.8 | 14.5 |
| 1 | 2176 | 10.3 | 22.0 | 15.2 |
| 1 | 4200 | 12.0 | 23.0 | 17.5 |
| 4 | 96 | 5.6 | 6.1 | 6.3 |
| 4 | 320 | 9.4 | 13.9 | 10.3 |
| 4 | 640 | 10.4 | 22.0 | 15.4 |
| 4 | 1280 | 12.7 | 22.5 | 18.1 |
| 8 | 96 | 6.0 | 6.3 | 6.9 |
| 8 | 320 | 10.6 | 14.2 | 11.8 |
| 8 | 640 | 12.7 | 22.6 | 17.9 |

Batch 4 from 2176 keys, batch 8 from 1280 and batches 16 to 64 run `decode_kernel` as before (within 0.2 µs of `v1` in the log).

Bits: `test_invariance` passes with every decode position checked against prefill under the planner and with each decode kernel forced (`checks failed: 0 []`); the five attention hashes and the 108 GEMM hashes of `eval/xgpu_hashes.py` equal `results/modal/H100_sm90/hashes_H100.json`; the mutant `decode90_d_unfused` (d with two roundings instead of one fma) fails 14 path checks.

## Engine (`final_engine_H100.log`, `engine.sh`)

Qwen3-1.7B, `--deterministic --max-seq-len 8192`, one sandbox, greedy output tokens per second (median of 3 repeats). `PI_DECODE90=0` keeps `decode_kernel` everywhere; with 16 fold groups of capacity the rule takes `decode90_kernel` at batch 1 only.

| workload (in / out / concurrency) | `PI_DECODE90=0` | planner |
|---|---|---|
| 64 / 256 / 1 | 422.7 | 429.9 |
| 2048 / 128 / 1 | 351.9 | 375.4 |
| 512 / 256 / 8 | 2660.2 | 2655.7 |

`detcheck` reports DETERMINISTIC with 0 mismatches in every row, and the 64 H100 reference rollouts (`results/modal/H100_sm90/H100_prechange.rollout.gen.json`) replay with identical token ids and 14442/14442 log-probabilities bitwise equal.

