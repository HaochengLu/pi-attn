# H100 decode and prefill attention probes (not shipped)

These were measured on H100 and L4 against pi-attn @ 4848937. None of them shipped. Every variant is kept as a diff against that commit (`*_not_shipped.diff`), next to the logs that rejected it.

| Probe | Result | Log |
|---|---|---|
| Two-phase decode: all 16 tiles' S first, then the O/D chain; 3–4 K/V ring slots | Batch 1, 640–2176 keys: 19.5–20.0 µs per layer, against 21.8–22.1 µs today and 9.1–10.3 µs for FlashInfer. Batch 32 is 11–25% slower. | `dev2_H100.log` |
| Two-phase decode with per-page TMA boxes (sm_90) | 29.6–30.4 µs per layer: TMA boxes are slower than `cp.async` | `dev5_H100_decode90.log` |
| Two-launch decode split (S kernel, then an O kernel per 32 head dims) | Slower on L4 (24.2–45.0 µs) | `dev6b_L4_split.log` |
| Two-phase decode with the page table in shared memory | No gain on L4 | `dev7_L4_twophase_pt.log` |
| Prefill group totals in shared memory (249 → 211 registers) | Within ±1.6% on H100; ptxas still serialises the wgmma chain | `dev1_L4.log`, `dev2_H100.log` |

Every decode variant passes `test_invariance` with every decode position checked against prefill. The two-phase and split kernels were checked on L4, the TMA kernel on H100.

**Why decode does not speed up.** The ablations remove parts of the two-phase kernel at batch 1, 640 keys, on L4 (`dev3_L4_ablate.log`, `dev4_L4_ablate.log`):

| What is removed | µs per layer |
|---|---|
| nothing (full kernel) | 18.9 |
| tensor-core work | 18.1 |
| K/V copies | 10.3 |
| both | 6.9 |

The time is one CTA streaming its fold group's K and V. Barriers and MMAs account for little of it. The 512-key fold group puts each group's bytes on a single CTA. A faster small-batch decode on H100 must therefore spread those bytes over more SMs, or use a smaller group, which would change the bits.
