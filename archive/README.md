Earlier kernel iterations (v1, v2, v4), probes and one-off scripts, kept for the history of
the design; nothing here is used by the engine or by eval/. The current kernels live in
llm_infer_rust/src/kernels/pi/.

- `ceiling_probes/`: one probe per remaining gap to the non-deterministic engine, each
  asking whether the gap is a kernel problem or a cost of the canonical program; its
  README lists every probe with its result, adopted or not.
- `decode_fuse_probe/`: decode-kernel variants (`-DVAR=1..12`: group fold and arrival
  counter schemes) and the version adopted by the engine (`final_*`), with timings.
- `gemm_seg_probe/`: the first K-segmented GEMM for M <= 16 (sweep over segment count,
  k tile and warps against cuBLAS), from which the shape-only segment rule came.
- `gemm_stream_probe/`: weight-streaming GEMV prototypes for small M (per-warp X ring,
  balanced unit split).
- `micro/gemm_m_invariance.py`: the first check of whether cuBLAS bf16 GEMM rows depend on M.
- `kernels_v6/`: benchmarks of the chunk-pure design (64-key chunk partials, decode
  split variants 1-9), superseded by online softmax within 512-key fold groups.
- `prefill2_probe/`: the standalone probes that led to the current kernels: ablations of
  the chunk-pure prefill (`probe_prefill.cu`), the FlashInfer-style prefill/decode
  prototype with its bitwise test (`prefill2.cuh`, `test_p2.py`) and benchmarks.
- `ref_chunk_partials/`: the PyTorch reference of the chunk-partial design and its tests.
- `one_off/`: scripts written for a single check (`verify_v3.sh`, `verify_coop.sh`,
  `gemm_step_cost.py`); not maintained.
- `stale_results/`: results of earlier kernel and engine versions, moved out of `results/`
  on 2026-10-22.
