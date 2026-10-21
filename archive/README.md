Earlier kernel iterations (v1, v2, v4) and the first GEMM M-invariance probe, kept for the
history of the design; the current kernels live in llm_infer_rust/src/kernels/pi/.

- `kernels_v6/`: benchmarks of the chunk-pure design (64-key chunk partials, decode
  split variants 1-9), superseded by online softmax within 512-key fold groups.
- `prefill2_probe/`: the standalone probes that led to the current kernels: ablations of
  the chunk-pure prefill (`probe_prefill.cu`), the FlashInfer-style prefill/decode
  prototype with its bitwise test (`prefill2.cuh`, `test_p2.py`) and benchmarks.
