# Qwen3-8B shared: 4096:128:8:3968

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 13694.5 | 12379.1 |
| attention | 1196.3 | 115.4 |
| sampling | 3.9 | 2.6 |
| norm | 40.0 | 30.5 |
| reduction | 0.0 | 0.0 |
| other | 307.4 | 3336.2 |
| total | 15242.2 | 15863.9 |

## top kernels: flashinfer

-    8871.5 ms    77276x  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)
-    2873.2 ms     3672x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-    1040.4 ms     1620x  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
-     859.6 ms    19188x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-     621.1 ms     3603x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     193.3 ms     1421x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-     143.9 ms      324x  void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigne
-     132.7 ms    21960x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     130.1 ms    43920x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-      76.7 ms    21960x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-      65.0 ms      108x  ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn
-      47.6 ms    19332x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,

## top kernels: pi

-    4238.8 ms    27411x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)256, (int)3>(const __nv_bfloat16 *, long, 
-    3109.8 ms     3780x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-    2829.7 ms    13396x  void pi_attn::decode_partial_kernel<(int)1, (int)1>(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_b
-    2353.2 ms    27932x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    1594.7 ms     4932x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     250.8 ms      540x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     190.4 ms      324x  void pi_attn::prefill_kernel<(int)8, (int)1, (int)2, (bool)0>(const __nv_bfloat16 *, __nv_bfloat16 *, const __
-     183.1 ms      540x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     170.6 ms     1104x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-     124.2 ms    16096x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     122.0 ms      420x  void pi_gemm::gemm_small_kernel<(int)2, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-     114.4 ms    32192x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u

flashinfer: {"out_tok_s": 395.93, "req_s": 3.09, "ttft_ms": 173.02, "tpot_ms": 18.98, "itl_ms": 18.99, "p99_ttft_ms": 195.38, "e2e_ms": 2583.85, "tokens": 4096, "in": 4096, "out": 128, "concurrency": 8, "shared": 3968, "repeats": 1, "out_tok_s_all": [395.9]}

pi: {"out_tok_s": 296.42, "req_s": 2.32, "ttft_ms": 196.08, "tpot_ms": 25.64, "itl_ms": 25.64, "p99_ttft_ms": 220.15, "e2e_ms": 3451.87, "tokens": 4096, "in": 4096, "out": 128, "concurrency": 8, "shared": 3968, "repeats": 1, "out_tok_s_all": [296.4]}
