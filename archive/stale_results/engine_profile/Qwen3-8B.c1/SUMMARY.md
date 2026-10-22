# Qwen3-8B c1: 64:256:1

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 131697.3 | 137927.9 |
| attention | 4304.4 | 1249.9 |
| sampling | 44.3 | 46.4 |
| norm | 331.7 | 334.6 |
| reduction | 0.0 | 0.0 |
| other | 1318.9 | 3700.2 |
| total | 137696.6 | 143259.0 |

## top kernels: flashinfer

-  116231.8 ms   875048x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-   10779.5 ms   288997x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-    2679.3 ms   289429x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-    2285.3 ms     3024x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-    1136.1 ms   585985x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     972.8 ms     2556x  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
-     822.7 ms     4179x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     473.2 ms   292993x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     466.1 ms   289429x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,
-     414.6 ms   292993x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     399.1 ms     3612x  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)
-     331.7 ms   292993x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)

## top kernels: pi

-   70470.6 ms   296608x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)256, (int)3>(const __nv_bfloat16 *, long, 
-   46363.9 ms   575872x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-   15671.8 ms   287735x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-    2534.5 ms     2916x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-    2003.9 ms     9612x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-    1869.5 ms   287492x  pi_attn::decode_kernel(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, const int *, const
-    1235.0 ms   582111x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     531.4 ms   291056x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     458.8 ms   291056x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     371.1 ms   288680x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-     359.7 ms   287492x  pi_attn::decode_merge_kernel(const float *, const float2 *, const int *, int, __nv_bfloat16 *, int)
-     334.6 ms   291056x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)

flashinfer: {"out_tok_s": 60.08, "req_s": 0.23, "ttft_ms": 22.12, "tpot_ms": 16.62, "itl_ms": 16.68, "p99_ttft_ms": 23.66, "e2e_ms": 4261.04, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [60.1]}

pi: {"out_tok_s": 57.95, "req_s": 0.23, "ttft_ms": 21.9, "tpot_ms": 17.24, "itl_ms": 17.3, "p99_ttft_ms": 22.07, "e2e_ms": 4417.29, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [58.0]}
