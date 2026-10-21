# Qwen3-0.6B c1: 64:256:1

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 7873.9 | 11368.4 |
| attention | 1551.0 | 1976.2 |
| sampling | 28.1 | 37.4 |
| norm | 168.5 | 206.9 |
| reduction | 0.0 | 0.0 |
| other | 610.6 | 729.5 |
| total | 10232.2 | 14318.3 |

## top kernels: flashinfer

-    7533.2 ms   589938x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-     866.3 ms   146511x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-     443.1 ms   297949x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     229.6 ms   146511x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,
-     207.1 ms     5379x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     200.0 ms   148975x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     197.8 ms   148975x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)3072>(const T1 *, T1 *, long, long)
-     168.5 ms   148975x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)
-     167.6 ms   147127x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-      36.1 ms      252x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-      34.1 ms     2809x  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)
-      28.1 ms     5236x  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::native::ArgMaxOps<flo

## top kernels: pi

-    5698.7 ms   364324x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-    2687.6 ms   178779x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    1884.3 ms   178108x  void pi_gemm::gemm_small_kernel<(int)2, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    1376.3 ms   178234x  pi_attn::decode_kernel(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, const int *, const
-     584.7 ms   361618x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     289.6 ms     5150x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     275.4 ms     2238x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     237.9 ms   180809x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)3072>(const T1 *, T1 *, long, long)
-     229.6 ms   180810x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     208.2 ms   178962x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-     206.9 ms   180810x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)
-     161.9 ms      390x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm

flashinfer: {"out_tok_s": 503.12, "req_s": 1.97, "ttft_ms": 5.23, "tpot_ms": 1.97, "itl_ms": 2.0, "p99_ttft_ms": 6.39, "e2e_ms": 508.66, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [503.1]}

pi: {"out_tok_s": 470.11, "req_s": 1.84, "ttft_ms": 4.92, "tpot_ms": 2.12, "itl_ms": 2.14, "p99_ttft_ms": 5.19, "e2e_ms": 544.39, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [470.1]}
