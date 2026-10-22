# Qwen3-0.6B c1b: 64:256:1

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 8063.9 | 12710.2 |
| attention | 1589.7 | 1932.5 |
| sampling | 28.8 | 36.6 |
| norm | 172.1 | 202.5 |
| reduction | 0.0 | 0.0 |
| other | 623.9 | 716.8 |
| total | 10478.5 | 15598.6 |

## top kernels: flashinfer

-    7724.3 ms   605046x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-     889.3 ms   150254x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-     452.9 ms   305436x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     235.4 ms   150254x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,
-     206.2 ms     5379x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     204.1 ms   152719x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     202.3 ms   152718x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)3072>(const T1 *, T1 *, long, long)
-     172.1 ms   152719x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)
-     171.4 ms   150871x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-      36.1 ms      252x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-      34.1 ms     2809x  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)
-      28.8 ms     5370x  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<c10::BFloat16, at::native::ArgMaxOps<flo

## top kernels: pi

-    4496.0 ms   355809x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    3566.0 ms   187070x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-    2130.1 ms   177745x  void pi_gemm::gemm_balanced_kernel<__nv_bfloat16, (int)1, (int)256, (int)3>(const __nv_bfloat16 *, long, const
-    1345.3 ms   174664x  pi_attn::decode_kernel(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, const int *, const
-     572.1 ms   354478x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     564.1 ms     4730x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     384.2 ms     6578x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     362.4 ms     4898x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     361.0 ms     9658x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     234.7 ms   177239x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)3072>(const T1 *, T1 *, long, long)
-     225.6 ms   177240x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     203.6 ms   175392x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)

flashinfer: {"out_tok_s": 503.45, "req_s": 1.97, "ttft_ms": 4.85, "tpot_ms": 1.97, "itl_ms": 2.0, "p99_ttft_ms": 6.12, "e2e_ms": 508.3, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [503.5]}

pi: {"out_tok_s": 471.24, "req_s": 1.84, "ttft_ms": 5.19, "tpot_ms": 2.11, "itl_ms": 2.14, "p99_ttft_ms": 6.53, "e2e_ms": 543.06, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [471.2]}
