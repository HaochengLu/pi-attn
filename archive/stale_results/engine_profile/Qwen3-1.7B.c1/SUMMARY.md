# Qwen3-1.7B c1: 64:256:1

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 31076.3 | 33178.5 |
| attention | 2419.4 | 831.7 |
| sampling | 43.1 | 47.5 |
| norm | 250.1 | 262.9 |
| reduction | 0.0 | 0.0 |
| other | 933.0 | 3070.4 |
| total | 34721.9 | 37390.9 |

## top kernels: flashinfer

-   30061.8 ms   895099x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-    1320.3 ms   222124x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-     731.0 ms   449790x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     348.6 ms   222124x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,
-     318.7 ms   224895x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     298.2 ms     4987x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     294.3 ms   224896x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     272.4 ms      756x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-     250.1 ms   224896x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)
-     247.6 ms   223048x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-     157.7 ms      924x  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
-      97.0 ms     2812x  void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)

## top kernels: pi

-   17826.0 ms   237343x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)256, (int)3>(const __nv_bfloat16 *, long, 
-    9495.8 ms   459068x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-    4433.2 ms   229272x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    1749.1 ms   229068x  pi_attn::decode_kernel(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *, const int *, const
-     817.9 ms   463680x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     524.7 ms     6176x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     443.3 ms     1752x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-     342.3 ms   231840x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     326.3 ms   231840x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     291.9 ms   229992x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)
-     280.9 ms   229068x  pi_attn::decode_merge_kernel(const float *, const float2 *, const int *, int, __nv_bfloat16 *, int)
-     262.9 ms   231840x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)

flashinfer: {"out_tok_s": 230.23, "req_s": 0.9, "ttft_ms": 8.04, "tpot_ms": 4.33, "itl_ms": 4.33, "p99_ttft_ms": 11.77, "e2e_ms": 1111.77, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [230.2]}

pi: {"out_tok_s": 224.33, "req_s": 0.88, "ttft_ms": 9.36, "tpot_ms": 4.44, "itl_ms": 4.44, "p99_ttft_ms": 8.47, "e2e_ms": 1140.96, "tokens": 8192, "in": 64, "out": 256, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [224.3]}
