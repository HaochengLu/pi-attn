# Qwen3-1.7B long: 30000:128:1

| kernel time (ms) | flashinfer | pi |
|---|---|---|
| GEMM | 34514.3 | 36118.9 |
| attention | 48140.6 | 811.2 |
| sampling | 25.7 | 22.9 |
| norm | 256.1 | 237.2 |
| reduction | 0.0 | 0.0 |
| other | 1826.7 | 59224.2 |
| total | 84763.5 | 96414.4 |

## top kernels: flashinfer

-   32435.0 ms    12180x  void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::MaskMode)1, (unsigne
-   14954.1 ms   437288x  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
-   14313.4 ms   108576x  void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, (unsigned int)2, (unsigned 
-   13929.3 ms    25928x  ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn
-    3498.6 ms    14812x  ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn
-    1927.0 ms    16215x  void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(T1::Params)
-     904.4 ms   123472x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)
-     789.4 ms   246944x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     615.9 ms   123472x  void <unnamed>::rope_kernel<__nv_bfloat16>(T1 *, T1 *, const long *, const float *, const float *, long, long,
-     411.2 ms   108576x  void flashinfer::PersistentVariableLengthMergeStatesKernel<(unsigned int)8, (unsigned int)16, (unsigned int)8,
-     256.1 ms   123472x  void sglang::fused_qknorm_warp<(long)128, (bool)0, __nv_bfloat16>(sglang::QKNormParams)
-     231.8 ms   121624x  void sglang::store_kvcache<(long)2048, (long)2048, (int)4, (bool)0, int>(sglang::StoreKVCacheParams)

## top kernels: pi

-   35473.0 ms    13048x  void pi_attn::prefill_kernel<(int)8, (int)1, (int)2, (bool)0>(const __nv_bfloat16 *, __nv_bfloat16 *, const __
-   17724.6 ms    40812x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-   14959.2 ms   109564x  void pi_attn::decode_partial_kernel<(int)1, (int)1>(const __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_b
-    6582.8 ms   114004x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)256, (int)3>(const __nv_bfloat16 *, long, 
-    6025.2 ms   110104x  void pi_gemm::gemm_small_kernel<(int)2, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    4978.6 ms   109564x  pi_attn::decode_merge_kernel(const float *, const float2 *, const int *, const int *, int, int, __nv_bfloat16 
-    2152.6 ms   109936x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)64, (int)4>(const __nv_bfloat16 *, long, c
-    2101.8 ms    17404x  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::gemm::threadblock::MmaMultistage<cutlass::gemm::Gemm
-    1961.8 ms    12068x  pi_attn::prefill_merge_kernel(const float *, const float2 *, const int *, const int *, const int *, int, __nv_
-    1211.4 ms   110832x  void pi_gemm::gemm_small_kernel<(int)1, (int)4, __nv_bfloat16, (int)128, (int)6>(const __nv_bfloat16 *, long, 
-     801.5 ms   248920x  void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, T2 *, T2 *, unsigned int, u
-     739.3 ms   124460x  void <unnamed>::silu_mul_vector_kernel<__nv_bfloat16, (long)0>(const T1 *, T1 *, long, long)

flashinfer: {"out_tok_s": 46.21, "req_s": 0.36, "ttft_ms": 1759.36, "tpot_ms": 7.95, "itl_ms": 7.95, "p99_ttft_ms": 1785.24, "e2e_ms": 2769.51, "tokens": 4096, "in": 30000, "out": 128, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [46.2]}

pi: {"out_tok_s": 41.19, "req_s": 0.32, "ttft_ms": 1909.7, "tpot_ms": 9.43, "itl_ms": 9.43, "p99_ttft_ms": 1915.61, "e2e_ms": 3107.71, "tokens": 4096, "in": 30000, "out": 128, "concurrency": 1, "shared": 0, "repeats": 1, "out_tok_s_all": [41.2]}
