// PyTorch bindings for the PI-Attn kernels (used by tests and benchmarks).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "pi_attn_kernels.cu"

using namespace pi_attn;

static void check_common(const at::Tensor& q, const at::Tensor& k_cache) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous());
  TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && k_cache.is_contiguous());
  TORCH_CHECK(q.size(-1) == kHeadDim && k_cache.size(-1) == kHeadDim, "head_dim must be 128");
}

// q: [total_q, Hq, 128]; k/v_cache: [pages, page_size, Hkv, 128].
at::Tensor prefill(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,
                   at::Tensor kv_page_indptr, at::Tensor kv_page_indices, at::Tensor kv_lens,
                   int64_t max_q_len, double sm_scale, int64_t warps_override) {
  check_common(q, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  const int page_size = k_cache.size(1);
  const size_t smem = kPrefillSmemElems * sizeof(__nv_bfloat16);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, q.get_device());
  auto launch = [&](auto kernel, int warps) {
    const int tiles = (max_q_len + warps * 16 - 1) / (warps * 16);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(tiles, hq, batch), warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(), hq, hkv, page_size,
        (float)(sm_scale * 1.4426950408889634));
  };
  // 8 warps (128 rows) when that still yields >= 2 CTAs per SM worth of work.
  const int64_t ctas8 = (int64_t)((max_q_len + 127) / 128) * hq * batch;
  if (warps_override == 8 || (warps_override == 0 && ctas8 >= 2 * sms)) launch(prefill_kernel<8>, 8);
  else launch(prefill_kernel<4>, 4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// q: [batch, Hq, 128]. ws_o: [batch, Hq, max_chunks, 128] f32, ws_ml: [batch, Hq, max_chunks, 2].
at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                  at::Tensor kv_page_indices, at::Tensor kv_lens, at::Tensor ws_o,
                  at::Tensor ws_ml, int64_t splits, double sm_scale) {
  check_common(q, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = q.size(0), hq = q.size(1), hkv = k_cache.size(2);
  const int page_size = k_cache.size(1), max_chunks = ws_o.size(2);
  const size_t smem = kDecodeSmemElems * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(decode_partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_partial_kernel<<<dim3(hkv, batch, splits), kDecodeWarps * 32, smem, stream>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
      (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
      kv_page_indices.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
      (float2*)ws_ml.data_ptr<float>(), hq, hkv, page_size, max_chunks,
      (float)(sm_scale * 1.4426950408889634));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  decode_merge_kernel<<<dim3(hq, batch), kHeadDim, 0, stream>>>(
      ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), kv_lens.data_ptr<int32_t>(),
      (__nv_bfloat16*)out.data_ptr(), hq, max_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &prefill);
  m.def("decode", &decode);
  m.attr("CHUNK") = kChunk;
}
