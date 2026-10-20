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
                   int64_t max_q_len, double sm_scale, int64_t variant) {
  check_common(q, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  const int page_size = k_cache.size(1);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, q.get_device());
  auto launch = [&](auto kernel, int warps, int rows, size_t smem) {
    const int tiles = (max_q_len + rows - 1) / rows;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(tiles, hq, batch), warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(), hq, hkv, page_size,
        (float)(sm_scale * 1.4426950408889634));
  };
#define PI_LAUNCH(W, G, VB) \
  launch(prefill_kernel<W, G, VB>, W, W * 16 * G, prefill_smem_elems<W, G, VB>() * sizeof(__nv_bfloat16))
  // config: 1 = 4 warps x 16 rows, 2 = 8 warps x 16 rows, 3 = 4 warps x 32 rows,
  // 4 = 8 warps x 32 rows (needs ~139 KB smem: A100/A800/H100 class).
  int config = (int)variant;
  if (config == 0) {
    int smem_optin = 0;
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, q.get_device());
    const bool big = smem_optin >= (int)(prefill_smem_elems<8, 2, 2>() * sizeof(__nv_bfloat16));
    const int64_t ctas128 = (int64_t)((max_q_len + 127) / 128) * hq * batch;
    const int64_t ctas256 = (int64_t)((max_q_len + 255) / 256) * hq * batch;
    if (big && ctas256 >= sms) config = 4;
    else config = ctas128 >= sms / 2 ? 2 : 1;
  }
  if (config == 4) PI_LAUNCH(8, 2, 2);
  else if (config == 3) PI_LAUNCH(4, 2, 1);
  else if (config == 2) PI_LAUNCH(8, 1, 2);
  else PI_LAUNCH(4, 1, 2);
#undef PI_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// q: [batch, Hq, 128]. ws_o: [batch, Hq, max_chunks, 128] f32, ws_ml: [batch, Hq, max_chunks, 2].
at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                  at::Tensor kv_page_indices, at::Tensor kv_lens, at::Tensor ws_o,
                  at::Tensor ws_ml, int64_t splits, double sm_scale, int64_t variant) {
  check_common(q, k_cache);
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = q.size(0), hq = q.size(1), hkv = k_cache.size(2);
  const int page_size = k_cache.size(1), max_chunks = ws_o.size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, int warps, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(hkv, batch, splits), warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
        (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
        (float2*)ws_ml.data_ptr<float>(), hq, hkv, page_size, max_chunks,
        (float)(sm_scale * 1.4426950408889634));
  };
#define PI_DLAUNCH(W, B) launch(decode_partial_kernel<W, B>, W, decode_smem_elems<W, B>() * sizeof(__nv_bfloat16))
  // variant: 1 = 1 warp/1 tile, 2 = 2 warps/1 tile, 3 = 4 warps/1 tile, 4 = 2 warps/2 tiles, 5 = 4 warps/2 tiles
  switch ((int)variant) {
    case 1: PI_DLAUNCH(1, 1); break;
    case 3: PI_DLAUNCH(4, 1); break;
    case 4: PI_DLAUNCH(2, 2); break;
    case 5: PI_DLAUNCH(4, 2); break;
    default: PI_DLAUNCH(2, 1); break;
  }
#undef PI_DLAUNCH
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
