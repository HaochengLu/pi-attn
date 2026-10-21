// PERF PROBE binding for prefill2 (not part of the engine yet).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "pi_attn_kernels.cu"
#include "prefill2.cuh"
using namespace pi_attn;
static int pshift(int p) { int s = 0; while ((1 << s) < p) ++s; return (1 << s) == p ? s : -1; }

at::Tensor prefill2(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,
                    at::Tensor kv_page_indptr, at::Tensor kv_page_indices, at::Tensor kv_lens,
                    int64_t max_q_len, int64_t max_kv_len, double sm_scale, int64_t splits, int64_t gr) {
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2), page_size = k_cache.size(1);
  const int group = hq / hkv;
  const int rows = gr == 1 ? 64 : 128;
  const int tiles = (int)((max_q_len * group + rows - 1) / rows);
  const int max_groups = (int)((max_kv_len + 511) / 512);
  splits = std::max<int64_t>(1, std::min<int64_t>(splits, max_groups));
  auto f32 = q.options().dtype(at::kFloat);
  at::Tensor ws_tot = at::empty({q.size(0), hq, kHeadDim}, f32), ws_o, ws_ml;
  if (splits > 1) {
    ws_o = at::empty({q.size(0), hq, max_groups, kHeadDim}, f32);
    ws_ml = at::empty({q.size(0), hq, max_groups, 2}, f32);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(tiles, hkv, batch * splits), 128, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
        (const __nv_bfloat16*)v_cache.data_ptr(), qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), nullptr, hq, hkv, page_size,
        pshift(page_size), (float)(sm_scale * 1.4426950408889634), (int)splits,
        splits > 1 ? ws_o.data_ptr<float>() : nullptr, splits > 1 ? (float2*)ws_ml.data_ptr<float>() : nullptr,
        max_groups, ws_tot.data_ptr<float>());
  };
  if (gr == 1) launch(prefill2_kernel<1>, prefill2_smem_bytes<1>());
  else launch(prefill2_kernel<2>, prefill2_smem_bytes<2>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    // NOTE: prefill_merge_kernel folds with kGroupChunks*kChunk = 512-key groups: same grouping.
    prefill_merge_kernel<<<dim3(hq, q.size(0)), kHeadDim, 0, stream>>>(
        ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), qo_indptr.data_ptr<int32_t>(),
        kv_lens.data_ptr<int32_t>(), nullptr, batch, (__nv_bfloat16*)out.data_ptr(), hq, max_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
at::Tensor decode2(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                   at::Tensor kv_page_indices, at::Tensor kv_lens, int64_t max_kv_len, double sm_scale) {
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = q.size(0), hq = q.size(1), hkv = k_cache.size(2), page_size = k_cache.size(1);
  const int max_groups = (int)((max_kv_len + 511) / 512);
  auto f32 = q.options().dtype(at::kFloat);
  auto ws_o = at::empty({(int64_t)batch * max_groups, hq, kHeadDim}, f32);
  auto ws_ml = at::empty({(int64_t)batch * max_groups, hq, 2}, f32);
  const size_t smem = decode2_smem_bytes();
  cudaFuncSetAttribute(decode2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  auto stream = at::cuda::getCurrentCUDAStream();
  decode2_kernel<<<dim3(max_groups, hkv, batch), kD2Warps * 32, smem, stream>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
      (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
      kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
      (float2*)ws_ml.data_ptr<float>(), max_groups, hq, hkv, page_size, pshift(page_size),
      (float)(sm_scale * 1.4426950408889634));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  decode_merge_kernel<<<dim3(hq, batch), kHeadDim, 0, stream>>>(
      ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), kv_lens.data_ptr<int32_t>(), max_groups,
      (__nv_bfloat16*)out.data_ptr(), hq);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill2", &prefill2);
  m.def("decode2", &decode2);
}
