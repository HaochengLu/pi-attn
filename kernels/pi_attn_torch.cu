// PyTorch bindings for the PI-Attn kernels (used by tests and benchmarks).
// The split planner uses shapes and device properties only; every split count
// yields the same bits, so planning never affects results.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "pi_attn_kernels.cu"

using namespace pi_attn;

static void check_inputs(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
                         std::initializer_list<at::Tensor> idx) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous(), "q: contiguous bf16 CUDA");
  TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && k_cache.is_contiguous() && k_cache.dim() == 4);
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes() && v_cache.is_contiguous() && v_cache.scalar_type() == at::kBFloat16);
  TORCH_CHECK(q.size(-1) == kHeadDim && k_cache.size(-1) == kHeadDim, "head_dim must be 128");
  const int hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(hkv > 0 && hq % hkv == 0, "num_qo_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(hq / hkv <= 16, "GQA group size must be <= 16 (one 16-row MMA group per kv head)");
  for (const auto& t : idx) TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous(), "index tensors: contiguous int32 CUDA");
}

static int page_shift_of(int page_size) {
  return page_size > 0 && (page_size & (page_size - 1)) == 0 ? __builtin_ctz(page_size) : -1;
}

static int sm_count(const at::Tensor& t) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, t.get_device());
  return sms;
}

// q: [total_q, Hq, 128]; k/v_cache: [pages, page_size, Hkv, 128]. kv_lens[b] >= q_len[b] >= 1.
// kv_splits: 0 = the engine planner's choice (prefill_plan), n >= 1 forces n
// splits over fold groups. variant: 0 = the engine's choice for this device and shape
// (prefill_qreg, prefill_warps), 1 = Q in smem, 2 = Q in registers (QREG), 3 / 4 = the same
// with 8 warps (128 rows) per CTA; all give the same bits.
at::Tensor prefill(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,
                   at::Tensor kv_page_indptr, at::Tensor kv_page_indices, at::Tensor kv_lens,
                   int64_t max_q_len, int64_t max_kv_len, double sm_scale, int64_t kv_splits,
                   int64_t variant) {
  check_inputs(q, k_cache, v_cache, {qo_indptr, kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(qo_indptr.size(0) == batch + 1 && kv_page_indptr.size(0) == batch + 1);
  TORCH_CHECK(max_q_len >= 1 && max_kv_len >= max_q_len);
  const int page_size = k_cache.size(1), group = hq / hkv;
  TORCH_CHECK(variant >= 0 && variant <= 4, "variant: 0 to 4");
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device());
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device());
  const bool qreg = variant ? variant % 2 == 0 : prefill_qreg(major, minor);
  const int warps = variant ? (variant >= 3 ? 8 : kPrefillWarps)
                            : prefill_warps(qreg, batch, hq, hkv, (int)max_q_len, sm_count(q)), rows = warps * 16;
  const int ctas_per_sm = warps == 8 && qreg ? 1 : 2;
  const int tiles = (int)((max_q_len * group + rows - 1) / rows);
  const int max_groups = (int)((max_kv_len - 1) / kGroupKeys + 1);
  int splits = (int)kv_splits, plan_groups = 0;
  if (splits == 0) prefill_plan(batch, hq, hkv, (int)max_q_len, (int)max_kv_len, sm_count(q),
                               prefill_tail_splits(major, minor), &splits, &plan_groups, rows, ctas_per_sm);
  splits = std::max(1, std::min(splits, max_groups));
  at::Tensor ws_o, ws_ml;
  if (splits > 1) {
    ws_o = at::empty({q.size(0), hq, max_groups, kHeadDim}, q.options().dtype(at::kFloat));
    ws_ml = at::empty({q.size(0), hq, max_groups, 2}, q.options().dtype(at::kFloat));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, int smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(tiles, hkv, batch * splits), warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(), kv_page_indices.data_ptr<int32_t>(),
        (int64_t)0, kv_lens.data_ptr<int32_t>(), nullptr, hq, hkv, page_size, page_shift_of(page_size),
        (float)(sm_scale * 1.4426950408889634), splits, splits > 1 ? ws_o.data_ptr<float>() : nullptr,
        splits > 1 ? (float2*)ws_ml.data_ptr<float>() : nullptr, max_groups);
  };
  if (warps == 8) {
    if (qreg) launch(prefill_kernel<true, 8>, prefill_smem_bytes<true, 8>());
    else launch(prefill_kernel<false, 8>, prefill_smem_bytes<false, 8>());
  } else if (qreg) {
    launch(prefill_kernel<true>, prefill_smem_bytes<true>());
  } else {
    launch(prefill_kernel<false>, prefill_smem_bytes<false>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    prefill_merge_kernel<<<dim3(hq, q.size(0)), kHeadDim, 0, stream>>>(
        ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), qo_indptr.data_ptr<int32_t>(),
        kv_lens.data_ptr<int32_t>(), nullptr, batch, (__nv_bfloat16*)out.data_ptr(), hq, max_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

// q: [batch, Hq, 128]. Workspace ws_o [rows, Hq, 128] f32, ws_ml [rows, Hq,
// 2 * DECODE_WARPS] f32 with rows >= batch * ceil(max_kv_len / GROUP_KEYS);
// max_kv_len bounds every request. counters: [rows, Hq, DECODE_WARPS] int32, zero
// before the first call and left zero by each.
at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                  at::Tensor kv_page_indices, at::Tensor kv_lens, at::Tensor ws_o, at::Tensor ws_ml,
                  at::Tensor counters, int64_t max_kv_len, double sm_scale) {
  check_inputs(q, k_cache, v_cache, {kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = q.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(kv_lens.size(0) == batch && kv_page_indptr.size(0) == batch + 1);
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.dim() == 3 && ws_o.size(1) == hq &&
              ws_o.size(2) == kHeadDim && ws_o.is_contiguous());
  TORCH_CHECK(ws_ml.scalar_type() == at::kFloat && ws_ml.dim() == 3 && ws_ml.size(0) == ws_o.size(0) &&
              ws_ml.size(2) == 2 * kDecodeWarps && ws_ml.is_contiguous());
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= ws_o.size(0) * hq * kDecodeWarps &&
              counters.is_contiguous());
  const int page_size = k_cache.size(1);
  const int max_groups = (int)((max_kv_len + kGroupKeys - 1) / kGroupKeys);
  TORCH_CHECK(ws_o.size(0) >= (int64_t)batch * max_groups, "decode workspace below batch * max_groups");
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, decode_smem_bytes());
  decode_kernel<<<decode_grid(max_groups, hkv, batch), kDecodeWarps * 32, decode_smem_bytes(), stream>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
      (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
      kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
      (float2*)ws_ml.data_ptr<float>(), counters.data_ptr<int32_t>(), (__nv_bfloat16*)out.data_ptr(),
      max_groups, hq, hkv, page_size, page_shift_of(page_size), (float)(sm_scale * 1.4426950408889634),
      decode_group_fastest(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &prefill, py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("qo_indptr"),
        py::arg("kv_page_indptr"), py::arg("kv_page_indices"), py::arg("kv_lens"), py::arg("max_q_len"),
        py::arg("max_kv_len"), py::arg("sm_scale"), py::arg("kv_splits"), py::arg("variant") = 0);
  m.def("decode", &decode);
  m.attr("TILE") = kTile;
  m.attr("DECODE_WARPS") = kDecodeWarps;
  m.attr("GROUP_KEYS") = kGroupKeys;
}
