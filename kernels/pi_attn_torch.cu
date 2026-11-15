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

static int smem_optin(const at::Tensor& t) {
  int bytes = 0;
  cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, t.get_device());
  return bytes;
}

static int sm_count(const at::Tensor& t) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, t.get_device());
  return sms;
}

// q: [total_q, Hq, 128]; k/v_cache: [pages, page_size, Hkv, 128]. kv_lens[b] >= q_len[b] >= 1.
// kv_splits: 0 = the engine planner's choice (prefill_plan), n >= 1 forces n
// splits over fold groups. variant: 0 = the engine's choice for this device and shape
// (prefill_qreg, prefill_warps, prefill_wgmma, prefill90w_wg), 1 = Q in smem, 2 = Q in registers (QREG), 3 / 4 =
// the same with 8 warps (128 rows) per CTA; sm_90a builds on compute capability 9.0 only: 5 = the wgmma kernel
// of 64 rows per CTA, 6 / 7 = the warp-specialised wgmma kernel with 3 / 2 consumer warp groups (192 / 128 rows
// per CTA; the device's choice is one of the three, prefill90w_wg); all give the same bits.
// Prefill grid order of prefill_kernel: 0 = the engine's choice for this device and launch (prefill_order),
// 1 = ascending, 2 = longest first, 3 = descending; the same bits.
static int forced_prefill_order = 0;

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
  TORCH_CHECK(variant >= 0 && variant <= 7, "variant: 0 to 7");
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device());
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device());
  const bool wgmma = variant >= 5 || (variant == 0 && prefill_wgmma(major, minor));
  TORCH_CHECK(!wgmma || prefill_wgmma(major, minor), "variants 5 to 7 need an sm_90a build on compute capability 9.0");
  int nwg = variant == 5 ? 0 : variant == 6 ? 3 : 2;  // consumer warp groups; 0 = the 64-row wgmma kernel
#ifdef PI_SM90A
  if (variant == 0)
    nwg = prefill90w_wg(batch, hq, hkv, (int)max_q_len, (int)((max_kv_len - 1) / kGroupKeys + 1), sm_count(q));
#endif
  const bool ws = wgmma && nwg > 0;  // the warp-specialised wgmma kernel
  const bool qreg = wgmma ? false : variant ? variant % 2 == 0 : prefill_qreg(major, minor);
  const int warps = ws ? 4 * (1 + nwg) : wgmma ? kPrefillWarps : variant ? (variant >= 3 ? 8 : kPrefillWarps)
                            : prefill_warps(qreg, batch, hq, hkv, (int)max_q_len, sm_count(q));
  const int rows = ws ? 64 * nwg : warps * 16;
  const int ctas_per_sm = ws || (warps == 8 && qreg) ? 1 : 2;
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
  int l2_bytes = 0;
  cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, q.get_device());
  const int order = forced_prefill_order == 1 ? kPrefillAscending : forced_prefill_order == 2 ? kPrefillLongestFirst
                  : forced_prefill_order == 3 ? kPrefillDescending
                  : prefill_order(major, minor, batch, hkv, max_groups, splits, l2_bytes);
  dim3 grid = prefill_grid(tiles, hkv, batch * splits, order);
#ifdef PI_SM90A
  if (ws) grid = prefill90w_grid(tiles, hkv, batch * splits);
  else if (wgmma) grid = dim3(tiles, hkv, batch * splits);
#endif
  auto launch = [&](auto kernel, int smem, auto... order) {
    TORCH_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                "prefill variant ", variant, " needs ", smem, " bytes of shared memory, more than this GPU allows");
    kernel<<<grid, warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(), kv_page_indices.data_ptr<int32_t>(),
        (int64_t)0, kv_lens.data_ptr<int32_t>(), nullptr, hq, hkv, page_size, page_shift_of(page_size),
        (float)(sm_scale * 1.4426950408889634), splits, splits > 1 ? ws_o.data_ptr<float>() : nullptr,
        splits > 1 ? (float2*)ws_ml.data_ptr<float>() : nullptr, max_groups, order...);
  };
  if (wgmma) {
#ifdef PI_SM90A
    if (ws && nwg == 3) launch(prefill90w_kernel<3>, prefill90w_smem_bytes<3>());
    else if (ws) launch(prefill90w_kernel<2>, prefill90w_smem_bytes<2>());
    else launch(prefill90_kernel, prefill90_smem_bytes());
#endif
  } else if (warps == 8) {
    if (qreg) launch(prefill_kernel<true, 8>, prefill_smem_bytes<true, 8>(), prefill_tile_flip(order, tiles));
    else launch(prefill_kernel<false, 8>, prefill_smem_bytes<false, 8>(), prefill_tile_flip(order, tiles));
  } else if (qreg) {
    launch(prefill_kernel<true>, prefill_smem_bytes<true>(), prefill_tile_flip(order, tiles));
  } else {
    launch(prefill_kernel<false>, prefill_smem_bytes<false>(), prefill_tile_flip(order, tiles));
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

// Decode kernel: 0 = the engine's choice for this device, batch and bound of the kv lengths (decode_plan),
// 1 = decode_kernel, 2 = decode90_kernel wherever it can run (sm_90a builds on compute capability 9.0,
// batch * kv heads <= SMs), else decode_kernel, 3 = decode16_kernel wherever it can run (138 KB of shared memory,
// batch * kv heads <= SMs, GQA group <= 8), else decode_kernel, 4 = the same without helper CTAs; all give the
// same bits.
static int decode_variant = 0;

// q: [batch, Hq, 128]. Workspace ws_o [rows, Hq, 128] f32, ws_ml [rows, Hq,
// 2 * DECODE_WARPS] f32 with rows >= batch * ceil(max_kv_len / GROUP_KEYS);
// max_kv_len bounds every request. counters: [rows, Hq, DECODE_WARPS] int32, zero
// before the first call and left zero by each. kv_len_bound: the longest request of the call if the caller
// knows it (as the engine does), for decode_plan; 0 = max_kv_len.
at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                  at::Tensor kv_page_indices, at::Tensor kv_lens, at::Tensor ws_o, at::Tensor ws_ml,
                  at::Tensor counters, int64_t max_kv_len, double sm_scale, int64_t kv_len_bound) {
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
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device());
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, q.get_device());
  const float scale_log2 = (float)(sm_scale * 1.4426950408889634);
  int kernel = decode_variant == 0 ? decode_plan(major, minor, batch, max_groups, hq, hkv, sm_count(q), kv_len_bound) : 1;
  if (kernel == 3 || (decode_variant >= 3 && decode16_layout(batch, hkv, max_groups, sm_count(q)).slots >= 1 &&
                      hq <= kD16MaxGroup * hkv && decode16_smem_bytes() <= smem_optin(q))) {
    const Decode16Layout lay = decode16_layout(batch, hkv, max_groups, sm_count(q), decode_variant != 4);
    TORCH_CHECK(cudaFuncSetAttribute(decode16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, decode16_smem_bytes()) == cudaSuccess,
                "decode16_kernel needs ", decode16_smem_bytes(), " bytes of shared memory, more than this GPU allows");
    decode16_kernel<<<decode16_grid(lay, hkv, batch), kD16Warps * 32, decode16_smem_bytes(), stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
        (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
        (float2*)ws_ml.data_ptr<float>(), counters.data_ptr<int32_t>(), (__nv_bfloat16*)out.data_ptr(),
        max_groups, hq, hkv, page_size, page_shift_of(page_size), scale_log2, decode_group_fastest(batch), batch,
        lay.slots, lay.helper_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
#ifdef PI_SM90A
  if (decode_variant == 2 && major == 9 && minor == 0 && decode90_slots(batch, hkv, max_groups, sm_count(q)) >= 1)
    kernel = 2;
  if (kernel == 2) {
    const int slots = decode90_slots(batch, hkv, max_groups, sm_count(q));
    cudaFuncSetAttribute(decode90_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, decode90_smem_bytes());
    decode90_kernel<<<decode90_grid(slots, hkv, batch), kD90Warps * 32, decode90_smem_bytes(), stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
        (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
        (float2*)ws_ml.data_ptr<float>(), counters.data_ptr<int32_t>(), (__nv_bfloat16*)out.data_ptr(),
        max_groups, hq, hkv, page_size, page_shift_of(page_size), scale_log2, decode_group_fastest(batch), slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
#endif
  cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, decode_smem_bytes());
  decode_kernel<<<decode_grid(max_groups, hkv, batch), kDecodeWarps * 32, decode_smem_bytes(), stream>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
      (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
      kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
      (float2*)ws_ml.data_ptr<float>(), counters.data_ptr<int32_t>(), (__nv_bfloat16*)out.data_ptr(),
      max_groups, hq, hkv, page_size, page_shift_of(page_size), scale_log2, decode_group_fastest(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &prefill, py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("qo_indptr"),
        py::arg("kv_page_indptr"), py::arg("kv_page_indices"), py::arg("kv_lens"), py::arg("max_q_len"),
        py::arg("max_kv_len"), py::arg("sm_scale"), py::arg("kv_splits"), py::arg("variant") = 0);
  m.def("decode", &decode, py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("kv_page_indptr"),
        py::arg("kv_page_indices"), py::arg("kv_lens"), py::arg("ws_o"), py::arg("ws_ml"), py::arg("counters"),
        py::arg("max_kv_len"), py::arg("sm_scale"), py::arg("kv_len_bound") = 0);
  m.def("set_decode_variant", [](int64_t v) {
    TORCH_CHECK(v >= 0 && v <= 4, "decode variant: 0 to 4");
    decode_variant = (int)v;
  });
  m.def("set_prefill_order", [](int64_t v) {
    TORCH_CHECK(v >= 0 && v <= 3, "prefill order: 0 to 3");
    forced_prefill_order = (int)v;
  });
  // The kernel decode_plan gives a launch on this device (1, 2 or 3).
  m.def("decode_plan", [](int64_t batch, int64_t hkv, int64_t max_kv_len, int64_t kv_len_bound, int64_t hq) {
    if (hq <= 0) hq = hkv;
    int dev = 0, major = 0, minor = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return decode_plan(major, minor, (int)batch, (int)((max_kv_len + kGroupKeys - 1) / kGroupKeys), (int)hq, (int)hkv,
                       sms, kv_len_bound);
  }, py::arg("batch"), py::arg("hkv"), py::arg("max_kv_len"), py::arg("kv_len_bound"), py::arg("hq") = 0);
  // Tokens of a single request's prefill that the engine runs as a first launch (prefill_wave_head; 0: one launch).
  m.def("prefill_wave_head", [](int64_t hq, int64_t hkv, int64_t q_len, int64_t prefix) {
    int dev = 0, major = 0, minor = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return prefill_wave_head(major, minor, (int)hq, (int)hkv, (int)q_len, (int)prefix, sms);
  });
  // Shared memory a prefill variant (1-4) needs per CTA; tests skip variants the GPU cannot run.
  m.def("prefill_smem_bytes", [](int64_t variant) {
    TORCH_CHECK(variant >= 1 && variant <= 4, "variant: 1 to 4");
    return variant == 1 ? prefill_smem_bytes<false>() : variant == 2 ? prefill_smem_bytes<true>()
         : variant == 3 ? prefill_smem_bytes<false, 8>() : prefill_smem_bytes<true, 8>();
  });
  // Whether this build has the wgmma prefill kernels (variants 5 to 7; the device's choice on compute capability
  // 9.0 is one of them, by shape).
#ifdef PI_SM90A
  m.attr("WGMMA") = true;
#else
  m.attr("WGMMA") = false;
#endif
  // Whether this GPU can run decode16_kernel (decode variants 3 and 4).
  m.def("decode16_fits", []() {
    int dev = 0, bytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return decode16_smem_bytes() <= bytes;
  });
  m.attr("TILE") = kTile;
  m.attr("DECODE_WARPS") = kDecodeWarps;
  m.attr("GROUP_KEYS") = kGroupKeys;
}
