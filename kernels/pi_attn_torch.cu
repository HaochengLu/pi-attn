// PyTorch bindings for the PI-Attn kernels (used by tests and benchmarks).
// Planners pick configs from shapes and device properties only; every config
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
static int smem_optin(const at::Tensor& t) {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, t.get_device());
  return v;
}

// q: [total_q, Hq, 128]; k/v_cache: [pages, page_size, Hkv, 128]. kv_lens[b] >= q_len[b] >= 1.
// variant: 0 = planner, 1 = 4w x16 rows, 2 = 8w x16, 3 = 4w x32, 4 = 8w x32 (big smem),
// 5 = 8w x16 with Q in smem and a single V buffer, 6 = 4w x16 with Q in smem.
// kv_splits: 0 = planner, n >= 1 forces n splits over fold groups.
at::Tensor prefill(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor qo_indptr,
                   at::Tensor kv_page_indptr, at::Tensor kv_page_indices, at::Tensor kv_lens,
                   int64_t max_q_len, int64_t max_kv_len, double sm_scale, int64_t variant,
                   int64_t kv_splits) {
  check_inputs(q, k_cache, v_cache, {qo_indptr, kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = kv_lens.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(qo_indptr.size(0) == batch + 1 && kv_page_indptr.size(0) == batch + 1);
  TORCH_CHECK(max_q_len >= 1 && max_kv_len >= max_q_len);
  const int page_size = k_cache.size(1);
  const int sms = sm_count(q);
  int config = (int)variant;
  if (config == 0) {
    const bool big = smem_optin(q) >= (int)(prefill_smem_elems<8, 2, 2>() * sizeof(__nv_bfloat16));
    const int64_t ctas256 = (int64_t)((max_q_len * (hq / hkv) + 255) / 256) * hkv * batch;
    if (big && ctas256 >= sms) config = 4;
    else config = 2;
  }
  const int rows = config == 1 || config == 6 ? 64 : config == 4 ? 256 : 128;
  const int group = hq / hkv;
  const int tiles = (int)((max_q_len * group + rows - 1) / rows);
  const int max_groups = (max_kv_len - 1) / (kChunk * kGroupChunks) + 1;
  int splits = (int)kv_splits;
  if (splits == 0) {
    // Wave model: one prefill CTA per SM; a CTA's time ~ chunks it folds. Pick
    // the split count over fold groups that minimises waves x chunks per split
    // (ties go to fewer splits: less workspace traffic).
    const int64_t ctas = (int64_t)tiles * hkv * batch;
    const int64_t chunks = (max_kv_len + kChunk - 1) / kChunk;
    int64_t best = INT64_MAX;
    splits = 1;
    for (int sp = 1; sp <= max_groups; ++sp) {
      const int64_t groups_per = (max_groups + sp - 1) / sp;
      const int64_t per = std::min<int64_t>(chunks, groups_per * kGroupChunks);
      const int64_t waves = (ctas * sp + sms - 1) / sms;
      const int64_t cost = waves * per + (sp > 1 ? per / 8 : 0);  // + merge / workspace
      if (cost < best) { best = cost; splits = sp; }
    }
  }
  splits = std::max(1, std::min(splits, max_groups));
  at::Tensor ws_o, ws_ml;
  if (splits > 1) {
    ws_o = at::empty({q.size(0), hq, max_groups, kHeadDim}, q.options().dtype(at::kFloat));
    ws_ml = at::empty({q.size(0), hq, max_groups, 2}, q.options().dtype(at::kFloat));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel, int warps, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(tiles, hkv, batch * splits), warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)k_cache.data_ptr(), (const __nv_bfloat16*)v_cache.data_ptr(),
        qo_indptr.data_ptr<int32_t>(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), nullptr, hq, hkv,
        page_size, page_shift_of(page_size), (float)(sm_scale * 1.4426950408889634), splits,
        splits > 1 ? ws_o.data_ptr<float>() : nullptr,
        splits > 1 ? (float2*)ws_ml.data_ptr<float>() : nullptr, max_groups);
  };
#define PI_LAUNCH(W, G, VB) launch(prefill_kernel<W, G, VB>, W, prefill_smem_elems<W, G, VB>() * sizeof(__nv_bfloat16))
#define PI_LAUNCH_QS(W, VB) launch(prefill_kernel<W, 1, VB, true>, W, prefill_smem_elems<W, 1, VB, true>() * sizeof(__nv_bfloat16))
  if (config == 6) PI_LAUNCH_QS(4, 2);
  else if (config == 5) PI_LAUNCH_QS(8, 1);
  else if (config == 4) PI_LAUNCH(8, 2, 2);
  else if (config == 3) PI_LAUNCH(4, 2, 1);
  else if (config == 2) PI_LAUNCH(8, 1, 2);
  else PI_LAUNCH(4, 1, 2);
#undef PI_LAUNCH
#undef PI_LAUNCH_QS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    prefill_merge_kernel<<<dim3(hq, q.size(0)), kHeadDim, 0, stream>>>(
        ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), qo_indptr.data_ptr<int32_t>(),
        kv_lens.data_ptr<int32_t>(), nullptr, batch, (__nv_bfloat16*)out.data_ptr(), hq, max_groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

// q: [batch, Hq, 128]. Compact workspace: ws_o [capacity, Hq, 128] f32, ws_ml [capacity, Hq, 2] f32,
// with capacity >= batch * ceil(max_chunks / 32); below sum of ceil(kv_len / 64) the
// kernels run grouped. max_chunks bounds one request.
// splits/variant: 0 = planner.
at::Tensor decode(at::Tensor q, at::Tensor k_cache, at::Tensor v_cache, at::Tensor kv_page_indptr,
                  at::Tensor kv_page_indices, at::Tensor kv_lens, at::Tensor ws_o, at::Tensor ws_ml,
                  int64_t max_chunks, int64_t splits, double sm_scale, int64_t variant) {
  check_inputs(q, k_cache, v_cache, {kv_page_indptr, kv_page_indices, kv_lens});
  const c10::cuda::CUDAGuard guard(q.device());
  auto out = at::empty_like(q);
  const int batch = q.size(0), hq = q.size(1), hkv = k_cache.size(2);
  TORCH_CHECK(kv_lens.size(0) == batch && kv_page_indptr.size(0) == batch + 1);
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.dim() == 3 && ws_o.size(1) == hq &&
              ws_o.size(2) == kHeadDim && ws_o.is_contiguous());
  TORCH_CHECK(ws_ml.scalar_type() == at::kFloat && ws_ml.dim() == 3 && ws_ml.size(0) == ws_o.size(0) &&
              ws_ml.size(2) == 2 && ws_ml.is_contiguous());
  const int page_size = k_cache.size(1), capacity = ws_o.size(0);
  const int max_groups = (int)((max_chunks + kGroupChunks - 1) / kGroupChunks);
  TORCH_CHECK((int64_t)capacity >= (int64_t)batch * max_groups, "decode workspace below batch * max_groups");
  auto chunk_offsets = at::empty({batch + 1}, kv_lens.options());
  int var = (int)variant;
  if (var == 0) var = 1;
  int sp = (int)splits;
  if (sp == 0) {  // enough warps to cover the chunks of the batch about twice per SM
    const int warps_per_cta = var == 3 || var == 5 ? 4 : var == 1 ? 1 : 2;
    const int64_t want = (int64_t)8 * sm_count(q);
    sp = (int)std::max<int64_t>(1, std::min<int64_t>((max_chunks + warps_per_cta - 1) / warps_per_cta,
                                                     (want + (int64_t)hkv * batch - 1) / ((int64_t)hkv * batch)));
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  decode_chunk_offsets_kernel<<<1, 32, 0, stream>>>(kv_lens.data_ptr<int32_t>(), batch,
                                                    chunk_offsets.data_ptr<int32_t>());
  auto launch = [&](auto kernel, int warps, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<dim3(hkv, batch, sp), warps * 32, smem, stream>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k_cache.data_ptr(),
        (const __nv_bfloat16*)v_cache.data_ptr(), kv_page_indptr.data_ptr<int32_t>(),
        kv_page_indices.data_ptr<int32_t>(), (int64_t)0, kv_lens.data_ptr<int32_t>(), ws_o.data_ptr<float>(),
        (float2*)ws_ml.data_ptr<float>(), chunk_offsets.data_ptr<int32_t>(), capacity, max_groups, hq, hkv,
        page_size, page_shift_of(page_size), (float)(sm_scale * 1.4426950408889634));
  };
#define PI_DLAUNCH(W, B) launch(decode_partial_kernel<W, B>, W, decode_smem_elems<W, B>() * sizeof(__nv_bfloat16))
  // variant: 1 = 1 warp/1 tile, 2 = 2 warps/1 tile, 3 = 4 warps/1 tile, 4 = 2 warps/2 tiles, 5 = 4 warps/2 tiles
  switch (var) {
    case 2: PI_DLAUNCH(2, 1); break;
    case 3: PI_DLAUNCH(4, 1); break;
    case 4: PI_DLAUNCH(2, 2); break;
    case 5: PI_DLAUNCH(4, 2); break;
    default: PI_DLAUNCH(1, 1); break;
  }
#undef PI_DLAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  decode_merge_kernel<<<dim3(hq, batch), kHeadDim, 0, stream>>>(
      ws_o.data_ptr<float>(), (const float2*)ws_ml.data_ptr<float>(), kv_lens.data_ptr<int32_t>(),
      chunk_offsets.data_ptr<int32_t>(), capacity, max_groups, (__nv_bfloat16*)out.data_ptr(), hq);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prefill", &prefill);
  m.def("decode", &decode);
  m.attr("CHUNK") = kChunk;
  m.attr("GROUP_CHUNKS") = kGroupChunks;
}
