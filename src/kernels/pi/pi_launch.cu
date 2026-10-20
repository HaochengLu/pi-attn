// C ABI for the PI (partition-invariant) kernels: attention and M-invariant GEMM.
// No torch dependency; callers pass raw device pointers and the CUDA stream.
// Every kernel variant, split count and GEMM tile produces identical bits, so
// the planners and the GEMM tuner only affect speed. Errors are returned as a
// message (nullptr on success).
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "gemm_cutlass.cuh"
#include "gemm_inv.cuh"
#include "pi_attn_kernels.cu"

using namespace pi_attn;
using namespace pi_gemm;

namespace {

thread_local std::string last_error;

const char* fail(const std::string& message) {
  last_error = message;
  return last_error.c_str();
}

const char* check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return fail(std::string(what) + ": " + cudaGetErrorString(err));
  return nullptr;
}

struct DeviceInfo {
  int sms = 0, smem_optin = 0;
};

DeviceInfo device_info() {
  int device = 0;
  cudaGetDevice(&device);
  static std::mutex mutex;
  static std::map<int, DeviceInfo> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(device);
  if (it != cache.end()) return it->second;
  DeviceInfo info;
  cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&info.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  cache[device] = info;
  return info;
}

int page_shift_of(int page_size) {
  return page_size > 0 && (page_size & (page_size - 1)) == 0 ? __builtin_ctz(page_size) : -1;
}

// Function attributes are per-kernel host state: set once (outside capture).
template <typename Kernel>
void set_smem(Kernel kernel, size_t smem) {
  static std::mutex mutex;
  static std::map<const void*, size_t> done;
  std::lock_guard<std::mutex> lock(mutex);
  auto& v = done[(const void*)kernel];
  if (v < smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    v = smem;
  }
}

}  // namespace

extern "C" {

const char* pi_last_error() { return last_error.c_str(); }

// ------------------------------------------------------------------ attention

// Decode planner: split CTAs so the batch's chunks are covered ~8 warps per SM.
// Depends only on (batch, kv heads, workspace chunk capacity, device): fixed per
// CUDA-graph bucket.
int pi_attn_decode_splits(int batch, int num_kv_heads, int max_chunks) {
  const int64_t want = (int64_t)8 * device_info().sms;
  const int64_t per = (int64_t)num_kv_heads * std::max(batch, 1);
  return (int)std::max<int64_t>(1, std::min<int64_t>(max_chunks, (want + per - 1) / per));
}

// q: [batch, Hq, 128] bf16 contiguous; out: [batch, Hq, 128] bf16; caches:
// [pages, page_size, Hkv, 128] bf16; page table: dense [batch, table_stride]
// int32; seqlens: [batch] int32 (KV length including the decoded token).
// ws_o: [batch, Hq, max_chunks, 128] f32, ws_ml: [batch, Hq, max_chunks, 2] f32.
const char* pi_attn_decode(const void* q, const void* k_cache, const void* v_cache,
                           const int32_t* page_table, int64_t table_stride, const int32_t* seqlens,
                           float* ws_o, float* ws_ml, int max_chunks, void* out, int batch,
                           int num_qo_heads, int num_kv_heads, int page_size, int splits,
                           double sm_scale, void* stream) {
  if (num_kv_heads <= 0 || num_qo_heads % num_kv_heads || num_qo_heads / num_kv_heads > 16)
    return fail("PI attention needs num_qo_heads = g * num_kv_heads with g <= 16");
  if (batch == 0) return nullptr;
  auto st = (cudaStream_t)stream;
  constexpr int W = 1, B = 1;  // 1 warp per CTA, one shared K/V tile: best on Ampere/Ada tested
  const size_t smem = decode_smem_elems<W, B>() * sizeof(__nv_bfloat16);
  set_smem(decode_partial_kernel<W, B>, smem);
  decode_partial_kernel<W, B><<<dim3(num_kv_heads, batch, splits), W * 32, smem, st>>>(
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache,
      nullptr, page_table, table_stride, seqlens, ws_o, (float2*)ws_ml, num_qo_heads,
      num_kv_heads, page_size, page_shift_of(page_size), max_chunks,
      (float)(sm_scale * 1.4426950408889634));
  if (auto e = check_launch("pi decode partials")) return e;
  decode_merge_kernel<<<dim3(num_qo_heads, batch), kHeadDim, 0, st>>>(
      ws_o, (const float2*)ws_ml, seqlens, (__nv_bfloat16*)out, num_qo_heads, max_chunks);
  return check_launch("pi decode merge");
}

// Prefill planner. Returns variant (2 = 8 warps x 16 rows; 4 = 8 warps x 32
// rows on big-smem GPUs) and the split count over fold groups (wave model).
void pi_attn_prefill_plan(int batch, int num_qo_heads, int num_kv_heads, int max_q_len,
                          int max_kv_len, int* variant, int* splits, int* max_groups) {
  const DeviceInfo dev = device_info();
  const int group = num_qo_heads / std::max(num_kv_heads, 1);
  const bool big = dev.smem_optin >= (int)(prefill_smem_elems<8, 2, 2>() * sizeof(__nv_bfloat16));
  const int64_t rows = (int64_t)max_q_len * group;
  const int64_t ctas256 = (rows + 255) / 256 * num_kv_heads * batch;
  *variant = big && ctas256 >= dev.sms ? 4 : 2;
  const int tile_rows = *variant == 4 ? 256 : 128;
  const int64_t ctas = (rows + tile_rows - 1) / tile_rows * num_kv_heads * batch;
  *max_groups = (max_kv_len - 1) / (kChunk * kGroupChunks) + 1;
  const int64_t chunks = (max_kv_len + kChunk - 1) / kChunk;
  int64_t best = INT64_MAX;
  *splits = 1;
  for (int sp = 1; sp <= *max_groups; ++sp) {
    const int64_t groups_per = (*max_groups + sp - 1) / sp;
    const int64_t per = std::min<int64_t>(chunks, groups_per * kGroupChunks);
    const int64_t waves = (ctas * sp + dev.sms - 1) / dev.sms;
    const int64_t cost = waves * per + (sp > 1 ? per / 8 : 0);
    if (cost < best) {
      best = cost;
      *splits = sp;
    }
  }
}

// q/out: [total_q, Hq, 128] bf16 contiguous; qo_indptr: [batch + 1] int32;
// prefix_lens: [batch] int32 (KV of request b = prefix + q_len, already in cache);
// page table dense [batch, table_stride]. ws_o/ws_ml ([total_q, Hq, max_groups, 128|2]
// f32) are required iff splits > 1.
const char* pi_attn_prefill(const void* q, const void* k_cache, const void* v_cache,
                            const int32_t* qo_indptr, const int32_t* page_table,
                            int64_t table_stride, const int32_t* prefix_lens, void* out, int batch,
                            int total_q, int num_qo_heads, int num_kv_heads, int page_size,
                            int max_q_len, double sm_scale, int variant, int splits, float* ws_o,
                            float* ws_ml, int max_groups, void* stream) {
  if (num_kv_heads <= 0 || num_qo_heads % num_kv_heads || num_qo_heads / num_kv_heads > 16)
    return fail("PI attention needs num_qo_heads = g * num_kv_heads with g <= 16");
  if (batch == 0 || total_q == 0) return nullptr;
  if (splits > 1 && (!ws_o || !ws_ml)) return fail("PI prefill split mode needs a workspace");
  auto st = (cudaStream_t)stream;
  const int group = num_qo_heads / num_kv_heads;
  auto launch = [&](auto kernel, int warps, int rows, size_t smem) {
    set_smem(kernel, smem);
    const int tiles = (int)(((int64_t)max_q_len * group + rows - 1) / rows);
    kernel<<<dim3(tiles, num_kv_heads, batch * splits), warps * 32, smem, st>>>(
        (const __nv_bfloat16*)q, (__nv_bfloat16*)out, (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache, qo_indptr, nullptr, page_table, table_stride, nullptr,
        prefix_lens, num_qo_heads, num_kv_heads, page_size, page_shift_of(page_size),
        (float)(sm_scale * 1.4426950408889634), splits, ws_o, (float2*)ws_ml, max_groups);
  };
  if (variant == 4) launch(prefill_kernel<8, 2, 2>, 8, 256, prefill_smem_elems<8, 2, 2>() * sizeof(__nv_bfloat16));
  else launch(prefill_kernel<8, 1, 2>, 8, 128, prefill_smem_elems<8, 1, 2>() * sizeof(__nv_bfloat16));
  if (auto e = check_launch("pi prefill")) return e;
  if (splits > 1) {
    prefill_merge_kernel<<<dim3(num_qo_heads, total_q), kHeadDim, 0, st>>>(
        ws_o, (const float2*)ws_ml, qo_indptr, nullptr, prefix_lens, batch, (__nv_bfloat16*)out,
        num_qo_heads, max_groups);
    return check_launch("pi prefill merge");
  }
  return nullptr;
}

}  // extern "C"

// ------------------------------------------------------------------ GEMM

namespace {

// Kernel ids: 1/2/4 small (4 warps, G = 1/2/4), 6..9 small narrow, 10..13 CUTLASS.
const std::vector<int>& small_kernels() {
  static const std::vector<int> v{1, 2, 4, 6, 7, 8, 9};
  return v;
}
const std::vector<int>& large_kernels() {
  static const std::vector<int> v{10, 11, 12, 13};
  return v;
}

int bucket_of(int m) {
  int b = 1;
  while (b < m && b < (1 << 20)) b <<= 1;
  return b;
}

std::mutex tune_mutex;
std::map<std::tuple<int, int, int>, int> tuned;  // (N, K, M bucket) -> kernel

int heuristic_kernel(int M) { return M <= 16 ? 1 : M <= 32 ? 2 : M <= 64 ? 13 : M <= 1024 ? 12 : 11; }

template <typename OutT>
cudaError_t run_gemm(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M,
                     int N, int K, cudaStream_t st) {
  auto small = [&](auto kern, int G, int W) {
    const size_t smem = (size_t)kSmallStages * (16 * G + 8 * W) * kSmallStride * sizeof(__nv_bfloat16);
    set_smem(kern, smem);
    kern<<<dim3(N / (8 * W), (M + 16 * G - 1) / (16 * G)), W * 32, smem, st>>>(
        (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K);
    return cudaGetLastError();
  };
  switch (k) {
    case 1: return small(gemm_small_kernel<1, 4, OutT>, 1, 4);
    case 2: return small(gemm_small_kernel<2, 4, OutT>, 2, 4);
    case 4: return small(gemm_small_kernel<4, 4, OutT>, 4, 4);
    case 6: return small(gemm_small_kernel<1, 1, OutT>, 1, 1);
    case 7: return small(gemm_small_kernel<1, 2, OutT>, 1, 2);
    case 8: return small(gemm_small_kernel<2, 1, OutT>, 2, 1);
    case 9: return small(gemm_small_kernel<4, 2, OutT>, 4, 2);
    case 10: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 11: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 12: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    default: return CutlassGemm<OutT, 64, 64, 32, 32, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, st);
  }
}

// Time each candidate kernel for one (N, K, M bucket) on scratch activations
// and remember the fastest. Called outside stream capture only.
const char* tune_bucket(const void* w, int N, int K, int m, cudaStream_t st) {
  void *x = nullptr, *y = nullptr;
  if (cudaMallocAsync(&x, (size_t)m * K * 2, st) != cudaSuccess ||
      cudaMallocAsync(&y, (size_t)m * N * 2, st) != cudaSuccess)
    return fail("PI linear tune: scratch allocation failed");
  cudaMemsetAsync(x, 0, (size_t)m * K * 2, st);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<int> cands = large_kernels();
  if (m <= 64) cands.insert(cands.begin(), small_kernels().begin(), small_kernels().end());
  float best = 1e30f;
  int best_k = heuristic_kernel(m);
  for (int k : cands) {
    if (run_gemm(k, x, K, w, (__nv_bfloat16*)y, N, m, N, K, st) != cudaSuccess) continue;  // warm-up
    cudaEventRecord(e0, st);
    for (int r = 0; r < 5; ++r) run_gemm(k, x, K, w, (__nv_bfloat16*)y, N, m, N, K, st);
    cudaEventRecord(e1, st);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (ms < best) {
      best = ms;
      best_k = k;
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFreeAsync(x, st);
  cudaFreeAsync(y, st);
  cudaStreamSynchronize(st);
  {
    std::lock_guard<std::mutex> lock(tune_mutex);
    tuned[{N, K, m}] = best_k;
  }
  return check_launch("pi linear tune");
}

}  // namespace

extern "C" {

// y[M, N] = x[M, K] w[N, K]^T with a row-invariant reduction order (see gemm_inv.cuh).
// x rows have stride ldx (elements); y is [M, N] contiguous; out_fp32 selects f32 output.
// The first call for an (N, K, M bucket) outside stream capture tunes the tile
// choice (all choices give identical bits); inside capture an untuned bucket
// uses a fixed heuristic.
const char* pi_linear(const void* x, int64_t ldx, const void* w, void* y, int M, int N, int K,
                      int out_fp32, void* stream) {
  if (K % 64 || N % 32) return fail("PI linear needs K % 64 == 0 and N % 32 == 0");
  if (M == 0) return nullptr;
  auto st = (cudaStream_t)stream;
  const int bucket = bucket_of(M);
  int k = -1;
  {
    std::lock_guard<std::mutex> lock(tune_mutex);
    auto it = tuned.find({N, K, bucket});
    if (it != tuned.end()) k = it->second;
  }
  if (k < 0) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(st, &status);
    if (status == cudaStreamCaptureStatusNone) {
      if (auto e = tune_bucket(w, N, K, bucket, st)) return e;
      std::lock_guard<std::mutex> lock(tune_mutex);
      k = tuned[{N, K, bucket}];
    } else {
      k = heuristic_kernel(M);
    }
  }
  if (M > 64 && k < 10) k = heuristic_kernel(M);
  const cudaError_t err = out_fp32 ? run_gemm(k, x, ldx, w, (float*)y, N, M, N, K, st)
                                   : run_gemm(k, x, ldx, w, (__nv_bfloat16*)y, N, M, N, K, st);
  if (err != cudaSuccess) return fail(std::string("PI linear: ") + cudaGetErrorString(err));
  return nullptr;
}

}  // extern "C"
