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

// Raw launches go to the calling thread's current device: make it the tensors' device.
void pi_set_device(int device) {
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) cudaSetDevice(device);
}

// ------------------------------------------------------------------ attention

// q: [batch, Hq, 128] bf16 contiguous; out: [batch, Hq, 128] bf16; caches:
// [pages, page_size, Hkv, 128] bf16; page table: dense [batch, table_stride]
// int32; seqlens: [batch] int32 (KV length including the decoded token).
// Workspace: ws_o [batch * max_groups, Hq, 128] f32 and ws_ml [.., Hq, 8] f32 (one
// (m, l) pair per decode warp) with max_groups = pi_attn_decode_groups(table_stride
// * page_size), and counters [ws_rows, Hq, 4] int32, zero before the first call;
// every call leaves them zero. The grid depends
// only on (batch, table_stride), so it is fixed per CUDA-graph bucket.
static_assert(kDecodeWarps == 4, "pi.rs sizes the decode ml and counter workspaces for 4 warps");
int pi_attn_decode_groups(int64_t max_kv_len) {
  return (int)((max_kv_len + kGroupKeys - 1) / kGroupKeys);
}

const char* pi_attn_decode(const void* q, const void* k_cache, const void* v_cache,
                           const int32_t* page_table, int64_t table_stride, const int32_t* seqlens,
                           float* ws_o, float* ws_ml, int* counters, int64_t ws_rows, void* out, int batch,
                           int num_qo_heads, int num_kv_heads, int page_size, double sm_scale,
                           void* stream) {
  if (num_kv_heads <= 0 || num_qo_heads % num_kv_heads || num_qo_heads / num_kv_heads > 16)
    return fail("PI attention needs num_qo_heads = g * num_kv_heads with g <= 16");
  if (batch == 0) return nullptr;
  const int max_groups = pi_attn_decode_groups(table_stride * page_size);
  if (ws_rows < (int64_t)batch * max_groups)
    return fail("PI decode workspace smaller than batch * fold groups per request");
  auto st = (cudaStream_t)stream;
  set_smem(decode_kernel, decode_smem_bytes());
  decode_kernel<<<dim3(max_groups, num_kv_heads, batch), kDecodeWarps * 32, decode_smem_bytes(), st>>>(
      (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache,
      nullptr, page_table, table_stride, seqlens, ws_o, (float2*)ws_ml, counters, (__nv_bfloat16*)out,
      max_groups, num_qo_heads, num_kv_heads, page_size, page_shift_of(page_size),
      (float)(sm_scale * 1.4426950408889634));
  return check_launch("pi decode");
}

// Prefill planner: split count over fold groups and max_groups for the split
// workspace (prefill_plan in pi_attn_kernels.cu).
void pi_attn_prefill_plan(int batch, int num_qo_heads, int num_kv_heads, int max_q_len,
                          int max_kv_len, int* splits, int* max_groups) {
  prefill_plan(batch, num_qo_heads, num_kv_heads, max_q_len, max_kv_len, device_info().sms, splits, max_groups);
}

// q/out: [total_q, Hq, 128] bf16 contiguous; qo_indptr: [batch + 1] int32;
// prefix_lens: [batch] int32 (KV of request b = prefix + q_len, already in cache);
// page table dense [batch, table_stride]. ws_o/ws_ml ([total_q, Hq, max_groups, 128|2]
// f32) are required iff splits > 1.
const char* pi_attn_prefill(const void* q, const void* k_cache, const void* v_cache,
                            const int32_t* qo_indptr, const int32_t* page_table,
                            int64_t table_stride, const int32_t* prefix_lens, void* out, int batch,
                            int total_q, int num_qo_heads, int num_kv_heads, int page_size,
                            int max_q_len, double sm_scale, int splits, float* ws_o, float* ws_ml,
                            int max_groups, void* stream) {
  if (num_kv_heads <= 0 || num_qo_heads % num_kv_heads || num_qo_heads / num_kv_heads > 16)
    return fail("PI attention needs num_qo_heads = g * num_kv_heads with g <= 16");
  if (batch == 0 || total_q == 0) return nullptr;
  if (splits > 1 && (!ws_o || !ws_ml)) return fail("PI prefill split mode needs a workspace");
  auto st = (cudaStream_t)stream;
  const int group = num_qo_heads / num_kv_heads;
  const int tiles = (int)(((int64_t)max_q_len * group + kPrefillWarps * 16 - 1) / (kPrefillWarps * 16));
  set_smem(prefill_kernel, prefill_smem_bytes());
  prefill_kernel<<<dim3(tiles, num_kv_heads, batch * splits), kPrefillWarps * 32, prefill_smem_bytes(), st>>>(
      (const __nv_bfloat16*)q, (__nv_bfloat16*)out, (const __nv_bfloat16*)k_cache,
      (const __nv_bfloat16*)v_cache, qo_indptr, nullptr, page_table, table_stride, nullptr, prefix_lens,
      num_qo_heads, num_kv_heads, page_size, page_shift_of(page_size),
      (float)(sm_scale * 1.4426950408889634), splits, ws_o, (float2*)ws_ml, max_groups);
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

// Kernel ids: 1/2/4 small (4 warps, G = 1/2/4), 6..9 small narrow, 20..25 small with
// deep weight-streaming pipelines, 30..33 balanced (one CTA per SM; 30/31 M <= 16,
// 32 M <= 32, 33 M <= 64), 40..43 stream-K (M <= 8), 10..15 CUTLASS, 44..47 stream-K over
// CUTLASS tiles (S = 1 shapes), 48/49 segment-parallel CUTLASS tiles (S > 1 shapes). All
// realise the same k order; an id that cannot run a shape refuses it and is not tuned.
const std::vector<int>& small_kernels() {
  static const std::vector<int> v{1, 2, 4, 20, 21, 22, 24, 25, 30, 31, 32, 33, 40, 41, 42, 43};
  return v;
}
// Largest M a kernel id supports (balanced kernels hold all rows in one pass).
int kernel_max_m(int k) {
  return k == 30 || k == 31 ? 16 : k == 32 ? 32 : k == 33 ? 64 : k >= 40 && k <= 43 ? 8 : 1 << 30;
}

// K granularity a kernel id requires.
int kernel_bk(int k) { return k == 21 || k == 22 || k == 30 || k == 43 ? 256 : k >= 20 && k <= 42 ? 128 : 64; }
const std::vector<int>& large_kernels() {
  static const std::vector<int> v{10, 11, 12, 13, 14, 15, 44, 45, 46, 47, 48, 49};
  return v;
}

int bucket_of(int m) {
  int b = 1;
  while (b < m && b < (1 << 20)) b <<= 1;
  return b;
}

std::mutex tune_mutex;
std::map<std::tuple<int, int, int>, int> tuned;  // (N, K, M bucket) -> kernel

// Fixed choice when a bucket was never tuned (e.g. first seen inside capture),
// from whole-decode-step measurements: deep weight-streaming tiles for small M.
int heuristic_kernel(int M, int K = 0) {
  if (M <= 16) return K % 128 == 0 ? 20 : 1;
  if (M <= 32) return K % 128 == 0 ? 24 : 2;
  return M <= 64 ? 13 : M <= 1024 ? 12 : 11;
}

template <typename OutT>
cudaError_t run_gemm(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M,
                     int N, int K, cudaStream_t st) {
  auto small = [&](auto kern, int G, int W, int BK = kSmallBK, int ST = kSmallStages) {
    const size_t smem = (size_t)ST * (16 * G + 8 * W) * (BK + 8) * sizeof(__nv_bfloat16);
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
    case 20: return small(gemm_small_kernel<1, 4, OutT, 128, 6>, 1, 4, 128, 6);
    case 21: return small(gemm_small_kernel<1, 4, OutT, 256, 3>, 1, 4, 256, 3);
    case 22: return small(gemm_small_kernel<1, 2, OutT, 256, 4>, 1, 2, 256, 4);
    case 24: return small(gemm_small_kernel<2, 4, OutT, 128, 5>, 2, 4, 128, 5);
    case 25: return small(gemm_small_kernel<4, 4, OutT, 128, 3>, 4, 4, 128, 3);
    case 30: return run_balanced<OutT, 1, 256, 3>(x, ldx, w, y, ldy, M, N, K, device_info().sms, device_info().smem_optin, st);
    case 31: return run_balanced<OutT, 1, 128, 6>(x, ldx, w, y, ldy, M, N, K, device_info().sms, device_info().smem_optin, st);
    case 32: return run_balanced<OutT, 2, 128, 4>(x, ldx, w, y, ldy, M, N, K, device_info().sms, device_info().smem_optin, st);
    case 33: return run_balanced<OutT, 4, 128, 3>(x, ldx, w, y, ldy, M, N, K, device_info().sms, device_info().smem_optin, st);
    case 40: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 2, device_info().sms, st);
    case 41: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 3, device_info().sms, st);
    case 42: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 4, device_info().sms, st);
    case 43: return run_streamk<OutT, 256, 3>(x, ldx, w, y, ldy, M, N, K, 3, device_info().sms, st);
    case 44: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_streamk(x, ldx, w, y, ldy, M, N, K, device_info().sms, 1, st);
    case 45: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, device_info().sms, 2, st);
    case 46: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, device_info().sms, 1, st);
    case 47: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, device_info().sms, 1, st);
    case 48: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_segpar(x, ldx, w, y, ldy, M, N, K, st);
    case 49: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run_segpar(x, ldx, w, y, ldy, M, N, K, st);
    case 10: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 11: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 12: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 14: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 15: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    default: return CutlassGemm<OutT, 64, 64, 32, 32, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, st);
  }
}

// Time each candidate kernel for one (N, K, M bucket) and remember the fastest.
// Scratch comes from the caller (the torch allocator): x [m, K] and y [m, N] bf16,
// and an optional L2-flush buffer used for small m, where GEMMs stream weights
// from DRAM in a real forward (its weight-sized slices serve as weight copies).
// Called outside stream capture only.
const char* tune_bucket(const void* w, int N, int K, int m, void* x, void* y, void* flush,
                        size_t flush_bytes, cudaStream_t st) {
  cudaMemsetAsync(x, 0, (size_t)m * K * 2, st);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<int> cands = large_kernels();
  if (m <= 64) cands.insert(cands.begin(), small_kernels().begin(), small_kernels().end());
  float best = 1e30f;
  int best_k = heuristic_kernel(m, K);
  // Small m: time bursts of back-to-back launches over a ring of weight copies (the
  // weight itself and every weight-sized slice of the flush buffer, >= 2x L2 in all),
  // so that every launch streams from DRAM and launch gaps overlap as in a forward;
  // single cold launches misrank microsecond GEMVs. A weight larger than the flush
  // buffer is its own ring.
  const size_t w_bytes = (size_t)N * K * sizeof(__nv_bfloat16);
  const int copies = flush ? 1 + (int)(flush_bytes / w_bytes) : 0;
  const int burst = w_bytes >= ((size_t)64 << 20) ? 8 : 32;
  auto copy = [&](int i) {
    const int c = i % copies;
    return c == 0 ? w : (const void*)((const char*)flush + (size_t)(c - 1) * w_bytes);
  };
  for (int k : cands) {
    if (K % kernel_bk(k) || m > kernel_max_m(k)) continue;
    if (run_gemm(k, x, K, w, (__nv_bfloat16*)y, N, m, N, K, st) != cudaSuccess) continue;  // warm-up
    std::vector<float> times;
    for (int r = 0; r < 5; ++r) {
      cudaEventRecord(e0, st);
      if (copies) {
        for (int i = 0; i < burst; ++i) run_gemm(k, x, K, copy(r * burst + i), (__nv_bfloat16*)y, N, m, N, K, st);
      } else {
        run_gemm(k, x, K, w, (__nv_bfloat16*)y, N, m, N, K, st);
      }
      cudaEventRecord(e1, st);
      cudaEventSynchronize(e1);
      float ms = 0.f;
      cudaEventElapsedTime(&ms, e0, e1);
      times.push_back(copies ? ms / burst : ms);
    }
    std::sort(times.begin(), times.end());
    if (times[2] < best) {
      best = times[2];
      best_k = k;
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
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
// Uses the tuned tile for (N, K, M bucket) when present, a fixed heuristic otherwise;
// every choice gives identical bits.
const char* pi_linear(const void* x, int64_t ldx, const void* w, void* y, int M, int N, int K,
                      int out_fp32, void* stream) {
  if (K % 64 || N % 32) return fail("PI linear needs K % 64 == 0 and N % 32 == 0");
  if (M == 0) return nullptr;
  auto st = (cudaStream_t)stream;
  int k = -1;
  {
    std::lock_guard<std::mutex> lock(tune_mutex);
    auto it = tuned.find({N, K, bucket_of(M)});
    if (it != tuned.end()) k = it->second;
  }
  if (k < 0 || (M > 64 && k < 10) || M > kernel_max_m(k) || K % kernel_bk(k)) k = heuristic_kernel(M, K);
  const cudaError_t err = out_fp32 ? run_gemm(k, x, ldx, w, (float*)y, N, M, N, K, st)
                                   : run_gemm(k, x, ldx, w, (__nv_bfloat16*)y, N, M, N, K, st);
  if (err != cudaSuccess) return fail(std::string("PI linear: ") + cudaGetErrorString(err));
  return nullptr;
}

// M bucket (power of two) used for tuning, and the kernel id tuned for it (0 = not tuned).
int pi_linear_bucket(int M) { return bucket_of(M); }
int pi_linear_tuned_kernel(int N, int K, int bucket) {
  std::lock_guard<std::mutex> lock(tune_mutex);
  auto it = tuned.find({N, K, bucket});
  return it == tuned.end() ? 0 : it->second;
}
// Whether `stream` is capturing a CUDA graph (tuning must not run then).
int pi_stream_capturing(void* stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing((cudaStream_t)stream, &status);
  return status != cudaStreamCaptureStatusNone;
}
// L2 size in bytes of the current device (flush buffer sizing).
int64_t pi_l2_bytes() {
  int device = 0, l2 = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device);
  return l2;
}
const char* pi_linear_tune(const void* w, int N, int K, int bucket, void* x, void* y, void* flush,
                           int64_t flush_bytes, void* stream) {
  if (K % 64 || N % 32) return fail("PI linear needs K % 64 == 0 and N % 32 == 0");
  return tune_bucket(w, N, K, bucket, x, y, flush, (size_t)flush_bytes, (cudaStream_t)stream);
}

}  // extern "C"
