// GEMM kernel ids -> kernels. The engine (pi_launch.cu) and the PyTorch test/bench
// bindings (kernels/gemm_torch.cu in the PI-Attn repo) both dispatch through run_gemm, so
// the tests exercise the exact id -> configuration table the tuner chooses from.
//
// Ids: 1/2/4 small (4 warps, G = 1/2/4 row groups of 16), 7 small with 2 warps, 20..25
// small with deep weight-streaming pipelines, 30..33 balanced (one CTA per SM; 30/31
// M <= 16, 32 M <= 32, 33 M <= 64), 40..43, 51 and 53 stream-K (M <= 8, S = 1
// shapes; 51 and 53 stream 512- and 1024-column stages, for long K), 10..15
// CUTLASS tiles, 44..47 and 50 stream-K over CUTLASS tiles (S = 1 shapes), 48/49
// segment-parallel CUTLASS tiles (S > 1 shapes), 54/55 segment-split (M <= 16, S > 1
// shapes: one CTA per n8 tile, one warp per segment). All realise the same k order; an id that
// cannot run a shape returns an error before launching and is then not tuned.
#pragma once

#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <vector>

#include "gemm_cutlass.cuh"
#include "gemm_inv.cuh"

namespace pi_gemm {

// Candidates for M buckets <= 64 (tried in addition to large_kernels()).
inline const std::vector<int>& small_kernels() {
  static const std::vector<int> v{1, 2, 4, 7, 20, 21, 22, 24, 25, 30, 31, 32, 33, 40, 41, 42, 43, 51, 53, 54, 55};
  return v;
}
// Candidates for every M bucket.
inline const std::vector<int>& large_kernels() {
  static const std::vector<int> v{10, 11, 12, 13, 14, 15, 44, 45, 46, 47, 48, 49, 50};
  return v;
}
inline bool is_small_kernel(int k) { return k < 10 || (k >= 20 && k <= 43) || k == 51 || (k >= 53 && k <= 55); }

// Largest M a kernel id supports (balanced and stream-K kernels hold all rows in one pass).
inline int kernel_max_m(int k) {
  return k == 30 || k == 31 || k == 54 || k == 55 ? 16 : k == 32 ? 32 : k == 33 ? 64 : (k >= 40 && k <= 43) || k == 51 || k == 53 ? 8 : 1 << 30;
}
// K granularity a kernel id requires.
inline int kernel_bk(int k) {
  return k == 51 ? 512 : k == 53 ? 1024 : k == 21 || k == 22 || k == 30 || k == 43 ? 256 : k >= 20 && k <= 42 || k == 54 ? 128 : 64;
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

// Segment-split launch (ids 54/55): S > 1 shapes only, where it differs from a small kernel.
template <typename OutT, int BK, int STAGES>
cudaError_t run_segsplit(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                         cudaStream_t st) {
  const int segs = gemm_segments(N, K);
  if (segs < 2) return cudaErrorInvalidValue;
  const size_t smem = segsplit_smem_bytes<BK, STAGES>(segs);
  set_smem(gemm_segsplit_kernel<OutT, BK, STAGES>, smem);
  gemm_segsplit_kernel<OutT, BK, STAGES><<<N / 8, segs * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K);
  return cudaGetLastError();
}

struct GemmDevice {
  int sms = 0, smem_optin = 0;
};

// y[M, N] = x[M, K] w[N, K]^T with kernel id k. An unknown id, or a shape the id cannot
// run, returns cudaErrorInvalidValue without launching.
template <typename OutT>
cudaError_t run_gemm(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M,
                     int N, int K, GemmDevice dev, cudaStream_t st) {
  if (K % kernel_bk(k) || M > kernel_max_m(k)) return cudaErrorInvalidValue;
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
    case 7: return small(gemm_small_kernel<1, 2, OutT>, 1, 2);
    case 20: return small(gemm_small_kernel<1, 4, OutT, 128, 6>, 1, 4, 128, 6);
    case 21: return small(gemm_small_kernel<1, 4, OutT, 256, 3>, 1, 4, 256, 3);
    case 22: return small(gemm_small_kernel<1, 2, OutT, 256, 4>, 1, 2, 256, 4);
    case 24: return small(gemm_small_kernel<2, 4, OutT, 128, 5>, 2, 4, 128, 5);
    case 25: return small(gemm_small_kernel<4, 4, OutT, 128, 3>, 4, 4, 128, 3);
    case 30: return run_balanced<OutT, 1, 256, 3>(x, ldx, w, y, ldy, M, N, K, dev.sms, dev.smem_optin, st);
    case 31: return run_balanced<OutT, 1, 128, 6>(x, ldx, w, y, ldy, M, N, K, dev.sms, dev.smem_optin, st);
    case 32: return run_balanced<OutT, 2, 128, 4>(x, ldx, w, y, ldy, M, N, K, dev.sms, dev.smem_optin, st);
    case 33: return run_balanced<OutT, 4, 128, 3>(x, ldx, w, y, ldy, M, N, K, dev.sms, dev.smem_optin, st);
    case 40: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 2, dev.sms, st);
    case 41: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 3, dev.sms, st);
    case 42: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, 4, dev.sms, st);
    case 43: return run_streamk<OutT, 256, 3>(x, ldx, w, y, ldy, M, N, K, 3, dev.sms, st);
    case 51: return run_streamk<OutT, 512, 3>(x, ldx, w, y, ldy, M, N, K, 2, dev.sms, st);
    case 53: return run_streamk<OutT, 1024, 2>(x, ldx, w, y, ldy, M, N, K, 1, dev.sms, st);
    case 54: return run_segsplit<OutT, 128, 3>(x, ldx, w, y, ldy, M, N, K, st);
    case 55: return run_segsplit<OutT, 64, 6>(x, ldx, w, y, ldy, M, N, K, st);
    case 10: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 11: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 12: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 13: return CutlassGemm<OutT, 64, 64, 32, 32, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 14: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 15: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run(x, ldx, w, y, ldy, M, N, K, st);
    case 44: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_streamk(x, ldx, w, y, ldy, M, N, K, dev.sms, 1, st);
    case 45: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, dev.sms, 2, st);
    case 46: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, dev.sms, 1, st);
    case 47: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, dev.sms, 1, st);
    case 48: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_segpar(x, ldx, w, y, ldy, M, N, K, st);
    case 49: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 3>::run_segpar(x, ldx, w, y, ldy, M, N, K, st);
    case 50: return CutlassGemm<OutT, 128, 64, 64, 64, 32, 3>::run_streamk(x, ldx, w, y, ldy, M, N, K, dev.sms, 1, st);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace pi_gemm
