// GEMM kernel ids -> kernels. The engine (pi_launch.cu) and the PyTorch test/bench
// bindings (kernels/gemm_torch.cu in the PI-Attn repo) both dispatch through run_gemm, so
// the tests exercise the exact id -> configuration table the tuner chooses from.
//
// Ids: 1/2/4 small (4 warps, G = 1/2/4 row groups of 16), 7 small with 2 warps, 20..25
// small with deep weight-streaming pipelines, 30..33 balanced (one CTA per SM; 30/31
// M <= 16, 32 M <= 32, 33 M <= 64), 40..43, 51 and 53 stream-K (M <= 8; 51 and 53
// stream 512- and 1024-column stages, for long K), 60..62 stream-K over pairs of n8
// tiles (M <= 8; the pair shares each X fragment, for M = 8 of long-K projections), 63 = 60
// with 16 X rows (M <= 16: narrow weights, e.g. tensor-parallel column shards, past M = 8),
// 10..15
// CUTLASS tiles, 16 = 10 with CTAs swizzled 8 tiles wide (L2 reuse at M >= 8192), 17 / 18 / 19 =
// 10 / 16 / 11 with the segment total in L2 slots (S > 1 only), 44..47, 50 and 64..66 stream-K
// over CUTLASS tiles (64: 32x32 tiles, for narrow weights at M = 17..64; 65 / 66: 128x128 /
// 128x256, for wide weights at M = 256..1024, where those tiles alone leave SMs idle), 48/49
// segment-parallel CUTLASS tiles (S > 1 shapes), 54/55 segment-split (M <= 16, S > 1
// shapes: one CTA per n8 tile, one warp per segment). All realise the same k order; an id that
// cannot run a shape returns an error before launching and is then not tuned. S (segs) is
// the weight's canonical segment count, gemm_segments(N, K, t_max) of the whole weight:
// a launch over a column shard of it passes the whole weight's S.
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
  static const std::vector<int> v{1,  2,  4,  7,  20, 21, 22, 24, 25, 30, 31, 32, 33, 40, 41, 42, 43, 51,
                                  53, 54, 55, 60, 61, 62, 63};
  return v;
}
// Candidates for every M bucket.
inline const std::vector<int>& large_kernels() {
  static const std::vector<int> v{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 44, 45, 46, 47, 48, 49, 50, 64, 65, 66};
  return v;
}
inline bool is_small_kernel(int k) { return k < 10 || (k >= 20 && k <= 43) || k == 51 || (k >= 53 && k <= 55) || (k >= 60 && k <= 63); }

// Largest M a kernel id supports (balanced and stream-K kernels hold all rows in one pass).
inline int kernel_max_m(int k) {
  switch (k) {
    case 40: case 41: case 42: case 43: case 51: case 53: case 60: case 61: case 62: return 8;
    case 30: case 31: case 54: case 55: case 63: return 16;
    case 32: return 32;
    case 33: return 64;
    default: return 1 << 30;
  }
}
// K granularity (within one segment) a kernel id requires.
inline int kernel_bk(int k) {
  switch (k) {
    case 53: return 1024;
    case 51: case 61: return 512;
    case 21: case 22: case 30: case 43: case 60: case 62: case 63: return 256;
    case 20: case 24: case 25: case 31: case 32: case 33: case 40: case 41: case 42: case 54: return 128;
    default: return 64;
  }
}

// Function attributes are per-kernel, per-device host state: set once per device (outside capture).
template <typename Kernel>
void set_smem(Kernel kernel, size_t smem) {
  static std::mutex mutex;
  static std::map<std::pair<int, const void*>, size_t> done;
  int device = 0;
  cudaGetDevice(&device);
  std::lock_guard<std::mutex> lock(mutex);
  auto& v = done[{device, (const void*)kernel}];
  if (v < smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    v = smem;
  }
}

// Segment-split launch (ids 54/55): S > 1 shapes only, where it differs from a small kernel.
template <typename OutT, int BK, int STAGES>
cudaError_t run_segsplit(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                         int segs, int max_smem, cudaStream_t st) {
  const size_t smem = segsplit_smem_bytes<BK, STAGES>(segs);
  if (segs < 2 || smem > (size_t)max_smem) return cudaErrorInvalidValue;
  set_smem(gemm_segsplit_kernel<OutT, BK, STAGES>, smem);
  gemm_segsplit_kernel<OutT, BK, STAGES><<<N / 8, segs * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K);
  return cudaGetLastError();
}

struct GemmDevice {
  int sms = 0, smem_optin = 0;
};

// y[M, N] = x[M, K] w[N, K]^T with kernel id k and S = segs. An unknown id, or a shape the
// id cannot run, returns cudaErrorInvalidValue without launching.
template <typename OutT>
cudaError_t run_gemm(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M,
                     int N, int K, int segs, GemmDevice dev, cudaStream_t st) {
  if (segs < 1 || segs > kSegMax || K % segs || K / segs < kernel_bk(k) || (K / segs) % kernel_bk(k) ||
      M > kernel_max_m(k))
    return cudaErrorInvalidValue;
  auto small = [&](auto kern, int G, int W, int BK = kSmallBK, int ST = kSmallStages) {
    const size_t smem = (size_t)ST * (16 * G + 8 * W) * (BK + 8) * sizeof(__nv_bfloat16);
    set_smem(kern, smem);
    kern<<<dim3(N / (8 * W), (M + 16 * G - 1) / (16 * G)), W * 32, smem, st>>>(
        (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K, segs);
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
    case 30: return run_balanced<OutT, 1, 256, 3>(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, dev.smem_optin, st);
    case 31: return run_balanced<OutT, 1, 128, 6>(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, dev.smem_optin, st);
    case 32: return run_balanced<OutT, 2, 128, 4>(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, dev.smem_optin, st);
    case 33: return run_balanced<OutT, 4, 128, 3>(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, dev.smem_optin, st);
    case 40: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 41: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, segs, 3, dev.sms, dev.smem_optin, st);
    case 42: return run_streamk<OutT, 128, 4>(x, ldx, w, y, ldy, M, N, K, segs, 4, dev.sms, dev.smem_optin, st);
    case 43: return run_streamk<OutT, 256, 3>(x, ldx, w, y, ldy, M, N, K, segs, 3, dev.sms, dev.smem_optin, st);
    case 51: return run_streamk<OutT, 512, 3>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 53: return run_streamk<OutT, 1024, 2>(x, ldx, w, y, ldy, M, N, K, segs, 1, dev.sms, dev.smem_optin, st);
    case 60: return run_streamk<OutT, 256, 3, 2>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 61: return run_streamk<OutT, 512, 3, 2>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 62: return run_streamk<OutT, 256, 4, 2>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 63: return run_streamk<OutT, 256, 3, 2, 16>(x, ldx, w, y, ldy, M, N, K, segs, 2, dev.sms, dev.smem_optin, st);
    case 54: return run_segsplit<OutT, 128, 3>(x, ldx, w, y, ldy, M, N, K, segs, dev.smem_optin, st);
    case 55: return run_segsplit<OutT, 64, 6>(x, ldx, w, y, ldy, M, N, K, segs, dev.smem_optin, st);
    case 10: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 11: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 12: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 13: return CutlassGemm<OutT, 64, 64, 32, 32, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 14: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 15: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 16: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3, 8>::run(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 17: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run_segmented_slot(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, st);
    case 18: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3, 8>::run_segmented_slot(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, st);
    case 19: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run_segmented_slot(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, st);
    case 44: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    case 45: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 2, st);
    case 46: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    case 47: return CutlassGemm<OutT, 64, 128, 32, 32, 64, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    case 48: return CutlassGemm<OutT, 32, 64, 64, 16, 32, 5>::run_segpar(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 49: return CutlassGemm<OutT, 64, 64, 64, 32, 32, 3>::run_segpar(x, ldx, w, y, ldy, M, N, K, segs, st);
    case 50: return CutlassGemm<OutT, 128, 64, 64, 64, 32, 3>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    case 64: return CutlassGemm<OutT, 32, 32, 64, 16, 32, 5>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 2, st);
    case 65: return CutlassGemm<OutT, 128, 128, 32, 64, 64, 4>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    case 66: return CutlassGemm<OutT, 128, 256, 32, 64, 64, 3>::run_streamk(x, ldx, w, y, ldy, M, N, K, segs, dev.sms, 1, st);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace pi_gemm
