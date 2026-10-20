// M-invariant GEMM: Y[M,N] = X[M,K] * W[N,K]^T (nn.Linear), bf16 in, fp32 accumulate.
//
// Canonical per-element operation sequence: acc = 0; for k16 = 0 .. K/16-1 in
// ascending order: acc = mma.m16n8k16(x[m, k16], w[n, k16], acc); y = round(acc).
// No split-K, no reordering of k. Every kernel below realises exactly this
// sequence, so a row of Y is a pure function of that row of X and W, whatever M,
// the tile shape, or the other rows in the batch.
#pragma once
#include "pi_attn.cuh"

namespace pi_gemm {
using namespace pi_attn;

// ---------------------------------------------------------------- small M ----
// CTA = 4 warps; warp w owns output columns [n0 + 8w, n0 + 8w + 8). All warps
// share the X tile (16*G rows). K streams through a cp.async ring of STAGES
// stages of BK columns. One n8 tile per warp keeps the CTA count high for the
// narrow projections of decode (N = 1024 gives 32 CTAs x 4 warps).
constexpr int kSmallBK = 64;
constexpr int kSmallStages = 4;
constexpr int kSmallStride = kSmallBK + 8;  // bf16 elements per smem row (conflict-free ldmatrix)

template <int G, int W>
constexpr int small_smem_elems() {
  return kSmallStages * (16 * G + 8 * W) * kSmallStride;
}

// W warps per CTA (BN = 8 W): fewer warps per CTA means more CTAs for narrow N.
template <int G, int W, typename OutT>
__global__ void __launch_bounds__(W * 32)
    gemm_small_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                      const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y, int64_t ldy, int M,
                      int N, int K) {
  constexpr int kRows = 16 * G;
  constexpr int kSmallBN = 8 * W;
  constexpr int kStageElems = (kRows + kSmallBN) * kSmallStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * kSmallBN, m0 = blockIdx.y * kRows;
  const int ktiles = K / kSmallBK;

  auto issue = [&](int kt) {
    __nv_bfloat16* st = smem + (kt % kSmallStages) * kStageElems;
    constexpr int kVec = kSmallBK / 8;
    for (int i = threadIdx.x; i < (kRows + kSmallBN) * kVec; i += blockDim.x) {
      const int r = i / kVec, v = i % kVec;
      const __nv_bfloat16* src = w;
      int bytes = 0;
      if (r < kRows) {
        if (m0 + r < M) { src = x + (int64_t)(m0 + r) * ldx + kt * kSmallBK + v * 8; bytes = 16; }
      } else {
        src = w + (int64_t)(n0 + r - kRows) * K + kt * kSmallBK + v * 8;
        bytes = 16;
      }
      cp_async16(st + r * kSmallStride + v * 8, src, bytes);
    }
  };
#pragma unroll
  for (int s = 0; s < kSmallStages - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }

  float acc[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kSmallStages - 2>();
    __syncthreads();
    if (kt + kSmallStages - 1 < ktiles) issue(kt + kSmallStages - 1);
    cp_async_commit();
    const __nv_bfloat16* st = smem + (kt % kSmallStages) * kStageElems;
    const __nv_bfloat16* sx = st;
    const __nv_bfloat16* sw = st + kRows * kSmallStride + warp * 8 * kSmallStride;
#pragma unroll
    for (int ks = 0; ks < kSmallBK / 16; ks += 2) {
      // B fragments for two k16 steps of this warp's n8 tile.
      uint32_t b[4];
      ldmatrix_x4(b, sw + r8 * kSmallStride + ks * 16 + mat * 8);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + (g * 16 + (mat & 1) * 8 + r8) * kSmallStride + (ks + kk) * 16 + (mat >> 1) * 8);
          mma_bf16(acc[g], a, b[2 * kk], b[2 * kk + 1]);
        }
      }
    }
  }
  cp_async_wait<0>();
  const int gr = lane >> 2, t = lane & 3;
  const int n = n0 + warp * 8 + t * 2;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int m = m0 + g * 16 + gr + h * 8;
      if (m >= M) continue;
      if constexpr (sizeof(OutT) == 2) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[g][2 * h], acc[g][2 * h + 1]);
        *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = v;
      } else {
        *reinterpret_cast<float2*>(y + (int64_t)m * ldy + n) = make_float2(acc[g][2 * h], acc[g][2 * h + 1]);
      }
    }
  }
}

// ---------------------------------------------------------------- large M ----
// CTA tile 128 x 128, 8 warps (2 x 4), warp tile 64 x 32 (4 m16 x 4 n8), BK = 32,
// 4-stage cp.async ring. Per element the k16 steps run in ascending order.
constexpr int kLargeBM = 128, kLargeBN = 128, kLargeBK = 32, kLargeStages = 4;
constexpr int kLargeStride = kLargeBK + 8;
constexpr int kLargeStageElems = (kLargeBM + kLargeBN) * kLargeStride;
constexpr int kLargeSmemElems = kLargeStages * kLargeStageElems;

template <typename OutT>
__global__ void __launch_bounds__(256)
    gemm_large_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                      const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y, int64_t ldy, int M,
                      int N, int K) {
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wm = warp >> 2, wn = warp & 3;  // 2 x 4 warps
  const int m0 = blockIdx.y * kLargeBM, n0 = blockIdx.x * kLargeBN;
  const int ktiles = K / kLargeBK;

  auto issue = [&](int kt) {
    __nv_bfloat16* st = smem + (kt % kLargeStages) * kLargeStageElems;
    constexpr int kVec = kLargeBK / 8;
    for (int i = threadIdx.x; i < (kLargeBM + kLargeBN) * kVec; i += blockDim.x) {
      const int r = i / kVec, v = i % kVec;
      const __nv_bfloat16* src = w;
      int bytes = 0;
      if (r < kLargeBM) {
        if (m0 + r < M) { src = x + (int64_t)(m0 + r) * ldx + kt * kLargeBK + v * 8; bytes = 16; }
      } else if (n0 + r - kLargeBM < N) {
        src = w + (int64_t)(n0 + r - kLargeBM) * K + kt * kLargeBK + v * 8;
        bytes = 16;
      }
      cp_async16(st + r * kLargeStride + v * 8, src, bytes);
    }
  };
#pragma unroll
  for (int s = 0; s < kLargeStages - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kLargeStages - 2>();
    __syncthreads();
    if (kt + kLargeStages - 1 < ktiles) issue(kt + kLargeStages - 1);
    cp_async_commit();
    const __nv_bfloat16* sx = smem + (kt % kLargeStages) * kLargeStageElems;
    const __nv_bfloat16* sw = sx + kLargeBM * kLargeStride;
#pragma unroll
    for (int ks = 0; ks < kLargeBK / 16; ++ks) {
      uint32_t a[4][4], b[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        ldmatrix_x4(a[i], sx + (wm * 64 + i * 16 + (mat & 1) * 8 + r8) * kLargeStride + ks * 16 + (mat >> 1) * 8);
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        uint32_t bb[4];
        ldmatrix_x4(bb, sw + (wn * 32 + (j + (mat >> 1)) * 8 + r8) * kLargeStride + ks * 16 + (mat & 1) * 8);
        b[j][0] = bb[0]; b[j][1] = bb[1]; b[j + 1][0] = bb[2]; b[j + 1][1] = bb[3];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a[i], b[j][0], b[j][1]);
    }
  }
  cp_async_wait<0>();
  const int gr = lane >> 2, t = lane & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = n0 + wn * 32 + j * 8 + t * 2;
      if (n >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + wm * 64 + i * 16 + gr + h * 8;
        if (m >= M) continue;
        if constexpr (sizeof(OutT) == 2) {
          *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) =
              __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        } else {
          *reinterpret_cast<float2*>(y + (int64_t)m * ldy + n) =
              make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        }
      }
    }
  }
}

}  // namespace pi_gemm
