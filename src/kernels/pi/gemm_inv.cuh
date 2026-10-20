// M-invariant GEMM: Y[M,N] = X[M,K] * W[N,K]^T (nn.Linear), bf16 in, fp32 accumulate.
//
// Canonical per-element operation sequence: acc = 0; for k16 = 0 .. K/16-1 in
// ascending order: acc = mma.m16n8k16(x[m, k16], w[n, k16], acc); y = round(acc).
// No split-K, no reordering of k. The small-M kernels below and the CUTLASS
// tiles in gemm_cutlass.cuh realise exactly this sequence, so a row of Y is a
// pure function of that row of X and W, whatever M, the tile shape, or the other
// rows in the batch (checked bitwise by kernels/test_gemm.py).
#pragma once
#include "pi_attn.cuh"

namespace pi_gemm {
using namespace pi_attn;

// ---------------------------------------------------------------- small M ----
// CTA = W warps; warp w owns output columns [n0 + 8w, n0 + 8w + 8). All warps
// share the X tile (16*G rows). K streams through a cp.async ring of STAGES
// stages of BK columns. Decode GEMMs are weight-streaming: what matters is the
// bytes in flight per CTA ((STAGES-1) * 8W * BK * 2), so narrow-N projections
// use deep stages. BK and STAGES change scheduling only, never the k order.
constexpr int kSmallBK = 64;
constexpr int kSmallStages = 4;
constexpr int kSmallStride = kSmallBK + 8;  // bf16 elements per smem row (conflict-free ldmatrix)

template <int G, int W, int BK = kSmallBK, int STAGES = kSmallStages>
constexpr int small_smem_elems() {
  return STAGES * (16 * G + 8 * W) * (BK + 8);
}

// W warps per CTA (BN = 8 W): fewer warps per CTA means more CTAs for narrow N.
template <int G, int W, typename OutT, int BK = kSmallBK, int STAGES = kSmallStages>
__global__ void __launch_bounds__(W * 32)
    gemm_small_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                      const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y, int64_t ldy, int M,
                      int N, int K) {
  constexpr int kRows = 16 * G;
  constexpr int kSmallBN = 8 * W;
  constexpr int kStride = BK + 8;
  constexpr int kStageElems = (kRows + kSmallBN) * kStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * kSmallBN, m0 = blockIdx.y * kRows;
  const int ktiles = K / BK;

  auto issue = [&](int kt) {
    __nv_bfloat16* st = smem + (kt % STAGES) * kStageElems;
    constexpr int kVec = BK / 8;
    for (int i = threadIdx.x; i < (kRows + kSmallBN) * kVec; i += blockDim.x) {
      const int r = i / kVec, v = i % kVec;
      const __nv_bfloat16* src = w;
      int bytes = 0;
      if (r < kRows) {
        if (m0 + r < M) { src = x + (int64_t)(m0 + r) * ldx + kt * BK + v * 8; bytes = 16; }
      } else {
        src = w + (int64_t)(n0 + r - kRows) * K + kt * BK + v * 8;
        bytes = 16;
      }
      cp_async16(st + r * kStride + v * 8, src, bytes);
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }

  float acc[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
    cp_async_commit();
    const __nv_bfloat16* st = smem + (kt % STAGES) * kStageElems;
    const __nv_bfloat16* sx = st;
    const __nv_bfloat16* sw = st + kRows * kStride + warp * 8 * kStride;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks += 2) {
      // B fragments for two k16 steps of this warp's n8 tile.
      uint32_t b[4];
      ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + (g * 16 + (mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
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

}  // namespace pi_gemm
