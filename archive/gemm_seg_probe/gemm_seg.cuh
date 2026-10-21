// Prototype: K-segmented balanced GEMM for M <= 16 G. Canonical sequence per
// element: for each segment s of Ks = K / S columns: p_s = 0 + k16 MMA chain over
// the segment (ascending); y = round(((p_0 + p_1) + p_2) ...). Work item = (n8
// unit, segment), segment-major; CTA c owns items [c I / grid, (c + 1) I / grid),
// processed `warps` at a time within one segment so the warps share X stages.
// Partials go to ws [S][M][N] f32; the last warp to finish a unit (per-unit
// arrival counter, reset by it) folds the unit's S partials in ascending order.
#pragma once
#include "gemm_inv.cuh"

namespace pi_gemm {

template <int G, int BK, int STAGES>
__global__ void gemm_seg_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx, const __nv_bfloat16* __restrict__ w,
                                __nv_bfloat16* __restrict__ y, int64_t ldy, float* __restrict__ ws,
                                int* __restrict__ counters, int M, int N, int K, int S) {
  constexpr int kStride = BK + 8, kRows = 16 * G;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int stage_elems = (kRows + 8 * warps) * kStride;
  const int units = N / 8, items = units * S, Ks = K / S;
  const int i0 = (int)((int64_t)blockIdx.x * items / gridDim.x);
  const int i1 = (int)((int64_t)(blockIdx.x + 1) * items / gridDim.x);
  const int ktiles = Ks / BK;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;
  for (int base = i0; base < i1;) {
    const int seg = base / units, ubase = base - seg * units;
    const int cnt = min(min(warps, i1 - base), units - ubase);  // items of this pass, one segment
    const int nrows = cnt * 8;
    const int k0 = seg * Ks;
    auto issue = [&](int kt) {
      __nv_bfloat16* st = smem + (kt % STAGES) * stage_elems;
      constexpr int kVec = BK / 8;
      for (int i = threadIdx.x; i < (kRows + nrows) * kVec; i += blockDim.x) {
        const int r = i / kVec, v = i % kVec;
        const int kc = k0 + kt * BK + v * 8;
        if (r >= kRows) cp_async16(st + r * kStride + v * 8, w + ((int64_t)ubase * 8 + r - kRows) * K + kc, 16);
        else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
        else cp_async16(st + r * kStride + v * 8, x, 0);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (s < ktiles) issue(s);
      cp_async_commit();
    }
    float acc[G][4];
#pragma unroll
    for (int gi = 0; gi < G; ++gi) acc[gi][0] = acc[gi][1] = acc[gi][2] = acc[gi][3] = 0.f;
    const bool active = warp < cnt;
    for (int kt = 0; kt < ktiles; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();
      if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
      cp_async_commit();
      if (!active) continue;
      const __nv_bfloat16* sx = smem + (kt % STAGES) * stage_elems;
      const __nv_bfloat16* sw = sx + (kRows + warp * 8) * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int gi = 0; gi < G; ++gi) {
            uint32_t a[4];
            ldmatrix_x4(a, sx + (gi * 16 + (mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
            mma_bf16(acc[gi], a, b[2 * kk], b[2 * kk + 1]);
          }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    base += cnt;
    if (!active) continue;
    const int unit = ubase + warp, n = unit * 8 + t * 2;
    if (S == 1) {
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = gi * 16 + g + h * 8;
          if (m < M) *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(acc[gi][2 * h], acc[gi][2 * h + 1]);
        }
      continue;
    }
#pragma unroll
    for (int gi = 0; gi < G; ++gi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = gi * 16 + g + h * 8;
        if (m < M) *reinterpret_cast<float2*>(ws + ((int64_t)seg * M + m) * N + n) = make_float2(acc[gi][2 * h], acc[gi][2 * h + 1]);
      }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) {
      last = atomicAdd(counters + unit, 1) == S - 1;
      if (last) counters[unit] = 0;
    }
    if (!__shfl_sync(0xffffffffu, last, 0)) continue;
    __threadfence();
#pragma unroll
    for (int gi = 0; gi < G; ++gi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = gi * 16 + g + h * 8;
        if (m >= M) continue;
        float2 tot = __ldcg(reinterpret_cast<const float2*>(ws + (int64_t)m * N + n));
        for (int s = 1; s < S; ++s) {
          const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((int64_t)s * M + m) * N + n));
          tot.x = __fadd_rn(tot.x, p.x);
          tot.y = __fadd_rn(tot.y, p.y);
        }
        *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(tot.x, tot.y);
      }
  }
}

template <int G, int BK, int STAGES>
cudaError_t run_seg(const void* x, int64_t ldx, const void* w, __nv_bfloat16* y, int64_t ldy, float* ws, int* counters,
                    int M, int N, int K, int S, int grid, int warps, cudaStream_t st) {
  if (M > 16 * G || K % (S * BK)) return cudaErrorInvalidValue;
  const size_t smem = balanced_smem_bytes<G, BK, STAGES>(warps);
  cudaFuncSetAttribute(gemm_seg_kernel<G, BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_seg_kernel<G, BK, STAGES><<<grid, warps * 32, smem, st>>>((const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y,
                                                                 ldy, ws, counters, M, N, K, S);
  return cudaGetLastError();
}

}  // namespace pi_gemm
