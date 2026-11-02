// M-invariant GEMM: Y[M,N] = X[M,K] * W[N,K]^T (nn.Linear), bf16 in, fp32 accumulate.
//
// Canonical per-element operation sequence. K is cut into S = gemm_segments(N, K, t_max)
// equal segments, a function of the model's weight shape only. With S = 1: acc = 0; for
// k16 = 0 .. K/16-1 in ascending order: acc = mma.m16n8k16(x[m, k16], w[n, k16],
// acc); y = round(acc). With S > 1 each segment runs that chain over its k16 steps
// into p_s, and y = round((((+0 + p_0) + p_1) + ...) + p_{S-1}). Segments let
// narrow projections (few n8 output tiles over a long K) spread decode weight
// streaming over every SM, the one thing split-K is for, at a cost fixed per
// shape. The small-M kernels below and the CUTLASS tiles in gemm_cutlass.cuh
// realise exactly this sequence, so a row of Y is a pure function of that row of
// X and W, whatever M, the tile shape, or the other rows in the batch (checked
// bitwise by kernels/test_gemm.py).
//
// Every launch takes S as an argument rather than deriving it from its own N, so a
// launch over any subset of the weight's rows (a tensor-parallel column shard) computes
// the same chain for each output element as the whole weight does.
#pragma once
#include "pi_attn.cuh"

namespace pi_gemm {
using namespace pi_attn;

// K segments of a [N, K] weight served at tensor-parallel degrees up to t_max. Each
// output element is one dependent MMA chain per segment, about 28 ns per k16 step on
// A800, so a GEMM whose n8 tiles cannot occupy every SM (decode of a narrow weight, or of
// an N / t_max column shard of a wide one) is bounded by the chain length, not by weight
// bandwidth. Segments shorten the chain: S doubles while the narrowest shard is at most
// 1024 / S columns wide, up to 8, keeping segments at least 1024 long and a multiple of
// 256 (every kernel's K tile divides them). With t_max = 1 every Qwen3 / Qwen3.5 shape
// keeps the S of the engine before t_max existed: two for N <= 1024 with K >= 2048 and
// K % 512 == 0 (o / down of small models), else one; four and eight arise only for
// narrower shards (N / t_max <= 512).
constexpr int kSegMax = 8;
__host__ __device__ constexpr int gemm_segments(int N, int K, int t_max = 1) {
  int s = 1;
  while (s < kSegMax && N / t_max * s <= 1024 && K % (512 * s) == 0 && K / (2 * s) >= 1024) s *= 2;
  return s;
}

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
                      int N, int K, int segs) {
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

  float acc[G][4], tot[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[g][i] = tot[g][i] = 0.f;
  const int seg_tiles = ktiles / segs;
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
    if (segs > 1 && (kt + 1) % seg_tiles == 0) {  // segment done: fold, restart from 0
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          tot[g][i] = add(tot[g][i], acc[g][i]);
          acc[g][i] = 0.f;
        }
    }
  }
  cp_async_wait<0>();
  if (segs > 1) {
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[g][i] = tot[g][i];
  }
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

// ------------------------------------------------------------ balanced M <= 16 G ----
// GEMMs with few output tiles (decode, short prefills, narrow N with long K) are
// limited by how evenly the SMs share the weight stream. Work item = one n8
// output tile over one K segment and all M <= 16 G rows; items are ordered
// segment-major. CTA c of the grid (one per SM) owns the contiguous items
// [c I / grid, (c + 1) I / grid) and processes them `warps` at a time within one
// segment, one item per warp, all warps sharing each X stage as gemm_small_kernel
// does; the per-segment sequence is gemm_small_kernel's. With S > 1 a warp writes
// its partial to seg_ws and counts its arrival on the n8 tile; the last to arrive
// (it resets the counter) folds the S partials in ascending order. seg_ws and the
// counters are per device and per translation unit: launches that use them must
// not run concurrently (the engine runs its forward on one stream). They hold S * M * N
// partials and N / 8 counters; launches that need more refuse the shape.
constexpr int64_t kSegWs = 1 << 20;
constexpr int kSegMaxUnits = 2048;
static __device__ float g_seg_ws[kSegWs];
static __device__ int g_seg_counters[kSegMaxUnits];

inline bool seg_ws_fits(int segs, int M, int N) {
  return segs == 1 || ((int64_t)segs * M * N <= kSegWs && N / 8 <= kSegMaxUnits);
}

template <int G, int BK, int STAGES>
constexpr size_t balanced_smem_bytes(int warps) {
  return (size_t)STAGES * (16 * G + 8 * warps) * (BK + 8) * sizeof(__nv_bfloat16);
}

template <typename OutT, int G, int BK, int STAGES>
__global__ void gemm_balanced_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                     const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                     int64_t ldy, int M, int N, int K, int segs) {
  constexpr int kStride = BK + 8, kRows = 16 * G;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int stage_elems = (kRows + 8 * warps) * kStride;
  const int units = N / 8, seg_k = K / segs, items = units * segs;
  const int i0 = (int)((int64_t)blockIdx.x * items / gridDim.x);
  const int i1 = (int)((int64_t)(blockIdx.x + 1) * items / gridDim.x);
  const int ktiles = seg_k / BK;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;
  for (int item = i0; item < i1;) {
    const int seg = item / units, base = item - seg * units;
    const int cnt = min(min(warps, i1 - item), units - base);  // items of this pass
    const int nrows = cnt * 8, k0 = seg * seg_k;
    auto issue = [&](int kt) {
      __nv_bfloat16* st = smem + (kt % STAGES) * stage_elems;
      constexpr int kVec = BK / 8;
      for (int i = threadIdx.x; i < (kRows + nrows) * kVec; i += blockDim.x) {
        const int r = i / kVec, v = i % kVec, kc = k0 + kt * BK + v * 8;
        if (r >= kRows) cp_async16(st + r * kStride + v * 8, w + ((int64_t)base * 8 + r - kRows) * K + kc, 16);
        else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
        else cp_async16(st + r * kStride + v * 8, x, 0);  // zero rows >= M, as gemm_small_kernel
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
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
          for (int gi = 0; gi < G; ++gi) {
            uint32_t a[4];
            ldmatrix_x4(a, sx + (gi * 16 + (mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
            mma_bf16(acc[gi], a, b[2 * kk], b[2 * kk + 1]);
          }
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();
    item += cnt;
    if (!active) continue;
    const int unit = base + warp, n = unit * 8 + t * 2;
    if (segs > 1) {
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = gi * 16 + g + h * 8;
          if (m < M) *reinterpret_cast<float2*>(g_seg_ws + ((int64_t)seg * M + m) * N + n) = make_float2(acc[gi][2 * h], acc[gi][2 * h + 1]);
        }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        last = atomicAdd(g_seg_counters + unit, 1) == segs - 1;
        if (last) g_seg_counters[unit] = 0;  // ready for the next launch
      }
      if (!__shfl_sync(0xffffffffu, last, 0)) continue;
      __threadfence();
#pragma unroll
      for (int gi = 0; gi < G; ++gi)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = gi * 16 + g + h * 8;
          if (m >= M) continue;
          float2 tot = make_float2(0.f, 0.f);
          for (int sg = 0; sg < segs; ++sg) {
            const float2 p = __ldcg(reinterpret_cast<const float2*>(g_seg_ws + ((int64_t)sg * M + m) * N + n));
            tot.x = add(tot.x, p.x);
            tot.y = add(tot.y, p.y);
          }
          acc[gi][2 * h] = tot.x;
          acc[gi][2 * h + 1] = tot.y;
        }
    }
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = gi * 16 + g + h * 8;
        if (m >= M) continue;
        if constexpr (sizeof(OutT) == 2) {
          *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(acc[gi][2 * h], acc[gi][2 * h + 1]);
        } else {
          *reinterpret_cast<float2*>(y + (int64_t)m * ldy + n) = make_float2(acc[gi][2 * h], acc[gi][2 * h + 1]);
        }
      }
    }
  }
}

// Launch: one CTA per SM, as many warps as units per CTA within max_smem.
template <typename OutT, int G, int BK, int STAGES>
cudaError_t run_balanced(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N,
                         int K, int segs, int sms, int max_smem, cudaStream_t st) {
  if (M > 16 * G || !seg_ws_fits(segs, M, N)) return cudaErrorInvalidValue;
  const int units = N / 8 * segs, grid = units < sms ? units : sms;
  int warps = (units + grid - 1) / grid;
  if (warps > 16) warps = 16;
  while (warps > 1 && balanced_smem_bytes<G, BK, STAGES>(warps) > (size_t)max_smem) --warps;
  const size_t smem = balanced_smem_bytes<G, BK, STAGES>(warps);
  if (smem > (size_t)max_smem) return cudaErrorInvalidValue;  // refused, as every shape an id cannot run
  cudaFuncSetAttribute(gemm_balanced_kernel<OutT, G, BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_balanced_kernel<OutT, G, BK, STAGES><<<grid, warps * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K, segs);
  return cudaGetLastError();
}

// ------------------------------------------------------ segment-split, M <= 16 ----
// Decode GEMMs of S > 1 shapes (0.6B o and down: 128 n8 tiles, K = 2048 / 3072) stream a
// few MB in a few microseconds, so the balanced kernel's global fold (partial to g_seg_ws,
// arrival counter, re-read) and the second pipeline pass of the CTA whose items cross
// the segment boundary are a visible share of the time. Here a CTA owns one n8 tile and
// gives each segment its own warp with its own cp.async ring; the per-segment sequence is
// gemm_small_kernel's, and warp 0 folds the partials from shared memory in ascending
// segment order, as gemm_small_kernel's tot does.
template <int BK, int STAGES>
constexpr size_t segsplit_smem_bytes(int segs) {
  return (size_t)segs * STAGES * (16 + 8) * (BK + 8) * sizeof(__nv_bfloat16) + (size_t)segs * 32 * 4 * sizeof(float);
}

template <typename OutT, int BK, int STAGES>
__global__ void gemm_segsplit_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                     const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                     int64_t ldy, int M, int N, int K) {
  constexpr int kStride = BK + 8, kRows = 16, kStageElems = (kRows + 8) * kStride;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, segs = blockDim.x >> 5;
  const int seg_k = K / segs, ktiles = seg_k / BK, k0 = warp * seg_k, n0 = blockIdx.x * 8;
  __nv_bfloat16* ring = smem + warp * STAGES * kStageElems;
  float* part = reinterpret_cast<float*>(smem + segs * STAGES * kStageElems);
  auto issue = [&](int kt) {
    __nv_bfloat16* st = ring + (kt % STAGES) * kStageElems;
    constexpr int kVec = BK / 8;
    for (int i = lane; i < (kRows + 8) * kVec; i += 32) {
      const int r = i / kVec, v = i % kVec, kc = k0 + kt * BK + v * 8;
      if (r >= kRows) cp_async16(st + r * kStride + v * 8, w + (int64_t)(n0 + r - kRows) * K + kc, 16);
      else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
      else cp_async16(st + r * kStride + v * 8, x, 0);  // zero rows >= M, as gemm_small_kernel
    }
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int mat = lane >> 3, r8 = lane & 7;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncwarp();  // the ring is this warp's own: a warp barrier orders its reuse
    if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1);
    cp_async_commit();
    const __nv_bfloat16* sx = ring + (kt % STAGES) * kStageElems;
    const __nv_bfloat16* sw = sx + kRows * kStride;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ks += 2) {
      uint32_t b[4];
      ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t a[4];
        ldmatrix_x4(a, sx + ((mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
        mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
      }
    }
  }
  cp_async_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; ++i) part[(warp * 32 + lane) * 4 + i] = acc[i];
  __syncthreads();
  if (warp) return;
  float tot[4] = {0.f, 0.f, 0.f, 0.f};
  for (int sg = 0; sg < segs; ++sg)
#pragma unroll
    for (int i = 0; i < 4; ++i) tot[i] = add(tot[i], part[(sg * 32 + lane) * 4 + i]);
  const int g = lane >> 2, n = n0 + (lane & 3) * 2;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int m = g + h * 8;
    if (m >= M) continue;
    if constexpr (sizeof(OutT) == 2) {
      *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(tot[2 * h], tot[2 * h + 1]);
    } else {
      *reinterpret_cast<float2*>(y + (int64_t)m * ldy + n) = make_float2(tot[2 * h], tot[2 * h + 1]);
    }
  }
}

// ------------------------------------------------------------ stream-K, M <= 16 ----
// Decode GEMMs whose n8 tiles divide unevenly over the SMs (8B down: 512 tiles on 114
// SMs; 0.6B o and down: 256 (tile, segment) chains on 108) leave the busiest SMs
// streaming one chain more than the average. Here the work is the k-tile steps of all
// (n8 tile, segment) chains, tile-major, split into equal contiguous ranges over one
// warp-sized worker per (SM, warp). A chain cut between workers w and w + 1 is still
// one canonical chain: w runs its first k steps from zero and hands its fp32
// accumulator to w + 1, which continues the chain from it; bits are those of the
// uncut chain. Each worker runs its handed-off prefix first and the continuation
// last, so the wait is short, and waits only on a lower worker whose prefix never
// waits, so it cannot deadlock (all CTAs are resident: grid <= SMs). A finished chain
// of an S = 1 shape is the output; with S > 1 it is a segment partial, folded as
// gemm_balanced_kernel does (g_seg_ws, arrival count, the last to arrive folds the S
// partials in ascending order), which never waits. Workers have their own cp.async
// rings of XR X rows and 8 TN W rows. XR = 8 (M <= 8): rows 8..15 of the MMA's A operand
// are zero, as the zero-filled rows of the other kernels; XR = 16 (M <= 16): the whole A
// operand, at the same cost per weight byte (narrow weights at M = 9..16, e.g.
// tensor-parallel column shards, where tile kernels leave SMs idle). More X rows per warp
// lose: each warp re-reads them from L2 for its 8 TN columns (PI-Attn
// archive/ceiling_probes/a800/narrow/). The handoff state is per device and per
// translation unit, as g_seg_ws.
constexpr int kSkMaxWorkers = 2048, kSkMaxTN = 2;
static __device__ float g_sk_acc[kSkMaxWorkers * kSkMaxTN * 128];
static __device__ int g_sk_flag[kSkMaxWorkers];

// TN n8 tiles per chain: the worker's MMAs on the TN tiles share each X fragment, so
// the X rows it re-reads from L2 per weight byte fall to 1 / TN.
template <typename OutT, int BK, int STAGES, int TN, int XR = 8>
__global__ void gemm_streamk_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                    const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                    int64_t ldy, int M, int N, int K, int segs, int workers) {
  static_assert(TN >= 1 && TN <= kSkMaxTN, "handoff buffer holds kSkMaxTN tiles per worker");
  static_assert(XR == 8 || XR == 16, "X rows: 8 (A rows 8..15 zero) or 16");
  constexpr int kStride = BK + 8, kRows = XR + 8 * TN, kStage = kRows * kStride;
  constexpr int kHalves = XR / 8;  // accumulator rows g and g + 8 (XR = 8: the second is padding)
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wid = blockIdx.x * (blockDim.x >> 5) + warp;
  if (wid >= workers) return;
  __nv_bfloat16* ring = smem + warp * STAGES * kStage;
  const int seg_k = K / segs, kt_n = seg_k / BK;  // k tiles per chain
  const int64_t total = (int64_t)(N / (8 * TN)) * segs * kt_n;
  const int64_t s0 = wid * total / workers, s1 = (wid + 1) * total / workers;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;

  // Chain c (tiles TN (c / segs) .. + TN - 1, segment c % segs), k tiles [ka, kb): from
  // zero (ka == 0) or from worker wid - 1's accumulators; finishes the chain (kb ==
  // kt_n) or hands the accumulators to wid + 1.
  auto piece = [&](int c, int ka, int kb) {
    const int u = c / segs, sg = c % segs;
    auto issue = [&](int kt) {
      __nv_bfloat16* st = ring + (kt % STAGES) * kStage;
      constexpr int kVec = BK / 8;
#pragma unroll
      for (int i = lane; i < kRows * kVec; i += 32) {
        const int r = i / kVec, v = i % kVec, kc = sg * seg_k + kt * BK + v * 8;
        if (r >= XR) cp_async16(st + r * kStride + v * 8, w + ((int64_t)u * 8 * TN + r - XR) * K + kc, 16);
        else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
        else cp_async16(st + r * kStride + v * 8, x, 0);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
      if (ka + s < kb) issue(ka + s);
      cp_async_commit();
    }
    float acc[TN][4];
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    if (ka > 0) {
      if (lane == 0) {
        while (atomicAdd(g_sk_flag + wid - 1, 0) == 0) {}
        g_sk_flag[wid - 1] = 0;  // ready for the next launch
      }
      __syncwarp();
      __threadfence();
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = __ldcg(g_sk_acc + ((wid - 1) * kSkMaxTN + j) * 128 + lane * 4 + i);
    }
    for (int kt = ka; kt < kb; ++kt) {
      cp_async_wait<STAGES - 2>();
      __syncwarp();
      if (kt + STAGES - 1 < kb) issue(kt + STAGES - 1);
      cp_async_commit();
      const __nv_bfloat16* sx = ring + (kt % STAGES) * kStage;
      const __nv_bfloat16* sw = sx + XR * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[TN][4];
#pragma unroll
        for (int j = 0; j < TN; ++j) ldmatrix_x4(b[j], sw + (j * 8 + r8) * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          uint32_t a[4];
          if constexpr (XR == 8) {
            ldmatrix_x4(a, sx + r8 * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
            a[1] = a[3] = 0u;
          } else {
            ldmatrix_x4(a, sx + ((mat & 1) * 8 + r8) * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
          }
#pragma unroll
          for (int j = 0; j < TN; ++j) mma_bf16(acc[j], a, b[j][2 * kk], b[j][2 * kk + 1]);
        }
      }
      __syncwarp();
    }
    cp_async_wait<0>();
    __syncwarp();
    if (kb < kt_n) {
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) g_sk_acc[(wid * kSkMaxTN + j) * 128 + lane * 4 + i] = acc[j][i];
      __threadfence();
      __syncwarp();
      if (lane == 0) atomicExch(g_sk_flag + wid, 1);
      return;
    }
    if (segs > 1) {
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int h = 0; h < kHalves; ++h) {
          const int m = g + h * 8, n = (u * TN + j) * 8 + t * 2;
          if (m < M) *reinterpret_cast<float2*>(g_seg_ws + ((int64_t)sg * M + m) * N + n) = make_float2(acc[j][2 * h], acc[j][2 * h + 1]);
        }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        last = atomicAdd(g_seg_counters + u, 1) == segs - 1;
        if (last) g_seg_counters[u] = 0;  // ready for the next launch
      }
      if (!__shfl_sync(0xffffffffu, last, 0)) return;
      __threadfence();
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int h = 0; h < kHalves; ++h) {
          const int m = g + h * 8, n = (u * TN + j) * 8 + t * 2;
          if (m >= M) continue;
          float2 tot = make_float2(0.f, 0.f);
          for (int q = 0; q < segs; ++q) {
            const float2 p = __ldcg(reinterpret_cast<const float2*>(g_seg_ws + ((int64_t)q * M + m) * N + n));
            tot.x = add(tot.x, p.x);
            tot.y = add(tot.y, p.y);
          }
          acc[j][2 * h] = tot.x;
          acc[j][2 * h + 1] = tot.y;
        }
    }
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int h = 0; h < kHalves; ++h) {
        const int m = g + h * 8, n = (u * TN + j) * 8 + t * 2;
        if (m >= M) continue;
        if constexpr (sizeof(OutT) == 2) {
          *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)m * ldy + n) = __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
        } else {
          *reinterpret_cast<float2*>(y + (int64_t)m * ldy + n) = make_float2(acc[j][2 * h], acc[j][2 * h + 1]);
        }
      }
  };

  const int64_t h_end = min(s1, (s0 + kt_n - 1) / kt_n * kt_n);  // end of the continuation
  const int64_t tail0 = max(h_end, s1 / kt_n * kt_n);            // start of the handed-off prefix
  if (tail0 < s1) piece((int)(tail0 / kt_n), 0, (int)(s1 - tail0));
  for (int64_t s = h_end; s < tail0; s += kt_n) piece((int)(s / kt_n), 0, kt_n);
  if (s0 < h_end) piece((int)(s0 / kt_n), (int)(s0 % kt_n), (int)((h_end - 1) % kt_n) + 1);
}

// Launch: `warps` workers per SM, one CTA per SM; at most one worker per chain, so a
// chain is cut at most once. Refuses M > XR and rings that do not fit in shared memory.
template <typename OutT, int BK, int STAGES, int TN = 1, int XR = 8>
cudaError_t run_streamk(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N,
                        int K, int segs, int warps, int sms, int max_smem, cudaStream_t st) {
  if (M > XR || (K / segs) % BK || N % (8 * TN) || !seg_ws_fits(segs, M, N)) return cudaErrorInvalidValue;
  const int workers = min(min(N / (8 * TN) * segs, sms * warps), kSkMaxWorkers);
  const size_t smem = (size_t)warps * STAGES * (XR + 8 * TN) * (BK + 8) * sizeof(__nv_bfloat16);
  if (smem > (size_t)max_smem) return cudaErrorInvalidValue;
  cudaFuncSetAttribute(gemm_streamk_kernel<OutT, BK, STAGES, TN, XR>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem);
  gemm_streamk_kernel<OutT, BK, STAGES, TN, XR><<<(workers + warps - 1) / warps, warps * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K, segs, workers);
  return cudaGetLastError();
}

}  // namespace pi_gemm
