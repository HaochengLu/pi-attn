// M-invariant GEMM: Y[M,N] = X[M,K] * W[N,K]^T (nn.Linear), bf16 in, fp32 accumulate.
//
// Canonical per-element operation sequence. K is cut into S = gemm_segments(N, K)
// equal segments, a function of the weight shape only. With S = 1: acc = 0; for
// k16 = 0 .. K/16-1 in ascending order: acc = mma.m16n8k16(x[m, k16], w[n, k16],
// acc); y = round(acc). With S > 1 each segment runs that chain over its k16 steps
// into p_s, and y = round((((+0 + p_0) + p_1) + ...) + p_{S-1}). Segments let
// narrow projections (few n8 output tiles over a long K) spread decode weight
// streaming over every SM, the one thing split-K is for, at a cost fixed per
// shape. The small-M kernels below and the CUTLASS tiles in gemm_cutlass.cuh
// realise exactly this sequence, so a row of Y is a pure function of that row of
// X and W, whatever M, the tile shape, or the other rows in the batch (checked
// bitwise by kernels/test_gemm.py).
#pragma once
#include "pi_attn.cuh"

namespace pi_gemm {
using namespace pi_attn;

// K segments of a [N, K] weight: two for narrow, long-K projections (o / down of
// small models, whose n8 tiles alone cannot occupy every SM in decode), else one.
// Segment length is a multiple of 256, so every kernel's K tile divides it.
constexpr int kSegMaxN = 1024, kSegMax = 2;
__host__ __device__ constexpr int gemm_segments(int N, int K) {
  return N <= kSegMaxN && K >= 2048 && K % 512 == 0 ? 2 : 1;
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

  float acc[G][4], tot[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[g][i] = tot[g][i] = 0.f;
  const int segs = gemm_segments(N, K), seg_tiles = ktiles / segs;
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
// not run concurrently (the engine runs its forward on one stream).
static __device__ float g_seg_ws[kSegMax * 64 * kSegMaxN];
static __device__ int g_seg_counters[kSegMaxN / 8];

template <int G, int BK, int STAGES>
constexpr size_t balanced_smem_bytes(int warps) {
  return (size_t)STAGES * (16 * G + 8 * warps) * (BK + 8) * sizeof(__nv_bfloat16);
}

template <typename OutT, int G, int BK, int STAGES>
__global__ void gemm_balanced_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                     const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                     int64_t ldy, int M, int N, int K) {
  constexpr int kStride = BK + 8, kRows = 16 * G;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  const int stage_elems = (kRows + 8 * warps) * kStride;
  const int units = N / 8, segs = gemm_segments(N, K), seg_k = K / segs, items = units * segs;
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
                         int K, int sms, int max_smem, cudaStream_t st) {
  if (M > 16 * G) return cudaErrorInvalidValue;
  const int units = N / 8 * gemm_segments(N, K), grid = units < sms ? units : sms;
  int warps = (units + grid - 1) / grid;
  if (warps > 16) warps = 16;
  while (warps > 1 && balanced_smem_bytes<G, BK, STAGES>(warps) > (size_t)max_smem) --warps;
  const size_t smem = balanced_smem_bytes<G, BK, STAGES>(warps);
  if (smem > (size_t)max_smem) return cudaErrorInvalidConfiguration;
  cudaFuncSetAttribute(gemm_balanced_kernel<OutT, G, BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_balanced_kernel<OutT, G, BK, STAGES><<<grid, warps * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K);
  return cudaGetLastError();
}

// ------------------------------------------------------------- stream-K, M <= 8 ----
// Decode GEMMs whose n8 tiles divide unevenly over the SMs (8B down: 512 tiles on 114
// SMs) leave the busiest SMs streaming one tile more than the average. Here the work
// is the (n8 tile, k tile) steps of all tiles, split into equal contiguous ranges over
// one warp-sized worker per (SM, warp). A tile cut between workers w and w + 1 is
// still one canonical chain: w runs its first k steps from zero and hands its fp32
// accumulator to w + 1, which continues the chain from it; bits are those of the
// uncut chain. Each worker runs its handed-off prefix first and the continuation
// last, so the wait is short, and waits only on a lower worker whose prefix never
// waits, so it cannot deadlock (all CTAs are resident: grid <= SMs). Only for S = 1
// shapes. Workers have their own cp.async rings (8 X rows, 8 W rows; rows 8..15 of the
// MMA's A operand are zero, as the zero-filled rows of the other kernels), streamed
// continuously across the worker's pieces. The
// handoff state is per device and per translation unit, as g_seg_ws.
constexpr int kSkMaxWorkers = 2048;
static __device__ float g_sk_acc[kSkMaxWorkers * 128];
static __device__ int g_sk_flag[kSkMaxWorkers];

template <typename OutT, int BK, int STAGES>
__global__ void gemm_streamk_kernel(const __nv_bfloat16* __restrict__ x, int64_t ldx,
                                    const __nv_bfloat16* __restrict__ w, OutT* __restrict__ y,
                                    int64_t ldy, int M, int N, int K, int workers) {
  constexpr int kStride = BK + 8, kStage = 16 * kStride, kVec = BK / 8;
  extern __shared__ __align__(16) __nv_bfloat16 smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wid = blockIdx.x * (blockDim.x >> 5) + warp;
  if (wid >= workers) return;
  __nv_bfloat16* ring = smem + warp * STAGES * kStage;
  const int kt_n = K / BK;
  const int64_t total = (int64_t)(N / 8) * kt_n;
  const int64_t s0 = wid * total / workers, s1 = (wid + 1) * total / workers;
  const int g = lane >> 2, t = lane & 3, mat = lane >> 3, r8 = lane & 7;

  // The worker's pieces, in execution order: the handed-off prefix of the tile it ends
  // in, the whole tiles, then the continuation of the tile it starts in (from worker
  // wid - 1's accumulator). Piece p is tile u, k tiles [ka, kb).
  const int64_t h_end = min(s1, (s0 + kt_n - 1) / kt_n * kt_n);  // end of the continuation
  const int64_t tail0 = max(h_end, s1 / kt_n * kt_n);            // start of the handed-off prefix
  const int first_full = tail0 < s1 ? 1 : 0, n_full = (int)((tail0 - h_end) / kt_n);
  const int n_pieces = first_full + n_full + (s0 < h_end ? 1 : 0);
  struct Piece {
    int u, ka, kb;
  };
  auto piece_at = [&](int p) -> Piece {
    if (p < first_full) return {(int)(tail0 / kt_n), 0, (int)(s1 - tail0)};
    if (p < first_full + n_full) return {(int)(h_end / kt_n) + p - first_full, 0, kt_n};
    return {(int)(s0 / kt_n), (int)(s0 % kt_n), (int)((h_end - 1) % kt_n) + 1};
  };

  // X rows M..7 stay zero in every stage (MMA rows 8..15 are never read): zero them once
  // instead of zero-filling copies every k tile.
  for (int i = lane; i < STAGES * 8 * kStride; i += 32) {
    const int r = i / kStride % 8;
    if (r >= M) ring[i / (8 * kStride) * kStage + r * kStride + i % kStride] = __float2bfloat16(0.f);
  }
  // One cp.async stream runs through all pieces, so the ring never drains between them.
  int ip = 0, ikt = n_pieces ? piece_at(0).ka : 0, issued = 0;
  Piece iq = n_pieces ? piece_at(0) : Piece{0, 0, 0};
  auto issue_next = [&]() {
    if (ip < n_pieces) {
      __nv_bfloat16* st = ring + (issued % STAGES) * kStage;
#pragma unroll
      for (int i = lane; i < 16 * kVec; i += 32) {
        const int r = i / kVec, v = i % kVec, kc = ikt * BK + v * 8;
        if (r >= 8) cp_async16(st + r * kStride + v * 8, w + ((int64_t)iq.u * 8 + r - 8) * K + kc, 16);
        else if (r < M) cp_async16(st + r * kStride + v * 8, x + (int64_t)r * ldx + kc, 16);
      }
      ++issued;
      if (++ikt == iq.kb && ++ip < n_pieces) {
        iq = piece_at(ip);
        ikt = iq.ka;
      }
    }
    cp_async_commit();
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) issue_next();

  int consumed = 0;
  for (int p = 0; p < n_pieces; ++p) {
    const Piece pc = piece_at(p);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (pc.ka > 0) {
      if (lane == 0) {
        while (atomicAdd(g_sk_flag + wid - 1, 0) == 0) {}
        g_sk_flag[wid - 1] = 0;  // ready for the next launch
      }
      __syncwarp();
      __threadfence();
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __ldcg(g_sk_acc + (wid - 1) * 128 + lane * 4 + i);
    }
    for (int kt = pc.ka; kt < pc.kb; ++kt, ++consumed) {
      cp_async_wait<STAGES - 2>();
      __syncwarp();
      issue_next();
      const __nv_bfloat16* sx = ring + (consumed % STAGES) * kStage;
      const __nv_bfloat16* sw = sx + 8 * kStride;
#pragma unroll
      for (int ks = 0; ks < BK / 16; ks += 2) {
        uint32_t b[4];
        ldmatrix_x4(b, sw + r8 * kStride + ks * 16 + mat * 8);
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          uint32_t a[4];
          ldmatrix_x4(a, sx + r8 * kStride + (ks + kk) * 16 + (mat >> 1) * 8);
          a[1] = a[3] = 0u;
          mma_bf16(acc, a, b[2 * kk], b[2 * kk + 1]);
        }
      }
      __syncwarp();
    }
    if (pc.kb < kt_n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) g_sk_acc[wid * 128 + lane * 4 + i] = acc[i];
      __threadfence();
      __syncwarp();
      if (lane == 0) atomicExch(g_sk_flag + wid, 1);
    } else if (g < M) {
      const int n = pc.u * 8 + t * 2;
      if constexpr (sizeof(OutT) == 2) {
        *reinterpret_cast<__nv_bfloat162*>(y + (int64_t)g * ldy + n) = __floats2bfloat162_rn(acc[0], acc[1]);
      } else {
        *reinterpret_cast<float2*>(y + (int64_t)g * ldy + n) = make_float2(acc[0], acc[1]);
      }
    }
  }
  cp_async_wait<0>();
}

// Launch: `warps` workers per SM, one CTA per SM; at most one worker per n8 tile, so
// a tile is cut at most once.
template <typename OutT, int BK, int STAGES>
cudaError_t run_streamk(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N,
                        int K, int warps, int sms, cudaStream_t st) {
  if (M > 8 || K % BK || gemm_segments(N, K) != 1) return cudaErrorInvalidValue;
  const int workers = min(min(N / 8, sms * warps), kSkMaxWorkers);
  const size_t smem = (size_t)warps * STAGES * 16 * (BK + 8) * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(gemm_streamk_kernel<OutT, BK, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_streamk_kernel<OutT, BK, STAGES><<<(workers + warps - 1) / warps, warps * 32, smem, st>>>(
      (const __nv_bfloat16*)x, ldx, (const __nv_bfloat16*)w, y, ldy, M, N, K, workers);
  return cudaGetLastError();
}

}  // namespace pi_gemm
