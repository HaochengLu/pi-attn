// xcp_kernel.cuh -- portable (cross-class bit-identical, M-invariant) small-M GEMM kernel, program in xcp.h.
//
// One CTA per SM, NW warps. A warp owns one n8 output tile (8 rows of W) for a run of RS canonical segments and
// streams those weight bytes exactly once through its own cp.async ring (8 bytes per lane per k16 block: lane
// (g, t) holds W[n0 + g][16 b + 4 t .. + 3]; the k order inside a block is free because the fused sum is exact).
// The W group (column, block) is the lane quad: its max exponent comes from two shuffles and the two slices from
// five IEEE adds per element in registers. X is sliced once per CTA per chunk of CB blocks into shared memory as
// ready A fragments (double buffered, one barrier per chunk) and shared by all warps. Per block and row group:
// TRI 3 MMAs from zero accumulators + 3 fp32 folds, TRIF 3 MMAs (two chained) + 2 folds.
// Work items (tile, run) are split evenly over the CTAs. RS == S: the warp folds its segments itself and writes
// Y. RS < S (narrow weights, K-parallel): segment partials go to a workspace, every CTA counts itself done, and
// once the count reaches the launch's target each CTA folds its share of tiles in ascending segment order
// (grid <= SMs, so all CTAs are resident and the wait cannot deadlock; launches on one device are serialised).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define XCP_SMEM_DECL extern __shared__ __align__(16) unsigned char smem[]
#define XCP_LAUNCH(kern, grid, block, smem, st, ...) kern<<<grid, block, smem, st>>>(__VA_ARGS__)
#else
#include "cuda_emu.h"  // CPU emulation of the kernel (xcp_cpu.cpp)
#define XCP_SMEM_DECL unsigned char* smem = emu::tl_cta->smem.data()
#define XCP_LAUNCH(kern, grid, block, smem, st, ...) emu::launch(grid, block, smem, [&] { kern(__VA_ARGS__); })
#endif
#include "xcp.h"

namespace xcp {
#ifdef __CUDACC__
__device__ __forceinline__ void cpa8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#else
inline void cpa8(void* dst, const void* src) { memcpy(dst, src, 8); }
inline void cpa16(void* dst, const void* src) { memcpy(dst, src, 16); }
inline void cpa_commit() {}
template <int N> void cpa_wait() {}
using emu::mma;
#endif
struct Prog { int SA, SB, sA, sB, tri, fused, native, wcanon; };  // wcanon: W is already on its slice-0 lattice
constexpr Prog prog_of(int p) {
  return p == 0 ? Prog{2, 2, 2, 2, 1, 0, 0, 0} : p == 1 ? Prog{2, 2, 2, 2, 1, 1, 0, 0} : p == 2 ? Prog{2, 1, 0, 4, 0, 0, 0, 0}
       : p == 3 ? Prog{1, 1, 2, 2, 0, 0, 0, 0} : p == 5 ? Prog{2, 1, 0, 4, 0, 0, 0, 1} : p == 6 ? Prog{1, 1, 2, 2, 0, 0, 0, 1}
       : Prog{1, 1, 0, 0, 0, 0, 1, 0};
}
// Slice the lane's four elements (raw.x = k 4t, 4t+1; raw.y = k 4t+2, 4t+3) of the group held by its lane quad.
// S: slice parameter s. TWO: also produce slice 1. Slices come back packed as bf16 pairs.
template <int S, bool TWO> __device__ __forceinline__ void slice_quad(uint2 raw, uint2& s0, uint2& s1) {
  static_assert(!TWO || S <= 3, "slice 1 needs |x| < 2^22 u1");
  const float f0 = __uint_as_float(raw.x << 16), f1 = __uint_as_float(raw.x & 0xFFFF0000u);
  const float f2 = __uint_as_float(raw.y << 16), f3 = __uint_as_float(raw.y & 0xFFFF0000u);
  float m = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
  const uint32_t eb = __float_as_uint(m) & 0x7F800000u;
  uint32_t cb = eb + (((uint32_t)(16 - S) << 23) | 0x400000u);  // c = 1.5 * 2^23 u0
  if (eb < ((uint32_t)XCP_FLUSH << 23)) cb = 227u << 23;         // flushed group: every element -> +0
  const float c = __uint_as_float(cb);
  const float a0 = __fsub_rn(__fadd_rn(f0, c), c), a1 = __fsub_rn(__fadd_rn(f1, c), c);
  const float a2 = __fsub_rn(__fadd_rn(f2, c), c), a3 = __fsub_rn(__fadd_rn(f3, c), c);
  s0.x = __byte_perm(__float_as_uint(a0), __float_as_uint(a1), 0x7632);
  s0.y = __byte_perm(__float_as_uint(a2), __float_as_uint(a3), 0x7632);
  if (TWO) {
    const float c1 = __uint_as_float(cb - ((uint32_t)(8 + S) << 23));
    const float y0 = __fsub_rn(__fsub_rn(__fadd_rn(f0, c1), c1), a0), y1 = __fsub_rn(__fsub_rn(__fadd_rn(f1, c1), c1), a1);
    const float y2 = __fsub_rn(__fsub_rn(__fadd_rn(f2, c1), c1), a2), y3 = __fsub_rn(__fsub_rn(__fadd_rn(f3, c1), c1), a3);
    s1.x = __byte_perm(__float_as_uint(y0), __float_as_uint(y1), 0x7632);
    s1.y = __byte_perm(__float_as_uint(y2), __float_as_uint(y3), 0x7632);
  } else {
    s1.x = s1.y = 0u;
  }
}

// One (m16, n8, k16) block in three phases, so that a batch of blocks can issue all its independent tensor-core
// steps back to back and only then run the (dependent, canonical-order) fp32 folds: issuing the folds between the
// MMAs would serialise the warp on the MMA latency.
//   mma_l1: MMAs from zero accumulators (native: the chained MMA into acc).  mma_l2: TRIF's exact chain
//   e = X0.W1 -> e += X1.W0.  fold: acc = fl(acc + d) [, + e [, + f]] in program order (NA live components).
template <int PROG>
__device__ __forceinline__ void mma_l1(float (&acc)[4], float (&d)[4], float (&e)[4], float (&f)[4], const uint32_t (&a0)[4],
                                       const uint32_t (&a1)[4], uint2 w0, uint2 w1) {
  constexpr Prog P = prog_of(PROG);
  if (P.native) { mma(acc, a0, w0.x, w0.y); return; }
#pragma unroll
  for (int c = 0; c < 4; ++c) d[c] = e[c] = f[c] = 0.f;
  mma(d, a0, w0.x, w0.y);                                // X0 . W0
  if (P.tri) mma(e, a0, w1.x, w1.y);                     // X0 . W1
  if (P.tri && !P.fused) mma(f, a1, w0.x, w0.y);         // X1 . W0
  if (!P.tri && P.SA == 2) mma(e, a1, w0.x, w0.y);       // X1 . W0
}
template <int PROG> __device__ __forceinline__ void mma_l2(float (&e)[4], const uint32_t (&a1)[4], uint2 w0) {
  constexpr Prog P = prog_of(PROG);
  if (P.tri && P.fused) mma(e, a1, w0.x, w0.y);          // += X1 . W0 inside the tensor core (exact)
}
template <int PROG, int NA>
__device__ __forceinline__ void fold(float (&acc)[4], const float (&d)[4], const float (&e)[4], const float (&f)[4]) {
  constexpr Prog P = prog_of(PROG);
  if (P.native) return;
#pragma unroll
  for (int c = 0; c < NA; ++c) {
    acc[c] = __fadd_rn(acc[c], d[c]);
    if (P.tri && !P.fused) acc[c] = __fadd_rn(__fadd_rn(acc[c], e[c]), f[c]);
    else if (P.tri || P.SA == 2) acc[c] = __fadd_rn(acc[c], e[c]);
  }
}

constexpr int kU = 4;  // W ring: R chunks of kU blocks per warp (R - 1 chunks in flight)
// A fragments in shared memory per (block, row group, lane). XR = 8: {X0.a0, X0.a2, X1.a0, X1.a2}, 16 bytes
// (a1 = a3 = 0 are not stored: shared-memory bytes read per weight byte are what limits this kernel). XR = 16:
// two ready operand quads {a0, a1, a2, a3} (X0, X1), 32 bytes.
template <int G, int XR> constexpr int lane_bytes() { return XR == 8 ? 16 : 32; }

// G row groups of 16 rows (M <= 16 G); XR = 8 (G = 1, M <= 8: A rows 8..15 are zero) or 16. TN: n8 tiles per warp
// (they share every A fragment). CG: W copied as whole 128-byte lines with 16-byte cp.async.cg (two per tile and
// chunk) instead of 8 bytes per lane per block. R: ring depth in chunks.
template <int G, int XR, int NW, int PROG, int CG, int R, int TN>
__global__ void __launch_bounds__(NW * 32)
    kernel(const uint16_t* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w, void* __restrict__ yv, int64_t ldy,
           int M, int N, int K, int RS, int CB, int SEGB, float* __restrict__ ws, int* counter, int target, int out_bf16) {
  static_assert(XR == 16 || G == 1, "XR = 8 only with one row group");
  static_assert(TN == 1 || CG == 1, "several tiles per warp: line-wise weight copies only");
  static_assert(R >= 2 && (R & (R - 1)) == 0, "ring depth: a power of two");
  constexpr Prog P = prog_of(PROG);
  constexpr int LB = lane_bytes<G, XR>();
  constexpr int kR = R, kChunk = kU * 256 * TN;  // bytes of one ring slot: kU blocks of 8 TN weight rows
  constexpr int NA = XR == 8 ? 2 : 4;            // live accumulator components per lane (rows g; rows g + 8)
  constexpr int NBX = (NW <= 16 ? 4 : 2) / (G * TN);
  constexpr int NB = NBX < 1 ? 1 : NBX;          // blocks per MMA batch (registers)
  XCP_SMEM_DECL;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int xhalf = CB * G * 32 * LB;
  unsigned char* const xbuf = smem;
  unsigned char* const ring = smem + 2 * xhalf + warp * (kR * kChunk);
  const int T = N / 8, TU = T / TN, nb = K / 16, S = (nb + SEGB - 1) / SEGB, runs = (S + RS - 1) / RS;
  const int64_t items = (int64_t)TU * runs;  // work item = (unit of TN tiles, run of RS segments), run-major
  const int grid = gridDim.x, cta = blockIdx.x;
  const int64_t i0 = cta * items / grid, i1 = (cta + 1) * items / grid;
  const bool local = runs == 1;

  auto write_y = [&](int tile, int rg, const float* v) {
    const int n = tile * 8 + 2 * t;
#pragma unroll
    for (int h = 0; h < NA / 2; ++h) {
      const int m = rg * 16 + g + 8 * h;
      if (m >= M) continue;
      if (out_bf16) *reinterpret_cast<__nv_bfloat162*>((__nv_bfloat16*)yv + (int64_t)m * ldy + n) = __floats2bfloat162_rn(v[2 * h], v[2 * h + 1]);
      else *reinterpret_cast<float2*>((float*)yv + (int64_t)m * ldy + n) = make_float2(v[2 * h], v[2 * h + 1]);
    }
  };

  // Slice blocks [bx, bx + cbn) of X (all row groups) into xbuf half `buf`: task (block, row group) -> warp.
  auto slice_x = [&](int bx, int cbn, int buf) {
    for (int tid = warp; tid < cbn * G; tid += NW) {
      const int jb = tid / G, rg = tid - jb * G, row = rg * 16 + g;
      const int64_t ko = (int64_t)(bx + jb) * 16 + 4 * t;
      uint2 r0 = make_uint2(0u, 0u), r1 = make_uint2(0u, 0u);
      if (row < M) r0 = *reinterpret_cast<const uint2*>(x + (int64_t)row * ldx + ko);
      if (XR == 16 && row + 8 < M) r1 = *reinterpret_cast<const uint2*>(x + (int64_t)(row + 8) * ldx + ko);
      uint2 p0 = r0, p1 = make_uint2(0u, 0u), q0 = r1, q1 = make_uint2(0u, 0u);
      if (!P.native) {
        slice_quad<P.sA, P.SA == 2>(r0, p0, p1);
        if (XR == 16) slice_quad<P.sA, P.SA == 2>(r1, q0, q1);
      }
      unsigned char* d = xbuf + buf * xhalf + (tid * 32 + lane) * LB;
      if (XR == 8) {
        *reinterpret_cast<uint4*>(d) = make_uint4(p0.x, p0.y, p1.x, p1.y);  // X0 {a0, a2}, X1 {a0, a2}
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(p0.x, q0.x, p0.y, q0.y);  // X0 {a0, a1, a2, a3}
        *reinterpret_cast<uint4*>(d + 16) = make_uint4(p1.x, q1.x, p1.y, q1.y);
      }
    }
  };

  // One pass: units [base, base + cnt), one per warp, over segments [sa, sb) of the canonical fold.
  auto pass = [&](int run, int base, int cnt) {
    const int sa = run * RS, sb = min(S, sa + RS);
    const int b0 = sa * SEGB, nblk = min(nb, sb * SEGB) - b0;
    const bool active = warp < cnt;
    const int tile0 = (base + warp) * TN;
    // Ring slot = one chunk: kU blocks of the unit's 8 TN weight rows. CG = 0 (TN = 1): laid out
    // [block][lane][8 B], every lane copies its own 8 bytes per block (cp.async.ca). CG = 1: laid out
    // [tile][row][128 B]; the warp copies whole 128-byte lines, 16 B per lane, in two cp.async.cg per tile
    // (rows 0..3, rows 4..7), and lanes read each other's bytes.
    const uint16_t* wrow = CG ? w + (int64_t)(tile0 * 8 + (lane >> 3)) * K + (int64_t)b0 * 16 + (lane & 7) * 8
                              : w + (int64_t)(tile0 * 8 + g) * K + (int64_t)b0 * 16 + 4 * t;
    unsigned char* const ring_l = ring + (CG ? lane * 16 : lane * 8);                 // copy destination of this lane
    const unsigned char* const ring_r = ring + (CG ? g * 128 + t * 8 : lane * 8);    // read position of this lane
    constexpr int kStep = CG ? 32 : 256;                                             // bytes between blocks at ring_r
    auto issue = [&](int wc) {  // chunk wc = blocks [wc kU, wc kU + kU) of the pass: all inside it or all outside
      if (wc * kU < nblk) {
        unsigned char* d = ring_l + (wc & (kR - 1)) * kChunk;
        const uint16_t* src = wrow + (int64_t)wc * (kU * 16);
        if (CG) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            cpa16(d + j * 1024, src + (int64_t)(8 * j) * K);
            cpa16(d + j * 1024 + 512, src + (int64_t)(8 * j + 4) * K);
          }
        } else {
#pragma unroll
          for (int u = 0; u < kU; ++u) cpa8(d + u * 256, src + u * 16);
        }
      }
      cpa_commit();
    };
    if (active) {
#pragma unroll
      for (int wc = 0; wc < kR - 1; ++wc) issue(wc);
    }
    slice_x(b0, min(CB, nblk), 0);
    __syncthreads();
    float acc[G][TN][4], tot[G][TN][4];
#pragma unroll
    for (int rg = 0; rg < G; ++rg)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[rg][j][i] = tot[rg][j][i] = 0.f;
    int seg = sa, seg_left = min(SEGB, nblk);  // blocks left in the current segment
    for (int q = 0, cb0 = 0; cb0 < nblk; ++q, cb0 += CB) {
      const int cbn = min(CB, nblk - cb0);
      if (cb0 + CB < nblk) slice_x(b0 + cb0 + CB, min(CB, nblk - cb0 - CB), (q + 1) & 1);
      if (active) {
        const unsigned char* xp = xbuf + (q & 1) * xhalf + lane * LB;
        for (int wc = cb0 / kU; wc < (cb0 + cbn) / kU; ++wc) {
          cpa_wait<kR - 2>();
          if (CG) __syncwarp();
          issue(wc + kR - 1);
          const unsigned char* rp = ring_r + (wc & (kR - 1)) * kChunk;
#pragma unroll
          for (int ub = 0; ub < kU; ub += NB) {  // batch of NB blocks: all MMAs first, then the folds
            float d[NB][G][TN][4], e[NB][G][TN][4], f[NB][G][TN][4];
            uint32_t a1[NB][G][4];
            uint2 w0s[NB][TN];
#pragma unroll
            for (int v = 0; v < NB; ++v) {
              const int u = ub + v;
              uint2 w0[TN], w1[TN];
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                const uint2 wr = *reinterpret_cast<const uint2*>(rp + j * 1024 + u * kStep);
                w0[j] = wr; w1[j] = make_uint2(0u, 0u);
                if (!P.native && !P.wcanon) slice_quad<P.sB, P.SB == 2>(wr, w0[j], w1[j]);
                w0s[v][j] = w0[j];
              }
#pragma unroll
              for (int rg = 0; rg < G; ++rg) {
                const unsigned char* xq = xp + (u * G + rg) * 32 * LB;
                uint32_t a0[4];
                const uint4 x4 = *reinterpret_cast<const uint4*>(xq);
                if (XR == 8) {
                  a0[0] = x4.x; a0[1] = 0u; a0[2] = x4.y; a0[3] = 0u;
                  a1[v][rg][0] = x4.z; a1[v][rg][1] = 0u; a1[v][rg][2] = x4.w; a1[v][rg][3] = 0u;
                } else {
                  a0[0] = x4.x; a0[1] = x4.y; a0[2] = x4.z; a0[3] = x4.w;
                  if (P.SA == 2) {
                    const uint4 y4 = *reinterpret_cast<const uint4*>(xq + 16);
                    a1[v][rg][0] = y4.x; a1[v][rg][1] = y4.y; a1[v][rg][2] = y4.z; a1[v][rg][3] = y4.w;
                  } else {
                    a1[v][rg][0] = a1[v][rg][1] = a1[v][rg][2] = a1[v][rg][3] = 0u;
                  }
                }
#pragma unroll
                for (int j = 0; j < TN; ++j) mma_l1<PROG>(acc[rg][j], d[v][rg][j], e[v][rg][j], f[v][rg][j], a0, a1[v][rg], w0[j], w1[j]);
              }
            }
#pragma unroll
            for (int v = 0; v < NB; ++v)
#pragma unroll
              for (int rg = 0; rg < G; ++rg)
#pragma unroll
                for (int j = 0; j < TN; ++j) mma_l2<PROG>(e[v][rg][j], a1[v][rg], w0s[v][j]);
#pragma unroll
            for (int v = 0; v < NB; ++v)
#pragma unroll
              for (int rg = 0; rg < G; ++rg)
#pragma unroll
                for (int j = 0; j < TN; ++j) fold<PROG, NA>(acc[rg][j], d[v][rg][j], e[v][rg][j], f[v][rg][j]);
          }
          xp += kU * G * 32 * LB;
          seg_left -= kU;
          if (seg_left == 0) {  // segment done
#pragma unroll
            for (int rg = 0; rg < G; ++rg)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                if (local) {
#pragma unroll
                  for (int c = 0; c < NA; ++c) tot[rg][j][c] = __fadd_rn(tot[rg][j][c], acc[rg][j][c]);
                } else {
                  float* p = ws + (((int64_t)seg * T + tile0 + j) * G + rg) * 128 + lane * 4;
                  const int m = rg * 16 + g;
                  if (m < M) *reinterpret_cast<float2*>(p) = make_float2(acc[rg][j][0], acc[rg][j][1]);
                  if (XR == 16 && m + 8 < M) *reinterpret_cast<float2*>(p + 2) = make_float2(acc[rg][j][2], acc[rg][j][3]);
                }
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[rg][j][c] = 0.f;
              }
            ++seg;
            seg_left = min(SEGB, nblk - (wc + 1) * kU);
          }
        }
      }
      __syncthreads();
    }
    if (active) {
      cpa_wait<0>();
      if (local) {
#pragma unroll
        for (int rg = 0; rg < G; ++rg)
#pragma unroll
          for (int j = 0; j < TN; ++j) write_y(tile0 + j, rg, tot[rg][j]);
      }
    }
  };

  for (int64_t item = i0; item < i1;) {
    const int run = (int)(item / TU), base = (int)(item - (int64_t)run * TU);
    const int rem = (int)min(i1 - item, (int64_t)(TU - base));
    const int np = (rem + NW - 1) / NW, cnt = (rem + np - 1) / np;  // even passes
    pass(run, base, cnt);
    item += cnt;
  }
  if (local) return;

  // every partial of this CTA is written: count it, wait for all CTAs of this launch, fold this CTA's share
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(counter, 1);
    while (target - atomicAdd(counter, 0) > 0) {}
  }
  __syncthreads();
  const int t0 = (int)((int64_t)cta * T / grid), t1 = (int)((int64_t)(cta + 1) * T / grid);
  for (int tile = t0 + warp; tile < t1; tile += NW) {
#pragma unroll
    for (int rg = 0; rg < G; ++rg) {
      float tot[4] = {0.f, 0.f, 0.f, 0.f};
      const int m = rg * 16 + g;
      for (int s = 0; s < S; ++s) {
        const float* p = ws + (((int64_t)s * T + tile) * G + rg) * 128 + lane * 4;
        if (m < M) {
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p));
          tot[0] = __fadd_rn(tot[0], v.x); tot[1] = __fadd_rn(tot[1], v.y);
        }
        if (XR == 16 && m + 8 < M) {
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p + 2));
          tot[2] = __fadd_rn(tot[2], v.x); tot[3] = __fadd_rn(tot[3], v.y);
        }
      }
      write_y(tile, rg, tot);
    }
  }
}

// ------------------------------------------------------------------ host ----
struct Device {
  int sms = 0, smem_max = 0;
  float* ws = nullptr; size_t ws_floats = 0;
  int* counter = nullptr; int target = 0;  // CTAs finished over all K-parallel launches so far (wraps with int)
};
inline cudaError_t device_init(Device& d, size_t ws_floats = (size_t)24 << 20) {
  cudaError_t e;
  if ((e = cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, 0)) != cudaSuccess) return e;
  if ((e = cudaDeviceGetAttribute(&d.smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&d.ws, ws_floats * 4)) != cudaSuccess) return e;
  d.ws_floats = ws_floats;
  if ((e = cudaMalloc(&d.counter, 64)) != cudaSuccess) return e;
  d.target = 0;
  return cudaMemset(d.counter, 0, 64);
}
struct Plan { int grid, RS, CB, S, local; size_t smem; };
// RS: segments per run. Chosen to keep the most warps busy per CTA; among near-equal choices the longest run
// (RS == S needs no workspace). Scheduling only: never changes a bit of Y.
inline Plan make_plan(int G, int LB, int NW, int R, int TN, int N, int K, int SEGB, const Device& d, int rs_force, int grid_force) {
  Plan p;
  const int TU = N / (8 * TN), nb = K / 16;
  p.S = (nb + SEGB - 1) / SEGB;
  p.grid = grid_force > 0 ? grid_force : d.sms;
  p.grid = (int)std::min<int64_t>(p.grid, (int64_t)TU * p.S);
  double best = 0;
  std::vector<double> act(p.S + 1);
  for (int rs = 1; rs <= p.S; ++rs) {
    const double cpc = (double)TU * ((p.S + rs - 1) / rs) / p.grid;
    const double passes = std::ceil(cpc / NW - 1e-9);
    act[rs] = cpc / (passes < 1 ? 1 : passes);
    best = std::max(best, act[rs]);
  }
  p.RS = 1;
  for (int rs = p.S; rs >= 1; --rs) if (act[rs] >= 0.9 * best) { p.RS = rs; break; }
  if (rs_force > 0) p.RS = std::min(rs_force, p.S);
  p.local = (p.S + p.RS - 1) / p.RS == 1;
  p.CB = 0;
  for (int cb = std::min(32, SEGB); cb >= kU; cb /= 2) {
    p.smem = (size_t)2 * cb * G * 32 * LB + (size_t)NW * R * kU * 256 * TN;
    if (SEGB % cb == 0 && p.smem <= (size_t)d.smem_max) { p.CB = cb; break; }
  }
  return p;
}
template <int G, int XR, int NW, int PROG, int CG, int R = 4, int TN = 1>
cudaError_t run(Device& d, const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N, int K, int out_bf16,
                int SEGB, int rs_force, int grid_force, cudaStream_t st, Plan* plan_out = nullptr) {
  if (M < 1 || M > (XR == 8 ? 8 : 16 * G) || N % (8 * TN) || K % 64 || SEGB % kU) return cudaErrorInvalidValue;
  const Plan p = make_plan(G, lane_bytes<G, XR>(), NW, R, TN, N, K, SEGB, d, rs_force, grid_force);
  if (plan_out) *plan_out = p;
  if (!p.CB || p.grid > d.sms) return cudaErrorInvalidValue;
  if (!p.local && (size_t)p.S * (N / 8) * G * 128 > d.ws_floats) return cudaErrorInvalidValue;
  auto kern = kernel<G, XR, NW, PROG, CG, R, TN>;
  static size_t smem_set = 0;
  if (p.smem > smem_set) {
    const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)p.smem);
    if (e != cudaSuccess) return e;
    smem_set = p.smem;
  }
  const int target = (int)((unsigned)d.target + (unsigned)p.grid);
  XCP_LAUNCH(kern, p.grid, NW * 32, p.smem, st, (const uint16_t*)x, ldx, (const uint16_t*)w, y, ldy, M, N, K, p.RS, p.CB, SEGB, d.ws,
             d.counter, target, out_bf16);
  const cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess && !p.local) d.target = target;
  return e;
}

// ------------------------------------------------------------------ large M (prefill) ----
// CUTLASS-style tile kernel for the same program: CTA tile BM x BN = (16 BMW WR) x (8 BNW WC), one warp per
// (WR, WC) cell. Per stage of 4 k16 blocks: raw X and W tiles arrive by cp.async (two stages ahead), every group
// is sliced ONCE into shared memory as ready A / B fragments, then each warp runs its BMW x BNW block of
// (m16, n8) tiles: per k16 block 2 BMW + BNW 16-byte loads and BMW BNW x (3 MMAs + 2 folds) for TRIF.
// Segment partials are folded in registers in canonical order. Same bits as the small-M kernel.
#ifdef __CUDACC__
__device__ __forceinline__ void cpa16z(void* dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes) : "memory");
}
#else
inline void cpa16z(void* dst, const void* src, int bytes) { memset(dst, 0, 16); memcpy(dst, src, bytes); }
#endif
constexpr int kTB = 4;  // k16 blocks per stage
template <int BMW, int BNW, int WR, int WC> constexpr size_t tile_smem() {
  return (size_t)kTB * (BMW * WR) * 1024 + (size_t)kTB * (BNW * WC) * 512 + (size_t)2 * (BMW * WR * 16 + BNW * WC * 8) * (kTB * 16) * 2;
}
template <int BMW, int BNW, int WR, int WC, int PROG>
__global__ void __launch_bounds__(WR * WC * 32)
    tile_kernel(const uint16_t* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ w, void* __restrict__ yv, int64_t ldy, int M, int N,
                int K, int SEGB, int gx, int out_bf16) {
  constexpr Prog P = prog_of(PROG);
  constexpr int NWARP = WR * WC, TM = BMW * WR, TN = BNW * WC, XROWS = TM * 16, WROWS = TN * 8, SK = kTB * 16;
  constexpr int RAW = (XROWS + WROWS) * SK * 2;  // bytes of one raw stage
  XCP_SMEM_DECL;
  unsigned char* const xf = smem;                       // [kTB][TM][32][32]
  unsigned char* const wf = xf + kTB * TM * 1024;       // [kTB][TN][32][16]
  unsigned char* const raw = wf + kTB * TN * 512;       // [2]{X [XROWS][SK], W [WROWS][SK]} bf16
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int bx = blockIdx.x % gx, by = blockIdx.x / gx, m0 = by * TM * 16, n0 = bx * TN * 8;
  const int nb = K / 16, ns = nb / kTB;
  const int wr = warp / WC, wc = warp % WC;

  auto issue_raw = [&](int st) {
    unsigned char* dst = raw + (st & 1) * RAW;
    for (int i = threadIdx.x; i < (XROWS + WROWS) * (SK / 8); i += NWARP * 32) {
      const int r = i / (SK / 8), v = i % (SK / 8);
      const int64_t kc = (int64_t)st * SK + v * 8;
      if (r < XROWS) { const bool in = m0 + r < M; cpa16z(dst + (r * SK + v * 8) * 2, in ? x + (int64_t)(m0 + r) * ldx + kc : x, in ? 16 : 0); }
      else cpa16z(dst + (r * SK + v * 8) * 2, w + (int64_t)(n0 + r - XROWS) * K + kc, 16);
    }
    cpa_commit();
  };
  auto slice_stage = [&](int st) {
    const unsigned char* src = raw + (st & 1) * RAW;
    for (int tid = warp; tid < kTB * (TM + TN); tid += NWARP) {
      const int kb = tid / (TM + TN), u = tid - kb * (TM + TN);
      if (u < TM) {
        const uint2 r0 = *reinterpret_cast<const uint2*>(src + ((u * 16 + g) * SK + kb * 16 + 4 * t) * 2);
        const uint2 r1 = *reinterpret_cast<const uint2*>(src + ((u * 16 + g + 8) * SK + kb * 16 + 4 * t) * 2);
        uint2 p0 = r0, p1 = make_uint2(0u, 0u), q0 = r1, q1 = make_uint2(0u, 0u);
        if (!P.native) { slice_quad<P.sA, P.SA == 2>(r0, p0, p1); slice_quad<P.sA, P.SA == 2>(r1, q0, q1); }
        unsigned char* d = xf + ((kb * TM + u) * 32 + lane) * 32;
        *reinterpret_cast<uint4*>(d) = make_uint4(p0.x, q0.x, p0.y, q0.y);
        *reinterpret_cast<uint4*>(d + 16) = make_uint4(p1.x, q1.x, p1.y, q1.y);
      } else {
        const int nt = u - TM;
        const uint2 r0 = *reinterpret_cast<const uint2*>(src + ((XROWS + nt * 8 + g) * SK + kb * 16 + 4 * t) * 2);
        uint2 p0 = r0, p1 = make_uint2(0u, 0u);
        if (!P.native && !P.wcanon) slice_quad<P.sB, P.SB == 2>(r0, p0, p1);
        *reinterpret_cast<uint4*>(wf + ((kb * TN + nt) * 32 + lane) * 16) = make_uint4(p0.x, p0.y, p1.x, p1.y);
      }
    }
  };
  float acc[BMW][BNW][4], tot[BMW][BNW][4];
#pragma unroll
  for (int mi = 0; mi < BMW; ++mi)
#pragma unroll
    for (int ni = 0; ni < BNW; ++ni)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[mi][ni][c] = tot[mi][ni][c] = 0.f;
  issue_raw(0);
  if (ns > 1) issue_raw(1); else cpa_commit();
  for (int st = 0; st < ns; ++st) {
    cpa_wait<1>();
    __syncthreads();
    slice_stage(st);
    __syncthreads();
    if (st + 2 < ns) issue_raw(st + 2); else cpa_commit();
#pragma unroll
    for (int kb = 0; kb < kTB; ++kb) {
      uint4 bw[BNW];
#pragma unroll
      for (int ni = 0; ni < BNW; ++ni) bw[ni] = *reinterpret_cast<const uint4*>(wf + ((kb * TN + wc * BNW + ni) * 32 + lane) * 16);
      float d[BMW][BNW][4], e[BMW][BNW][4], f[BMW][BNW][4];
      uint32_t a1[BMW][4];
#pragma unroll
      for (int mi = 0; mi < BMW; ++mi) {
        const unsigned char* xp = xf + ((kb * TM + wr * BMW + mi) * 32 + lane) * 32;
        const uint4 v0 = *reinterpret_cast<const uint4*>(xp), v1 = *reinterpret_cast<const uint4*>(xp + 16);
        const uint32_t a0[4] = {v0.x, v0.y, v0.z, v0.w};
        a1[mi][0] = v1.x; a1[mi][1] = v1.y; a1[mi][2] = v1.z; a1[mi][3] = v1.w;
#pragma unroll
        for (int ni = 0; ni < BNW; ++ni)
          mma_l1<PROG>(acc[mi][ni], d[mi][ni], e[mi][ni], f[mi][ni], a0, a1[mi], make_uint2(bw[ni].x, bw[ni].y), make_uint2(bw[ni].z, bw[ni].w));
      }
#pragma unroll
      for (int mi = 0; mi < BMW; ++mi)
#pragma unroll
        for (int ni = 0; ni < BNW; ++ni) mma_l2<PROG>(e[mi][ni], a1[mi], make_uint2(bw[ni].x, bw[ni].y));
#pragma unroll
      for (int mi = 0; mi < BMW; ++mi)
#pragma unroll
        for (int ni = 0; ni < BNW; ++ni) fold<PROG, 4>(acc[mi][ni], d[mi][ni], e[mi][ni], f[mi][ni]);
    }
    if (((st + 1) * kTB) % SEGB == 0 || st + 1 == ns) {
#pragma unroll
      for (int mi = 0; mi < BMW; ++mi)
#pragma unroll
        for (int ni = 0; ni < BNW; ++ni)
#pragma unroll
          for (int c = 0; c < 4; ++c) { tot[mi][ni][c] = __fadd_rn(tot[mi][ni][c], acc[mi][ni][c]); acc[mi][ni][c] = 0.f; }
    }
  }
  cpa_wait<0>();
#pragma unroll
  for (int mi = 0; mi < BMW; ++mi)
#pragma unroll
    for (int ni = 0; ni < BNW; ++ni)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int m = m0 + (wr * BMW + mi) * 16 + g + 8 * h, n = n0 + (wc * BNW + ni) * 8 + 2 * t;
        if (m >= M) continue;
        if (out_bf16) *reinterpret_cast<__nv_bfloat162*>((__nv_bfloat16*)yv + (int64_t)m * ldy + n) = __floats2bfloat162_rn(tot[mi][ni][2 * h], tot[mi][ni][2 * h + 1]);
        else *reinterpret_cast<float2*>((float*)yv + (int64_t)m * ldy + n) = make_float2(tot[mi][ni][2 * h], tot[mi][ni][2 * h + 1]);
      }
}
template <int BMW, int BNW, int WR, int WC, int PROG>
cudaError_t run_tile(const Device& d, const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N, int K, int out_bf16, int SEGB,
                     cudaStream_t st) {
  const int BM = 16 * BMW * WR, BN = 8 * BNW * WC;
  const size_t smem = tile_smem<BMW, BNW, WR, WC>();
  if (M < 1 || N % BN || K % (kTB * 16) || SEGB % kTB || smem > (size_t)d.smem_max) return cudaErrorInvalidValue;
  auto kern = tile_kernel<BMW, BNW, WR, WC, PROG>;
  const cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;
  const int gx = N / BN, gy = (M + BM - 1) / BM;
  XCP_LAUNCH(kern, gx * gy, WR * WC * 32, smem, st, (const uint16_t*)x, ldx, (const uint16_t*)w, y, ldy, M, N, K, SEGB, gx, out_bf16);
  return cudaGetLastError();
}
}  // namespace xcp
