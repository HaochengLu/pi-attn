// gx.cu -- class-independent GEMM prototypes: Y[M,N] = X[M,K] W[N,K]^T, bf16 in, bf16 (or fp32) out.
//
// Every tensor-core partial in the "exact" variants is exactly representable, so no tensor-core
// alignment/truncation rule can fire; partials are combined with IEEE fp32 mul/add on CUDA cores in a
// fixed order. The value is then a pure function of (row of X, W), independent of GPU class, M, tiling.
//
// Block fixed point. K is cut into blocks of B columns. For a row r and block j, E = max biased
// exponent of the block's nonzero elements. A bf16 element is +-mm * 2^(e-134), mm < 256.
//   fmt 0 (16u): v = RNE(mm * 2^(7-(E-e))), |v| <= 32640, two's complement hi (s8) / lo (u8) planes,
//                scale 2^(E-141)
//   fmt 1 (16b): v = RNE(mm * 2^(6-(E-e))), |v| <= 16320, balanced digits hi (s8) / lo (s8), 2^(E-140)
//   fmt 2 (8)  : v = min(RNE(mm * 2^(-1-(E-e))), 127), one s8 plane, scale 2^(E-133)
//   fmt 3 (FP) : x~ = x rounded (RNE) to a multiple of 2^(max(E-s,1)-134), stored as bf16, no scale
// GEMM modes (block partial P_j, exact integer; f_j = RN_fp32(P_j); tot = fl(tot + fl(fl(f_j sx) sw))):
//   0 F : bf16 mma.m16n8k16 chain over the block from zero (exact iff the fmt-3 widths fit fp32);
//         with raw inputs and B = K this is the plain native chain (class dependent)
//   1 I1: x8 . w8                          (1 IMMA / k32)
//   2 I2: x16u . w8                        (2 IMMA / k32)
//   3 I3: x16b . w16b minus lo.lo          (3 IMMA / k32)
//   4 I4: x16u . w16u exact                (4 IMMA / k32)
//   5 K3: x14 . w14 exact, Karatsuba       (3 IMMA / k32); fmt 4: v = RNE(mm * 2^(5-(E-e))), |v| <= 8160,
//         radix-128 digits hi, lo (s8) with |hi - lo| <= 127, scale 2^(E-139)
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

__device__ __forceinline__ void cp_async16(uint32_t s, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void mma_f(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#define DEF_IMMA(NAME, TA, TB)                                                                    \
  __device__ __forceinline__ void NAME(int* d, const uint32_t* a, const uint32_t* b) {            \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32." #TA "." #TB                             \
                 ".s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                   \
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])                                 \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));             \
  }
DEF_IMMA(mma_ss, s8, s8)
DEF_IMMA(mma_su, s8, u8)
DEF_IMMA(mma_us, u8, s8)
DEF_IMMA(mma_uu, u8, u8)

__device__ __forceinline__ uint16_t f2bf(float f) {  // fp32 -> bf16, round to nearest even
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ uint32_t bsub(uint32_t a, uint32_t b) {  // per-byte a - b (mod 256)
  const uint32_t H = 0x80808080u;
  return ((a | H) - (b & ~H)) ^ ((a ^ ~b) & H);
}

// smem row stride in bytes: BKB payload + 16 (conflict-free ldmatrix for BKB = 64 and 128)
template <int MODE, int G, int WN, int WMW, int WNW, int STAGES, int BKB>
struct Cfg {
  static constexpr int kST = BKB + 16;
  static constexpr int VPR = BKB / 16;
  static constexpr int PX = (MODE == 0 || MODE == 1) ? 1 : 2;
  static constexpr int PW = (MODE >= 3) ? 2 : 1;
  static constexpr int BM = 16 * G * WMW, BN = 8 * WN * WNW;
  static constexpr int ROWS = PX * BM + PW * BN;
  static constexpr int SMEM = STAGES * ROWS * kST;
  static constexpr int THREADS = WMW * WNW * 32;
  static constexpr int NV = (ROWS * VPR + THREADS - 1) / THREADS;  // cp.async vectors per thread
};

// xs: PX planes of [M][RB] bytes (RB = row bytes: K for int8 planes, 2K for bf16), plane stride xplane.
// ws: PW planes of [N][RB] bytes. sx [M][nb], sw [N][nb] block scales (nullptr in mode 0).
// One step = 32 row bytes = one MMA k-extent (k32 int8 or k16 bf16). Fold every bsteps steps.
template <int MODE, int G, int WN, int WMW, int WNW, int STAGES, int BKB>
__global__ void __launch_bounds__(WMW * WNW * 32)
    gx_kernel(const uint8_t* __restrict__ xs, int64_t xplane, const uint8_t* __restrict__ ws,
              int64_t wplane, const float* __restrict__ sx, const float* __restrict__ sw,
              void* __restrict__ y, int M, int N, int RB, int bsteps, int out32) {
  typedef Cfg<MODE, G, WN, WMW, WNW, STAGES, BKB> C;
  constexpr int kST = C::kST;
  extern __shared__ __align__(16) uint8_t smem[];
  const uint32_t sm32 = (uint32_t)__cvta_generic_to_shared(smem);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WNW, wn = warp % WNW;
  const int m0 = blockIdx.x * C::BM, n0 = blockIdx.y * C::BN;
  const int ktiles = RB / BKB;
  const int nb = (RB / 32) / bsteps;

  // Per-thread cp.async table: this thread always copies the same NV 16-byte vectors of a k tile.
  const uint8_t* vsrc[C::NV];
  uint32_t vdst[C::NV];
  int vbytes[C::NV];
#pragma unroll
  for (int j = 0; j < C::NV; ++j) {
    const int i = tid + j * C::THREADS;
    const int r = i / C::VPR, v = i % C::VPR;
    vsrc[j] = ws;
    vbytes[j] = i < C::ROWS * C::VPR ? 0 : -1;
    vdst[j] = sm32 + r * kST + v * 16;
    if (i < C::ROWS * C::VPR) {
      if (r < C::PX * C::BM) {
        const int p = r / C::BM, rr = r % C::BM;
        if (m0 + rr < M) {
          vsrc[j] = xs + p * xplane + (int64_t)(m0 + rr) * RB + v * 16;
          vbytes[j] = 16;
        }
      } else {
        const int q = r - C::PX * C::BM;
        const int p = q / C::BN, rr = q % C::BN;
        vsrc[j] = ws + p * wplane + (int64_t)(n0 + rr) * RB + v * 16;
        vbytes[j] = 16;
      }
    }
  }
  auto issue = [&](int kt, int stage) {
    const uint32_t so = stage * (C::ROWS * kST);
    const int64_t ko = (int64_t)kt * BKB;
#pragma unroll
    for (int j = 0; j < C::NV; ++j)
      if (vbytes[j] >= 0) cp_async16(vdst[j] + so, vsrc[j] + ko, vbytes[j]);
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) issue(s, s);
    cp_async_commit();
  }

  float tot[G][WN][4], facc[G][WN][4];
  int iacc[3][G][WN][4];
#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int c = 0; c < WN; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        tot[g][c][i] = 0.f;
        facc[g][c][i] = 0.f;
        iacc[0][g][c][i] = iacc[1][g][c][i] = iacc[2][g][c][i] = 0;
      }

  const int mat = lane >> 3, r8 = lane & 7;
  const int row_base = m0 + wm * 16 * G + (lane >> 2);
  const int col_base = n0 + wn * 8 * WN + (lane & 3) * 2;
  int cnt = 0, blk = 0;

  auto fold = [&]() {
    float sxv[G][2], swv[WN][2];
    if (sx != nullptr) {
#pragma unroll
      for (int g = 0; g < G; ++g)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = row_base + g * 16 + h * 8;
          sxv[g][h] = row < M ? sx[(int64_t)row * nb + blk] : 0.f;
        }
#pragma unroll
      for (int c = 0; c < WN; ++c)
#pragma unroll
        for (int h = 0; h < 2; ++h) swv[c][h] = sw[(int64_t)(col_base + c * 8 + h) * nb + blk];
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
      for (int c = 0; c < WN; ++c)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float f;
          if constexpr (MODE == 0) {
            f = facc[g][c][i];
            facc[g][c][i] = 0.f;
          } else {
            long long P;
            if constexpr (MODE == 1) P = iacc[0][g][c][i];
            if constexpr (MODE == 2) P = iacc[0][g][c][i] * 256LL + iacc[1][g][c][i];
            if constexpr (MODE == 3) P = iacc[0][g][c][i] * 65536LL + iacc[1][g][c][i] * 256LL;
            if constexpr (MODE == 4)
              P = iacc[0][g][c][i] * 65536LL + iacc[1][g][c][i] * 256LL + iacc[2][g][c][i];
            if constexpr (MODE == 5)  // Karatsuba: mid = hh + ll - (hi-lo)(hi'-lo')
              P = iacc[0][g][c][i] * 16384LL +
                  (long long)(iacc[0][g][c][i] + iacc[2][g][c][i] - iacc[1][g][c][i]) * 128LL +
                  iacc[2][g][c][i];
            iacc[0][g][c][i] = iacc[1][g][c][i] = iacc[2][g][c][i] = 0;
            f = (float)P;  // cvt.rn.f32.s64: the one rounding of the block partial
          }
          if (sx != nullptr) f = __fmul_rn(__fmul_rn(f, sxv[g][i >> 1]), swv[c][i & 1]);
          tot[g][c][i] = __fadd_rn(tot[g][c][i], f);
        }
  };

  // ldmatrix addresses of this lane inside a stage
  uint32_t aoff[C::PX][G], boff[C::PW][(WN + 1) / 2];
#pragma unroll
  for (int p = 0; p < C::PX; ++p)
#pragma unroll
    for (int g = 0; g < G; ++g)
      aoff[p][g] = sm32 + (p * C::BM + wm * 16 * G + g * 16 + (mat & 1) * 8 + r8) * kST + (mat >> 1) * 16;
#pragma unroll
  for (int p = 0; p < C::PW; ++p)
#pragma unroll
    for (int c = 0; c < WN; c += 2) {
      int cc = c + (mat >> 1);
      if (cc > WN - 1) cc = WN - 1;
      boff[p][c / 2] = sm32 + (C::PX * C::BM + p * C::BN + wn * 8 * WN + cc * 8 + r8) * kST + (mat & 1) * 16;
    }
  int stage = 0, istage = STAGES - 1;
  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1, istage);
    cp_async_commit();
    if (++istage == STAGES) istage = 0;
    const uint32_t st = stage * (C::ROWS * kST);
    if (++stage == STAGES) stage = 0;
#pragma unroll
    for (int s = 0; s < BKB / 32; ++s) {
      uint32_t b[C::PW][WN][2];
#pragma unroll
      for (int p = 0; p < C::PW; ++p)
#pragma unroll
        for (int c = 0; c < WN; c += 2) {
          uint32_t t4[4];
          ldmatrix_x4(t4, boff[p][c / 2] + st + s * 32);
          b[p][c][0] = t4[0];
          b[p][c][1] = t4[1];
          if (c + 1 < WN) {
            b[p][c + 1][0] = t4[2];
            b[p][c + 1][1] = t4[3];
          }
        }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        uint32_t a[C::PX][4];
#pragma unroll
        for (int p = 0; p < C::PX; ++p)
          ldmatrix_x4(a[p], aoff[p][g] + st + s * 32);
        // One product at a time over the warp's n8 tiles: consecutive MMAs hit different accumulators.
#define OVER_C(STMT)                                                                               \
  _Pragma("unroll") for (int c = 0; c < WN; ++c) { STMT; }
        if constexpr (MODE == 0) { OVER_C(mma_f(facc[g][c], a[0], b[0][c])) }
        if constexpr (MODE == 1) { OVER_C(mma_ss(iacc[0][g][c], a[0], b[0][c])) }
        if constexpr (MODE == 2) {
          OVER_C(mma_ss(iacc[0][g][c], a[0], b[0][c]))
          OVER_C(mma_us(iacc[1][g][c], a[1], b[0][c]))
        }
        if constexpr (MODE == 3) {
          OVER_C(mma_ss(iacc[0][g][c], a[0], b[0][c]))
          OVER_C(mma_ss(iacc[1][g][c], a[0], b[1][c]))
          OVER_C(mma_ss(iacc[1][g][c], a[1], b[0][c]))
        }
        if constexpr (MODE == 4) {
          OVER_C(mma_ss(iacc[0][g][c], a[0], b[0][c]))
          OVER_C(mma_su(iacc[1][g][c], a[0], b[1][c]))
          OVER_C(mma_uu(iacc[2][g][c], a[1], b[1][c]))
          OVER_C(mma_us(iacc[1][g][c], a[1], b[0][c]))
        }
        if constexpr (MODE == 5) {
          uint32_t ad[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) ad[q] = bsub(a[0][q], a[1][q]);
          OVER_C(mma_ss(iacc[0][g][c], a[0], b[0][c]))
          OVER_C(mma_ss(iacc[2][g][c], a[1], b[1][c]))
#pragma unroll
          for (int c = 0; c < WN; ++c) {
            uint32_t bd[2] = {bsub(b[0][c][0], b[1][c][0]), bsub(b[0][c][1], b[1][c][1])};
            mma_ss(iacc[1][g][c], ad, bd);
          }
        }
#undef OVER_C
      }
      if (++cnt == bsteps) {
        fold();
        cnt = 0;
        ++blk;
      }
    }
  }
  cp_async_wait<0>();

#pragma unroll
  for (int g = 0; g < G; ++g)
#pragma unroll
    for (int c = 0; c < WN; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row_base + g * 16 + (i >> 1) * 8;
        const int col = col_base + c * 8 + (i & 1);
        if (row < M) {
          if (out32)
            ((float*)y)[(int64_t)row * N + col] = tot[g][c][i];
          else
            ((uint16_t*)y)[(int64_t)row * N + col] = f2bf(tot[g][c][i]);
        }
      }
}

template <int MODE, int G, int WN, int WMW, int WNW, int STAGES, int BKB>
int launch(const uint8_t* xs, int64_t xplane, const uint8_t* ws, int64_t wplane, const float* sx,
           const float* sw, void* y, int M, int N, int RB, int bsteps, int out32) {
  typedef Cfg<MODE, G, WN, WMW, WNW, STAGES, BKB> C;
  if (N % C::BN || RB % BKB || (RB / 32) % bsteps) return -2;
  auto k = gx_kernel<MODE, G, WN, WMW, WNW, STAGES, BKB>;
  cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
  if (e != cudaSuccess) return (int)e;
  dim3 grid((M + C::BM - 1) / C::BM, N / C::BN);  // x = M tiles: neighbours share the W tile
  k<<<grid, C::THREADS, C::SMEM>>>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32);
  return (int)cudaGetLastError();
}

__device__ __forceinline__ int rne_shr(int mm, int sh) {  // mm < 256
  if (sh <= 0) return mm << (-sh);
  if (sh >= 9) return 0;
  int q = mm >> sh;
  const int rem = mm & ((1 << sh) - 1), half = 1 << (sh - 1);
  if (rem > half || (rem == half && (q & 1))) ++q;
  return q;
}
__device__ __forceinline__ float pow2f(int ex) {  // 2^ex, ex in [-149, 127]
  return __uint_as_float(ex >= -126 ? (uint32_t)(ex + 127) << 23 : 1u << (ex + 149));
}

// One CTA of 256 threads handles 2048 consecutive elements of one row; a thread handles 8.
__global__ void __launch_bounds__(256)
    slice_kernel(const uint16_t* __restrict__ x, int rows, int K, int B, int fmt, int s,
                 uint8_t* __restrict__ p0, uint8_t* __restrict__ p1, float* __restrict__ scale) {
  __shared__ int smax[256];
  const int tid = threadIdx.x, row = blockIdx.y;
  const int k0 = blockIdx.x * 2048 + tid * 8;
  const bool valid = k0 < K;
  uint16_t h[8];
  int e[8], mm[8];
  int emax = 1;
  if (valid) {
    const uint64_t* src = (const uint64_t*)(x + (int64_t)row * K + k0);
    const uint64_t q0 = src[0], q1 = src[1];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      h[i] = (uint16_t)((i < 4 ? q0 : q1) >> (16 * (i & 3)));
      e[i] = (h[i] >> 7) & 0xff;
      mm[i] = h[i] & 0x7f;
      if (e[i]) mm[i] |= 0x80; else e[i] = 1;
      if (mm[i] && e[i] > emax) emax = e[i];
    }
  }
  smax[tid] = emax;
  __syncthreads();
  if (!valid) return;
  const int per = B / 8, b0 = (tid / per) * per;
  int E = 1;
  for (int i = 0; i < per; ++i) E = max(E, smax[b0 + i]);
  if (fmt == 3) {
    const int Eq = max(E - s, 1);
    uint64_t o0 = 0, o1 = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint16_t o = h[i];
      if (e[i] < Eq) {
        const int r = rne_shr(mm[i], Eq - e[i]);
        const uint16_t sg = h[i] & 0x8000;
        if (r == 0) {
          o = sg;
        } else {
          const int p = 31 - __clz(r);
          const int f = Eq + p - 7;
          o = f >= 1 ? (uint16_t)(sg | (f << 7) | ((r << (7 - p)) & 0x7f)) : (uint16_t)(sg | (r << (Eq - 1)));
        }
      }
      if (i < 4) o0 |= (uint64_t)o << (16 * i); else o1 |= (uint64_t)o << (16 * (i - 4));
    }
    uint64_t* dst = (uint64_t*)(p0 + ((int64_t)row * K + k0) * 2);
    dst[0] = o0;
    dst[1] = o1;
    return;
  }
  uint64_t o0 = 0, o1 = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = E - e[i];
    int v, hi, lo;
    if (fmt == 0) {
      v = rne_shr(mm[i], d - 7);
      if (h[i] & 0x8000) v = -v;
      hi = v >> 8;
      lo = v & 0xff;
    } else if (fmt == 1) {
      v = rne_shr(mm[i], d - 6);
      if (h[i] & 0x8000) v = -v;
      lo = (int)(int8_t)(v & 0xff);
      hi = (v - lo) >> 8;
    } else if (fmt == 4) {
      v = rne_shr(mm[i], d - 5);
      if (h[i] & 0x8000) v = -v;
      lo = ((v + 64) & 127) - 64;
      hi = (v - lo) >> 7;
      if (hi - lo > 127) { hi = 63; lo = 64; }
    } else {
      v = min(rne_shr(mm[i], d + 1), 127);
      if (h[i] & 0x8000) v = -v;
      hi = v;
      lo = 0;
    }
    o0 |= (uint64_t)(hi & 0xff) << (8 * i);
    o1 |= (uint64_t)(lo & 0xff) << (8 * i);
  }
  *(uint64_t*)(p0 + (int64_t)row * K + k0) = o0;
  if (fmt != 2) *(uint64_t*)(p1 + (int64_t)row * K + k0) = o1;
  if (tid % per == 0)
    scale[(int64_t)row * (K / B) + k0 / B] = pow2f(E - (fmt == 0 ? 141 : fmt == 1 ? 140 : fmt == 4 ? 139 : 133));
}

// Register-only MMA rate probe: every warp runs iters x 8 independent MMAs (no memory traffic).
template <int TYPE>
__global__ void __launch_bounds__(256) rate_kernel(int iters, float* sink) {
  uint32_t a[4] = {0x3f803f80u + threadIdx.x % 2, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u};
  uint32_t b[2] = {0x3f803f80u, 0x3f803f80u};
  if (TYPE == 1) { a[0] = a[1] = a[2] = a[3] = 0x01010101u; b[0] = b[1] = 0x01010101u; }
  float fd[8][4];
  int id[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) { fd[j][i] = 0.f; id[j][i] = 0; }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if constexpr (TYPE == 0) mma_f(fd[j], a, b);
      if constexpr (TYPE == 1) mma_ss(id[j], a, b);
    }
  }
  float sacc = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) sacc += fd[j][i] + (float)id[j][i];
  if (threadIdx.x == 0) sink[blockIdx.x] = sacc;
}

}  // namespace

extern "C" {

int gx_rate(int type, int iters, int nblocks, float* sink) {
  if (type == 0) rate_kernel<0><<<nblocks, 256>>>(iters, sink);
  else rate_kernel<1><<<nblocks, 256>>>(iters, sink);
  return (int)cudaGetLastError();
}

int gx_slice(const uint16_t* x, int rows, int K, int B, int fmt, int s, uint8_t* p0, uint8_t* p1,
             float* scale) {
  if (K % 8 || B % 8 || 2048 % B) return -2;
  dim3 grid((K + 2047) / 2048, rows);
  slice_kernel<<<grid, 256>>>(x, rows, K, B, fmt, s, p0, p1, scale);
  return (int)cudaGetLastError();
}

// cfg 0: CTA 64 x 128 (8 warps, warp tile 32 x 32), 3 stages. cfg 2: CTA 64 x 64 (4 warps of 32 x 32).
// cfg 1: CTA 16 x 32 (4 warps of 16 x 8), 4 stages. cfg 3: cfg 0 with 128-byte k tiles, 2 stages.
// cfg 4: CTA 64 x 64, 8 warps of 16 x 32, 3 stages. cfg 5: CTA 64 x 128, 16 warps of 16 x 32, 2 stages.
int gx_gemm(int mode, int cfg, const uint8_t* xs, int64_t xplane, const uint8_t* ws, int64_t wplane,
            const float* sx, const float* sw, void* y, int M, int N, int RB, int bsteps, int out32) {
#define GO(MODE)                                                                                   \
  case MODE:                                                                                       \
    if (cfg == 0)                                                                                  \
      return launch<MODE, 2, 4, 2, 4, 3, 64>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32); \
    if (cfg == 2)                                                                                  \
      return launch<MODE, 2, 4, 2, 2, 3, 64>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32); \
    if (cfg == 3)                                                                                  \
      return launch<MODE, 2, 4, 2, 4, 2, 128>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32); \
    if (cfg == 4)                                                                                  \
      return launch<MODE, 1, 4, 4, 2, 3, 64>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32); \
    if (cfg == 5)                                                                                  \
      return launch<MODE, 1, 4, 4, 4, 2, 64>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32); \
    return launch<MODE, 1, 1, 1, 4, 4, 64>(xs, xplane, ws, wplane, sx, sw, y, M, N, RB, bsteps, out32);
  switch (mode) {
    GO(0) GO(1) GO(2) GO(3) GO(4) GO(5)
  }
  return -3;
}
}
