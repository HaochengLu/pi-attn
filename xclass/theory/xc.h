// xc.h -- XC ("cross-class") canonical k16 program: shared host/device definition of the pre-transform, plus the
// CPU reference (host only). The same slicing code is compiled for GPU and CPU; xc_ref.py is an independent
// re-implementation of the whole program in exact integer arithmetic.
//
// Program XC(SA, SB, sA, sB), sA + sB <= 4, for C = A * B with A [M][K] bf16, B [K][N] bf16, K % 16 == 0:
//   1. Blocks. A is cut into (row, k16-block) groups of 16 elements, B into (column, k16-block) groups.
//   2. Slicing (per group, integer ops only). E = max exponent field in the group. If E < XC_FLUSH the group is
//      zero. U0 = E - s is the unit field of slice 0: every element is rounded (nearest, ties to even, on the
//      magnitude) to a multiple of 2^(U0 - 134); elements with exponent field >= U0 are unchanged. Slice 1 (if
//      used) is the exact residual rounded the same way to unit field U1 = U0 - 8 - s. Each slice element is a
//      bf16 value n * 2^(U - 134) with integer |n| <= 2^(8 + s).
//   3. Step. For block t and slice pair (j, l): d = sum_{k<16} A_j[i][16t+k] * B_l[16t+k][n], computed by ONE
//      tensor-core MMA with a zero accumulator. All 16 products are multiples of one unit u and
//      sum |products| <= 16 * 2^(16 + sA + sB) u <= 2^24 u, so the fused sum is exact on every tensor-core class.
//   4. Fold. acc = fl32_RNE(acc + d), in the order t ascending, then j ascending, then l ascending; acc starts +0.
//      "Triangular" variant: the pair (j, l) = (1, 1) is skipped (3 MMAs per k16 instead of 4).
// Program XR(sA, sB), sA + sB <= 10 (single-rounding family, 2 MMAs per k16): groups of 8 elements, slice 0 only;
//   step d = RZ24(exact sum of the 8 products) from one MMA whose other k8 half is zero; same fold. Relies on both
//   classes rounding the final fused sum toward zero (measured), not only on exactness.
// Domain: finite inputs with |x| < 2^50 (larger: outside the ABI). Groups whose largest magnitude is below
// 2^(XC_FLUSH-127) are flushed to zero (declared), which keeps every product unit above the fp32 subnormal range.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define XC_HD __host__ __device__ __forceinline__
#else
#define XC_HD static inline
#endif
#define XC_FLUSH 80
#define XC_MAXE 177

// encode sign * n * 2^(U - 134), 0 <= n <= 255 (or a power of two up to 256), as bf16. n == 0 -> +0.
XC_HD uint16_t xc_enc(int sign, int n, int U) {
  if (n == 0) return 0;
  int top = 0;
  while ((n >> top) > 1) ++top;
  const int sig = top <= 7 ? (n << (7 - top)) : (n >> (top - 7));
  return (uint16_t)((sign << 15) | ((U - 7 + top) << 7) | (sig & 127));
}
// round finite bf16 x to a multiple of 2^(U - 134) (RNE on the magnitude); *res = exact residual x - y as bf16
XC_HD uint16_t xc_round(uint16_t x, int U, uint16_t* res) {
  const int Ef = (x >> 7) & 255, sign = x >> 15, sig = 128 | (x & 127);
  *res = 0;
  if (Ef == 0) return 0;           // zeros and bf16 subnormals -> +0
  if (Ef >= U) return x;
  const int sh = U - Ef;
  int n = 0;
  if (sh <= 8) { const int q = sig >> sh, rem = sig & ((1 << sh) - 1), half = 1 << (sh - 1); n = q + ((rem > half) || (rem == half && (q & 1))); }
  const int r = sig - (sh <= 8 ? (n << sh) : 0);     // residual in units of 2^(Ef - 134), |r| <= 255
  *res = xc_enc(r < 0 ? sign ^ 1 : sign, r < 0 ? -r : r, Ef);
  return xc_enc(sign, n, U);
}
// slice one group of G (16, or 8 for the XR family) elements into s0 (and s1). Returns 1 if outside the ABI domain.
XC_HD int xc_slice(const uint16_t* x, int G, int s, uint16_t* s0, uint16_t* s1) {
  int E = 0;
  for (int k = 0; k < G; ++k) { const int Ef = (x[k] >> 7) & 255; E = Ef > E ? Ef : E; }
  if (E < XC_FLUSH) { for (int k = 0; k < G; ++k) { s0[k] = 0; s1[k] = 0; } return 0; }
  const int U0 = E - s, U1 = U0 - 8 - s;
  for (int k = 0; k < G; ++k) { uint16_t r, r2; s0[k] = xc_round(x[k], U0, &r); s1[k] = xc_round(r, U1, &r2); }
  return E >= XC_MAXE;
}

#ifndef __CUDACC__
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// exact dot of n bf16 pairs. exact_required: abort unless the exact sum is an fp32 value (XC theorem); otherwise
// return the exact sum rounded toward zero to 24 bits (XR family).
static inline float xc_dot(const uint16_t* a, const uint16_t* b, int n, bool exact_required) {
  int lo = 1 << 30; long long sig[16]; int ex[16];
  for (int k = 0; k < n; ++k) {
    const uint16_t x = a[k], y = b[k];
    if (!(x & 0x7FFF) || !(y & 0x7FFF)) { sig[k] = 0; continue; }
    sig[k] = (long long)(128 | (x & 127)) * (128 | (y & 127)); if ((x ^ y) >> 15) sig[k] = -sig[k];
    ex[k] = ((x >> 7) & 255) + ((y >> 7) & 255) - 268; if (ex[k] < lo) lo = ex[k];
  }
  long long S = 0;
  for (int k = 0; k < n; ++k) if (sig[k]) { if (ex[k] - lo > 40) { fprintf(stderr, "xc: exponent spread\n"); abort(); } S += sig[k] << (ex[k] - lo); }
  if (S == 0) return 0.f;
  long long m = S < 0 ? -S : S; int e = lo;
  if (!exact_required) while (m >= (1ll << 24)) { m >>= 1; ++e; }      // truncate toward zero to 24 bits
  else { while (!(m & 1)) { m >>= 1; ++e; } if (m >= (1ll << 24)) { fprintf(stderr, "xc: exact sum not an fp32 value (theorem violated)\n"); abort(); } }
  if (e < -126 || e > 100) { fprintf(stderr, "xc: exponent out of ABI range\n"); abort(); }
  return (float)std::ldexp((double)(S < 0 ? -m : m), e);
}
// reference GEMM for the probe layout: X [T][64][16] (row-major per k16 block), W [T][8][16]; out [64][8].
// mode 0: XC, all SA x SB slice pairs. mode 1: XC triangular (pair (1,1) skipped). mode 2: XR, k8 groups, slice 0 only.
static inline void xc_reference_tile(const uint16_t* X, const uint16_t* W, int T, int SA, int SB, int sA, int sB, int mode, float* out) {
  static thread_local uint16_t xs[2][64][16], ws[2][8][16];
  const int G = mode == 2 ? 8 : 16;
  float acc[64][8]; memset(acc, 0, sizeof acc);
  for (int t = 0; t < T; ++t) {
    for (int r = 0; r < 64; ++r) for (int h = 0; h < 16; h += G) if (xc_slice(X + (size_t)t * 1024 + r * 16 + h, G, sA, xs[0][r] + h, xs[1][r] + h)) { fprintf(stderr, "xc: out of domain\n"); abort(); }
    for (int n = 0; n < 8; ++n) for (int h = 0; h < 16; h += G) if (xc_slice(W + (size_t)t * 128 + n * 16 + h, G, sB, ws[0][n] + h, ws[1][n] + h)) { fprintf(stderr, "xc: out of domain\n"); abort(); }
    for (int r = 0; r < 64; ++r) for (int n = 0; n < 8; ++n) {
      if (mode == 2) { for (int h = 0; h < 16; h += 8) { volatile float d = xc_dot(xs[0][r] + h, ws[0][n] + h, 8, false); volatile float s = acc[r][n] + d; acc[r][n] = s; } continue; }
      for (int j = 0; j < SA; ++j) for (int l = 0; l < SB; ++l) {
        if (mode == 1 && j == 1 && l == 1) continue;
        volatile float d = xc_dot(xs[j][r], ws[l][n], 16, true);
        volatile float s = acc[r][n] + d;   // IEEE binary32 RNE add
        acc[r][n] = s;
      }
    }
  }
  memcpy(out, acc, sizeof acc);
}
#endif
