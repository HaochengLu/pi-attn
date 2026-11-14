// xcp.h -- "portable" GEMM mode: canonical program, exact CPU reference, deterministic test data.
// Host-only code (plain C++17, no libm-dependent results); shared by xcp.cu (GPU build) and xcp_cpu.cpp (any CPU).
//
// Canonical program XCP(prog, SEGB) for Y[M][N] = X[M][K] * W[N][K]^T, bf16 in, fp32 out, K % 64 == 0:
//   blocks   : K is cut into k16 blocks b = 0 .. K/16-1; (row, block) groups of X and (column, block) groups of W.
//   slicing  : XC slicing of round 2 (theory/xc.h): E = max exponent field of the group; group := 0 if E < 80;
//              slice0 = RNE(x / u0) u0 with u0 = 2^(E - s - 134); slice1 = RNE((x - slice0) / u1) u1, u1 = u0 2^-(8+s).
//   step     : for block b and slice pair (j, l): d_jl = sum_{k<16} Xj[m][16b+k] Wl[n][16b+k]   (exact: Theorem T1)
//   fold     : segments of SEGB blocks (default 32 = 512 k). Within a segment p = +0, blocks ascending:
//                TRI  : p = fl(p + d00); p = fl(p + d01); p = fl(p + d10)         (round-2 XC-triangular, s = 2, 2)
//                TRIF : p = fl(p + d00); p = fl(p + (d01 + d10))                  (d01 + d10 is exact: same unit,
//                       |sum| <= 2^24 units; on the GPU it is one tensor-core chain X0.W1 -> += X1.W0)
//                P21  : X sliced 2 (s = 0), W sliced 1 (s = 4): p = fl(p + d00); p = fl(p + d10)
//                P11  : X, W sliced 1 (s = 2, 2): p = fl(p + d00)
//              y = fl(...fl(fl(+0 + p_0) + p_1)... + p_{S-1}), segments ascending. fl = IEEE binary32 RNE add.
//   The value is a pure function of (row of X, row of W, K, prog, SEGB): independent of M, N, GPU class, scheduling.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>

#define XCP_FLUSH 80
struct XcpProg { int SA, SB, sA, sB, tri, fused; const char* name; };
// P21C / P11C: the same functions as P21 / P11, for weights that are already canonical (xcp_canon_weights: every
// element on its group's slice-0 lattice, stored as ordinary bf16, no extra bytes): the kernel skips W slicing.
static const XcpProg kXcpProgs[7] = {{2, 2, 2, 2, 1, 0, "TRI"}, {2, 2, 2, 2, 1, 1, "TRIF"}, {2, 1, 0, 4, 0, 0, "P21"},
                                     {1, 1, 2, 2, 0, 0, "P11"}, {0, 0, 0, 0, 0, 0, "NATIVE"}, {2, 1, 0, 4, 0, 0, "P21C"},
                                     {1, 1, 2, 2, 0, 0, "P11C"}};

// ------------------------------------------------------------------ exact reference ----
static inline uint32_t xcp_rne(uint32_t m, int sh) {  // round-to-nearest-even of m / 2^sh
  if (sh <= 0) return m << (-sh);
  if (sh > 30) return 0;
  const uint32_t q = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
  return q + ((rem > half) || (rem == half && (q & 1)));
}
// Slice one group of 16 bf16 (stride 1). Returns U0 (0: flushed group, all zero). slice0[k] = n0[k] 2^(U0-134),
// slice1[k] = n1[k] 2^(U0-8-s-134).
static inline int xcp_ref_slice(const uint16_t* x, int s, int16_t* n0, int16_t* n1) {
  int E = 0;
  for (int k = 0; k < 16; ++k) { const int Ef = (x[k] >> 7) & 255; E = Ef > E ? Ef : E; }
  if (E < XCP_FLUSH) { for (int k = 0; k < 16; ++k) n0[k] = n1[k] = 0; return 0; }
  const int U0 = E - s, U1 = U0 - 8 - s;
  for (int k = 0; k < 16; ++k) {
    const int Ef = (x[k] >> 7) & 255, neg = x[k] >> 15;
    if (Ef == 0) { n0[k] = n1[k] = 0; continue; }  // zeros and bf16 subnormals -> 0
    const int sig = 128 | (x[k] & 127), sh = U0 - Ef;
    const int a0 = (int)xcp_rne((uint32_t)sig, sh);
    const int r = sh <= 0 ? 0 : (sh > 30 ? sig : sig - (a0 << sh));  // residual, units of 2^(Ef-134), on the magnitude
    const int a1 = (int)xcp_rne((uint32_t)(r < 0 ? -r : r), U1 - Ef);
    n0[k] = (int16_t)(neg ? -a0 : a0);
    n1[k] = (int16_t)(((r < 0) != (neg != 0)) ? -a1 : a1);
  }
  return U0;
}
static std::atomic<long> g_xcp_violations{0};  // exact step sums that are not fp32 values (must stay 0)
static inline float xcp_mk(int64_t s, int e) {
  if (s > (1 << 24) || s < -(1 << 24) || e < -126 || e > 100) g_xcp_violations++;
  return std::ldexp((float)s, e);  // exact
}
static inline float xcp_add(float a, float b) { volatile float r = a + b; return r; }

struct XcpSliced {  // slices of rows [r0, r0 + nr) of a [*, K] bf16 matrix
  int nr, nb; std::vector<int16_t> n0, n1; std::vector<int> U;
};
static inline XcpSliced xcp_slice_rows(const uint16_t* a, int64_t ld, int nr, int K, int s) {
  XcpSliced o; o.nr = nr; o.nb = K / 16; o.n0.resize((size_t)nr * K); o.n1.resize((size_t)nr * K); o.U.resize((size_t)nr * o.nb);
  for (int i = 0; i < nr; ++i) for (int b = 0; b < o.nb; ++b)
    o.U[(size_t)i * o.nb + b] = xcp_ref_slice(a + i * ld + b * 16, s, &o.n0[(size_t)i * K + b * 16], &o.n1[(size_t)i * K + b * 16]);
  return o;
}
// Reference: y[M][N] fp32. Threads over output columns.
static inline void xcp_reference(const uint16_t* x, int64_t ldx, const uint16_t* w, int M, int N, int K, int prog, int segb, float* y, int threads = 8) {
  const XcpProg P = kXcpProgs[prog];
  const XcpSliced xs = xcp_slice_rows(x, ldx, M, K, P.sA);
  const int nb = K / 16, dA = 8 + P.sA, dB = 8 + P.sB;
  auto work = [&](int tid) {
    std::vector<int16_t> b0(K), b1(K); std::vector<int> UB(nb);
    for (int n = tid; n < N; n += threads) {
      for (int b = 0; b < nb; ++b) UB[b] = xcp_ref_slice(w + (int64_t)n * K + b * 16, P.sB, &b0[b * 16], &b1[b * 16]);
      for (int m = 0; m < M; ++m) {
        const int16_t *a0 = &xs.n0[(size_t)m * K], *a1 = &xs.n1[(size_t)m * K]; const int* UA = &xs.U[(size_t)m * nb];
        float tot = 0.f;
        for (int s0 = 0; s0 < nb; s0 += segb) {
          float p = 0.f;
          const int s1 = s0 + segb < nb ? s0 + segb : nb;
          for (int b = s0; b < s1; ++b) {
            if (!UA[b] || !UB[b]) continue;  // every product is zero: p + 0 = p (p is never -0)
            int32_t s00 = 0, s01 = 0, s10 = 0;
            for (int k = b * 16; k < b * 16 + 16; ++k) { s00 += (int32_t)a0[k] * b0[k]; s01 += (int32_t)a0[k] * b1[k]; s10 += (int32_t)a1[k] * b0[k]; }
            const int e00 = UA[b] + UB[b] - 268;
            p = xcp_add(p, xcp_mk(s00, e00));
            if (P.tri && !P.fused) { p = xcp_add(p, xcp_mk(s01, e00 - dB)); p = xcp_add(p, xcp_mk(s10, e00 - dA)); }
            else if (P.tri) { if (dA != dB) g_xcp_violations++; p = xcp_add(p, xcp_mk((int64_t)s01 + s10, e00 - dB)); }
            else if (P.SA == 2) p = xcp_add(p, xcp_mk(s10, e00 - dA));
          }
          tot = xcp_add(tot, p);
        }
        y[(size_t)m * N + n] = tot;
      }
    }
  };
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t) th.emplace_back(work, t);
  for (auto& t : th) t.join();
}
// Canonical weights: W := slice 0 of W for slice parameter s (12-bit lattice for s = 4, 10-bit for s = 2), in place.
// Idempotent; P21(X, W) == P21(X, canon(W, 4)) and P11(X, W) == P11(X, canon(W, 2)) bit for bit.
static inline uint16_t xcp_f2bf(float f);
static inline long xcp_canon_weights(uint16_t* w, int N, int K, int s, int threads = 8) {
  std::atomic<long> changed{0};
  auto work = [&](int tid) {
    long ch = 0;
    for (int n = tid; n < N; n += threads) for (int b = 0; b < K / 16; ++b) {
      uint16_t* g = w + (int64_t)n * K + b * 16; int16_t n0[16], n1[16];
      const int U0 = xcp_ref_slice(g, s, n0, n1);
      for (int k = 0; k < 16; ++k) { const uint16_t v = U0 ? xcp_f2bf(std::ldexp((float)n0[k], U0 - 134)) : 0; ch += v != g[k]; g[k] = v; }
    }
    changed += ch;
  };
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t) th.emplace_back(work, t);
  for (auto& t : th) t.join();
  return changed.load();
}
// fp64 product (accuracy reference)
static inline void xcp_exact64(const uint16_t* x, int64_t ldx, const uint16_t* w, int M, int N, int K, double* y, int threads = 8) {
  auto bfd = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return (double)f; };
  std::vector<double> xd((size_t)M * K);
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) xd[(size_t)m * K + k] = bfd(x[m * ldx + k]);
  auto work = [&](int tid) {
    std::vector<double> wd(K);
    for (int n = tid; n < N; n += threads) {
      for (int k = 0; k < K; ++k) wd[k] = bfd(w[(int64_t)n * K + k]);
      for (int m = 0; m < M; ++m) {
        long double s = 0; const double* xr = &xd[(size_t)m * K];
        for (int k = 0; k < K; ++k) s += (long double)(xr[k] * wd[k]);
        y[(size_t)m * N + n] = (double)s;
      }
    }
  };
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t) th.emplace_back(work, t);
  for (auto& t : th) t.join();
}

// ------------------------------------------- the floating-point slicing used on the GPU ----
// slice0 = (x + c) - c, c = 1.5 * 2^23 u0;  slice1 = ((x + c1) - c1) - slice0, c1 = c 2^-(8+s)   (IEEE RNE adds)
struct XcpConst { float c, c1; };
static inline XcpConst xcp_host_consts(float maxabs, int s) {
  uint32_t mb; memcpy(&mb, &maxabs, 4);
  const uint32_t eb = mb & 0x7F800000u;
  uint32_t cb = eb + (((uint32_t)(16 - s) << 23) | 0x400000u);
  if (eb < ((uint32_t)XCP_FLUSH << 23)) cb = 227u << 23;  // flushed group: c = 2^100 absorbs every element -> +0
  const uint32_t c1b = cb - ((uint32_t)(8 + s) << 23);
  XcpConst r; memcpy(&r.c, &cb, 4); memcpy(&r.c1, &c1b, 4); return r;
}
static inline void xcp_host_fp_slice(const uint16_t* x, int s, uint16_t* s0, uint16_t* s1) {
  float f[16], m = 0.f;
  for (int k = 0; k < 16; ++k) { const uint32_t u = (uint32_t)x[k] << 16; memcpy(&f[k], &u, 4); const float a = std::fabs(f[k]); m = a > m ? a : m; }
  const XcpConst C = xcp_host_consts(m, s);
  for (int k = 0; k < 16; ++k) {
    volatile float t = f[k] + C.c; volatile float a = t - C.c;
    volatile float t1 = f[k] + C.c1; volatile float y = t1 - C.c1; volatile float b = y - a;
    float af = a, bf = b; uint32_t ua, ub; memcpy(&ua, &af, 4); memcpy(&ub, &bf, 4);
    if ((ua & 0xFFFF) || (ub & 0xFFFF)) g_xcp_violations++;  // slices must be bf16 values
    s0[k] = (uint16_t)(ua >> 16); s1[k] = (uint16_t)(ub >> 16);
  }
}

// ------------------------------------------------------------------ deterministic data ----
static inline uint64_t xcp_mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static inline uint16_t xcp_f2bf(float f) {  // fp32 -> bf16, RNE (finite inputs)
  uint32_t u; memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
static inline float xcp_gauss(uint64_t h) {  // Irwin-Hall(12): integer sum, one exact scale
  int64_t s = 0;
  for (int i = 0; i < 3; ++i) { const uint64_t v = xcp_mix(h + 0x1234567ull * (i + 1)); s += (v & 0xFFFF) + ((v >> 16) & 0xFFFF) + ((v >> 32) & 0xFFFF) + (v >> 48); }
  return (float)(s - 6 * 65535) * (1.0f / 65536.0f);
}
// kind 0: Gaussian. 1: heavy tail (Gaussian * 2^[-6,6] for X, 2^[-3,3] for W). 2: wide exponents, uniform in
// [-20, 20] per element. 3: adversarial: exact zeros (+0 / -0), outliers * 2^12, groups of bf16 subnormals,
// flushed groups (~2^-70), cancelling pairs (x, -x against equal weights), zero rows.
static inline uint16_t xcp_gen(int is_w, int kind, uint64_t seed, int row, int k) {
  const uint64_t h = xcp_mix(seed * 0x100000001B3ull + ((uint64_t)is_w << 62) + ((uint64_t)row << 24) + (uint64_t)k);
  const float sc = is_w ? 0.02f : 1.0f;
  if (kind == 0) return xcp_f2bf(sc * xcp_gauss(h));
  if (kind == 1) { const int r = is_w ? 3 : 6; return xcp_f2bf(std::ldexp(sc * xcp_gauss(h), (int)((h >> 40) % (2 * r + 1)) - r)); }
  if (kind == 2) { const int e = (int)((h >> 20) % 41) - 20; return (uint16_t)(((h & 1) << 15) | ((uint64_t)(127 + e) << 7) | ((h >> 1) & 127)); }
  const uint64_t gh = xcp_mix(seed * 31 + ((uint64_t)is_w << 62) + ((uint64_t)row << 24) + (uint64_t)(k / 16));
  if (!is_w && row % 13 == 5) return 0;
  const int gk = (int)(gh % 16);
  if (gk == 0) return xcp_f2bf(std::ldexp(xcp_gauss(h), -70));                            // flushed group
  if (gk == 2 && (h >> 8) % 4 == 0) return (uint16_t)(((h & 1) << 15) | (1 + (h >> 1) % 127));  // bf16 subnormal
  int kk = k;
  const bool pair = is_w ? (xcp_mix(seed * 77 + (uint64_t)(k / 16)) % 4 == 1) : (xcp_mix(seed * 77 + (uint64_t)(k / 16)) % 4 == 1 && gk % 2 == 1);
  if (pair) kk = k & ~1;                                                                  // duplicated pair
  const uint64_t h2 = xcp_mix(seed * 0x100000001B3ull + ((uint64_t)is_w << 62) + ((uint64_t)row << 24) + (uint64_t)kk);
  const int p = (int)((h2 >> 12) % 100);
  if (p < 20) return (uint16_t)(((h2 >> 3) & 1) << 15);                                   // +0 / -0
  float v = sc * xcp_gauss(h2);
  if (p < 23) v = std::ldexp(v, 12);
  if (p >= 95) v = std::ldexp(v, -15);
  uint16_t b = xcp_f2bf(v);
  if (pair && !is_w && (k & 1)) b ^= 0x8000;                                              // x[k+1] = -x[k], w[k+1] = w[k]
  return b;
}
static inline void xcp_fill(std::vector<uint16_t>& v, int is_w, int kind, uint64_t seed, int rows, int K, int threads = 8) {
  v.resize((size_t)rows * K);
  auto work = [&](int tid) { for (int r = tid; r < rows; r += threads) for (int k = 0; k < K; ++k) v[(size_t)r * K + k] = xcp_gen(is_w, kind, seed, r, k); };
  std::vector<std::thread> th;
  for (int t = 0; t < threads; ++t) th.emplace_back(work, t);
  for (auto& t : th) t.join();
}
static inline uint64_t xcp_fnv(const void* p, size_t bytes) {
  uint64_t h = 1469598103934665603ull; const unsigned char* c = (const unsigned char*)p;
  for (size_t i = 0; i < bytes; ++i) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}
// verification cases: {N, K, data kind}; seed = 1000 + index
struct XcpCase { int N, K, kind; };
static const XcpCase kXcpCases[] = {{4096, 2048, 0}, {4096, 2048, 1}, {4096, 2048, 2}, {4096, 2048, 3}, {2048, 6144, 0}, {2048, 6144, 3},
                                    {12288, 2048, 1}, {6144, 4096, 2}, {64, 2560, 3}, {1024, 1088, 2}, {24576, 4096, 0}};
static const int kXcpNumCases = sizeof(kXcpCases) / sizeof(kXcpCases[0]);
static const int kXcpRefRows = 16;  // rows checked against the CPU reference
