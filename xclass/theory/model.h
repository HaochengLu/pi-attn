// model.h -- parametrised software model of one fused tensor-core group: d = c + sum_i a_i b_i  (bf16, bf16 -> fp32)
#pragma once
#include <cstdint>
#include <cstring>
typedef __int128 i128;
#ifndef ZERO_SIGN_KEPT
#define ZERO_SIGN_KEPT 0
#endif
struct MP {
  int F;       // alignment keeps bits down to 2^(emax - F)
  int amode;   // 0: truncate magnitude (toward zero), 1: floor (two's complement, toward -inf)
  int fmode;   // final rounding to fp32: 0 RZ, 1 RNE, 2 floor (toward -inf), 3 away
  int pnorm;   // 1: product exponent is the normalised one (ea+eb+1 when ma*mb >= 2), 0: ea+eb
  int group;   // 8 (two chained groups, k 0..7 then 8..15) or 16
};
static inline int msb128(i128 v) { int n = -1; while (v) { ++n; v >>= 1; } return n; }
// round sign * mag * 2^lsb to fp32 bits (finite path; overflow -> inf; subnormals by fmode at 2^-149)
static inline uint32_t round_f32(int sign, i128 mag, int lsb, int fmode) {
  if (mag == 0) return (uint32_t)sign << 31;
  int top = msb128(mag); int e = top + lsb;        // value in [2^e, 2^(e+1))
  int q = e - 23; if (q < -149) q = -149;           // target lsb weight
  int sh = q - lsb; i128 m;
  if (sh <= 0) m = mag << (-sh);
  else if (sh > 100) { m = 0; if ((fmode == 2 && sign) || fmode == 3 || (fmode == 4 && !sign)) m = 1; }
  else {
    i128 rem = mag & (((i128)1 << sh) - 1), half = (i128)1 << (sh - 1); m = mag >> sh;
    bool up = false;
    if (fmode == 1) up = rem > half || (rem == half && (m & 1));
    else if (fmode == 2) up = sign && rem != 0;
    else if (fmode == 3) up = rem != 0;
    else if (fmode == 4) up = !sign && rem != 0;
    if (up) ++m;
  }
  if (m >> 24) { m >>= 1; ++q; }
  int E = q + 23 + 127;
  if (m == 0) return ZERO_SIGN_KEPT ? (uint32_t)sign << 31 : 0;
  if (!(m >> 23)) return ((uint32_t)sign << 31) | (uint32_t)m;      // subnormal (q == -149)
  if (E >= 255) return ((uint32_t)sign << 31) | 0x7F800000u;
  return ((uint32_t)sign << 31) | ((uint32_t)E << 23) | ((uint32_t)m & 0x7FFFFF);
}
struct Term { int s; int64_t m; int lsb; int e; bool nz; };
static inline Term term_c(uint32_t c) {
  Term t; t.s = c >> 31; int E = (c >> 23) & 255; uint32_t f = c & 0x7FFFFF;
  if (E == 0) { t.m = f; t.lsb = -149; t.e = -126; } else { t.m = f | 0x800000; t.lsb = E - 150; t.e = E - 127; }
  t.nz = t.m != 0; return t;
}
static inline Term term_p(uint16_t a, uint16_t b, int pnorm) {
  Term t; t.s = (a >> 15) ^ (b >> 15); int Ea = (a >> 7) & 255, Eb = (b >> 7) & 255; int ma = a & 127, mb = b & 127;
  int ea, eb;
  if (Ea == 0) ea = -126; else { ma |= 128; ea = Ea - 127; }
  if (Eb == 0) eb = -126; else { mb |= 128; eb = Eb - 127; }
  t.m = (int64_t)ma * mb; t.lsb = ea + eb - 14; t.e = ea + eb + ((pnorm && t.m >= 32768) ? 1 : 0); t.nz = t.m != 0; return t;
}
static inline bool is_special16(uint16_t a) { return ((a >> 7) & 255) == 255; }
// one fused group over products idx[0..n)
static inline uint32_t group_step(const MP& p, uint32_t c, const uint16_t* a, const uint16_t* b, int k0, int n) {
  Term t[17]; int nt = 0; t[nt++] = term_c(c);
  bool nan = false; int inf = 0;  // inf: bit0 +inf seen, bit1 -inf seen
  if (((c >> 23) & 255) == 255) { if (c & 0x7FFFFF) nan = true; else inf |= (c >> 31) ? 2 : 1; t[0].nz = false; t[0].m = 0; }
  for (int k = k0; k < k0 + n; ++k) {
    const bool sa = is_special16(a[k]), sb = is_special16(b[k]);
    if (sa || sb) {
      const bool nana = sa && (a[k] & 127), nanb = sb && (b[k] & 127), za = !(a[k] & 0x7FFF), zb = !(b[k] & 0x7FFF);
      if (nana || nanb || za || zb) nan = true; else inf |= (((a[k] ^ b[k]) >> 15) & 1) ? 2 : 1;
      continue;
    }
    t[nt++] = term_p(a[k], b[k], p.pnorm);
  }
  if (nan || inf == 3) return 0x7FFFFFFFu;
  if (inf) return inf == 1 ? 0x7F800000u : 0xFF800000u;
  int emax = -100000; bool any = false;
  for (int i = 0; i < nt; ++i) if (t[i].nz) { any = true; if (t[i].e > emax) emax = t[i].e; }
  if (!any) { bool allneg = true; for (int i = 0; i < nt; ++i) allneg = allneg && t[i].s; return allneg ? 0x80000000u : 0; }
  const int unit = emax - p.F; i128 S = 0;
  for (int i = 0; i < nt; ++i) if (t[i].nz) {
    int sh = t[i].lsb - unit; i128 v;
    if (sh >= 0) v = (i128)t[i].m << sh;
    else if (-sh > 62) v = (p.amode == 1 && t[i].s) ? 1 : 0;
    else { v = t[i].m >> (-sh); if (p.amode == 1 && t[i].s && (t[i].m & (((int64_t)1 << (-sh)) - 1))) ++v; }
    S += t[i].s ? -v : v;
  }
  if (S == 0) return 0;
  return round_f32(S < 0, S < 0 ? -S : S, unit, p.fmode);
}
static inline uint32_t mma_k16(const MP& p, uint32_t c, const uint16_t* a, const uint16_t* b) {
  if (p.group == 16) return group_step(p, c, a, b, 0, 16);
  return group_step(p, group_step(p, c, a, b, 0, 8), a, b, 8, 8);
}
