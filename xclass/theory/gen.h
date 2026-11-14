// gen.h -- deterministic case generator shared by the GPU probe (mmaprobe.cu) and the CPU model fit.
// A case is one dot product with accumulator: d = c + sum_{k<16} a_k b_k  (a, b bf16; c, d fp32).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
struct Case { uint16_t a[16], b[16]; uint32_t c; };
namespace gen {
static uint64_t g_state;
static inline uint64_t rnd() {
  uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static inline int ri(int lo, int hi) { return lo + (int)(rnd() % (uint64_t)(hi - lo + 1)); }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) { uint32_t u = f2u(f); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static inline uint16_t bfb(int s, int e, int m7) { return (uint16_t)((s << 15) | ((127 + e) << 7) | (m7 & 127)); }
static inline uint32_t f32b(int s, int e, uint32_t m23) { return ((uint32_t)s << 31) | ((uint32_t)(127 + e) << 23) | (m23 & 0x7FFFFF); }
static inline float gauss() {
  double s = 0;
  for (int i = 0; i < 4; ++i) s += (double)(rnd() >> 32) / 4294967296.0;
  return (float)((s - 2.0) * 1.7320508075688772);
}
static inline int mant7() {  // varied mantissa shapes
  switch (ri(0, 5)) { case 0: return 0; case 1: return 127; case 2: return 1 << ri(0, 6); default: return (int)(rnd() & 127); }
}
static inline uint32_t mant23() {
  switch (ri(0, 6)) { case 0: return 0; case 1: return 0x7FFFFF; case 2: return 1u << ri(0, 22); case 3: return 0x7FFFFF ^ (1u << ri(0, 22)); default: return (uint32_t)(rnd() & 0x7FFFFF); }
}
// Explicitly sequenced helpers (third argument drawn first, then second, then first): the original calls relied on
// unspecified argument evaluation order; this is the order g++ (nvcc host compiler) used in run cal1.
static inline uint16_t bfb_R(int elo, int ehi) { const int m = (int)(rnd() & 127); const int e = ri(elo, ehi); const int s = (int)(rnd() & 1); return bfb(s, e, m); }
static inline uint16_t bfb_Rsub(int e0, int lo, int hi) { const int m = (int)(rnd() & 127); const int e = e0 - ri(lo, hi); const int s = (int)(rnd() & 1); return bfb(s, e, m); }
static inline uint32_t f32b_Rsub(int e0, int lo, int hi) { const uint32_t m = (uint32_t)(rnd() & 0x7FFFFF); const int e = e0 - ri(lo, hi); const int s = (int)(rnd() & 1); return f32b(s, e, m); }
static inline uint16_t bfb_Rfix(int e) { const int m = (int)(rnd() & 127); const int s = (int)(rnd() & 1); return bfb(s, e, m); }
static inline uint16_t bfb_RM(int elo, int ehi) { const int m = mant7(); const int e = ri(elo, ehi); const int s = (int)(rnd() & 1); return bfb(s, e, m); }
static inline uint16_t bfb_RMfix(int e) { const int m = mant7(); const int s = (int)(rnd() & 1); return bfb(s, e, m); }
static inline uint32_t f32b_R(int elo, int ehi) { const uint32_t m = (uint32_t)(rnd() & 0x7FFFFF); const int e = ri(elo, ehi); const int s = (int)(rnd() & 1); return f32b(s, e, m); }
static inline uint32_t f32b_RM(int elo, int ehi) { const uint32_t m = mant23(); const int e = ri(elo, ehi); const int s = (int)(rnd() & 1); return f32b(s, e, m); }
static inline uint32_t f32b_RMfix(int e) { const uint32_t m = mant23(); const int s = (int)(rnd() & 1); return f32b(s, e, m); }
static inline uint32_t f32b_sM(int s, int elo, int ehi) { const uint32_t m = mant23(); const int e = ri(elo, ehi); return f32b(s, e, m); }
static inline uint16_t bfb_sM(int s, int elo, int ehi) { const int m = mant7(); const int e = ri(elo, ehi); return bfb(s, e, m); }
// integer digit d (|d| <= 255) times 2^e as bf16 (exact)
static inline uint16_t digit_bf(int d, int e) {
  if (d == 0) return 0;
  int s = d < 0; unsigned m = s ? -d : d; int top = 31 - __builtin_clz(m);
  return bfb(s, e + top, (int)((m << (7 - top)) & 127));
}
// set ids: 1 gauss, 2 wide-exp, 3 cancel, 4 narrow, 5 sparse directed, 6 specials, 7 exact family,
//          8 exact-family boundary (may violate), 9 sliced-program steps
static inline void make(std::vector<Case>& cs, std::vector<uint8_t>& set, uint64_t seed, const int* counts) {
  g_state = seed;
  auto push = [&](const Case& c, int s) { cs.push_back(c); set.push_back((uint8_t)s); };
  for (int i = 0; i < counts[1]; ++i) {
    Case c; for (int k = 0; k < 16; ++k) { c.a[k] = f2bf(gauss()); c.b[k] = f2bf(0.02f * gauss()); }
    if (i & 3) { const float sc = 0.08f * (float)ri(0, 6); c.c = f2u(sc * gauss()); } else c.c = 0; push(c, 1);
  }
  for (int i = 0; i < counts[2]; ++i) {
    Case c; for (int k = 0; k < 16; ++k) { c.a[k] = bfb_R(-10, 10); c.b[k] = bfb_R(-10, 10); }
    c.c = f32b_R(-10, 10); push(c, 2);
  }
  for (int i = 0; i < counts[3]; ++i) {
    Case c; const int e1 = ri(-4, 4), e2 = ri(-4, 4), k1 = ri(0, 15); int k2 = ri(0, 14); if (k2 >= k1) ++k2;
    for (int k = 0; k < 16; ++k) { c.a[k] = bfb_Rsub(e1, 5, 14); c.b[k] = bfb_Rsub(e2, 5, 14); }
    const int s = rnd() & 1, m = rnd() & 127; c.a[k1] = bfb(s, e1, m); c.a[k2] = bfb(s ^ 1, e1, m);
    c.b[k1] = c.b[k2] = bfb_Rfix(e2);
    c.c = (i & 1) ? f32b_Rsub(e1 + e2, 8, 26) : 0; push(c, 3);
  }
  for (int i = 0; i < counts[4]; ++i) {
    Case c; for (int k = 0; k < 16; ++k) { c.a[k] = bfb_RM(-1, 1); c.b[k] = bfb_RM(-1, 1); }
    c.c = f32b_RM(-3, 8); push(c, 4);
  }
  for (int i = 0; i < counts[5]; ++i) {
    Case c; memset(&c, 0, sizeof c); const int nz = ri(1, 4), ec = ri(-3, 3);
    for (int j = 0; j < nz; ++j) {
      const int k = ri(0, 15), ep = ec + ri(-40, 12), ea = ri(-6, 6) + ep / 2, eb = ep - ea;
      c.a[k] = bfb_RMfix(ea); c.b[k] = bfb_RMfix(eb);
    }
    c.c = ri(0, 7) ? f32b_RMfix(ec) : (uint32_t)(rnd() & 1) << 31; push(c, 5);
  }
  for (int i = 0; i < counts[6]; ++i) {
    Case c; memset(&c, 0, sizeof c); const int nz = ri(1, 3);
    auto sp = [&]() -> uint16_t {
      const int s = rnd() & 1;
      switch (ri(0, 9)) {
        case 0: return (uint16_t)(s << 15); case 1: return (uint16_t)((s << 15) | 0x7F80);
        case 2: return (uint16_t)((s << 15) | 0x7FC0 | (rnd() & 63)); case 3: return (uint16_t)((s << 15) | 1);
        case 4: return (uint16_t)((s << 15) | (rnd() & 127)); case 5: return (uint16_t)((s << 15) | 0x7F7F);
        case 6: return (uint16_t)((s << 15) | 0x0080 | (rnd() & 127)); case 7: return bfb_sM(s, 50, 127);
        case 8: return bfb_sM(s, -126, -50); default: return bfb_sM(s, -8, 8);
      }
    };
    for (int j = 0; j < nz; ++j) { const int k = ri(0, 15); c.a[k] = sp(); c.b[k] = sp(); }
    const int s = rnd() & 1;
    switch (ri(0, 8)) {
      case 0: c.c = (uint32_t)s << 31; break; case 1: c.c = ((uint32_t)s << 31) | 0x7F800000u; break;
      case 2: c.c = ((uint32_t)s << 31) | 0x7FC00000u | (rnd() & 0xFFFF); break; case 3: c.c = ((uint32_t)s << 31) | (rnd() & 0x7FFFFF); break;
      case 4: c.c = ((uint32_t)s << 31) | 0x7F7FFFFFu; break; case 5: c.c = f32b_sM(s, 60, 127); break;
      case 6: c.c = f32b_sM(s, -126, -60); break; default: c.c = f32b_sM(s, -8, 8); break;
    }
    push(c, 6);
  }
  // 7: exact family. digits |da| <= 2^wa - 1, |db| <= 2^wb - 1, common exponents EA, EB;
  //    c = integer multiple of 2^(EA+EB) with |c| + sum |p| <= 2^24.
  // 8: same but |c| up to 2^25 units and random (partial sums / total may leave the exact range).
  for (int s78 = 7; s78 <= 8; ++s78)
    for (int i = 0; i < counts[s78]; ++i) {
      Case c; const int wa = ri(1, 8), wb = ri(1, 8), EA = ri(-20, 20), EB = ri(-20, 20);
      const int ma = (1 << wa) - 1, mb = (1 << wb) - 1; long long sabs = 0;
      const int dens = ri(0, 3);
      for (int k = 0; k < 16; ++k) {
        int da = ri(-ma, ma), db = ri(-mb, mb);
        if (dens == 0 && ri(0, 3)) da = 0;
        if (dens == 1 && ri(0, 1)) { da = (rnd() & 1) ? ma : -ma; db = (rnd() & 1) ? mb : -mb; }
        c.a[k] = digit_bf(da, EA); c.b[k] = digit_bf(db, EB); sabs += (long long)(da < 0 ? -da : da) * (db < 0 ? -db : db);
      }
      long long lim = (s78 == 7) ? (1ll << 24) - sabs : (1ll << 25);
      long long cv = 0;
      if (lim > 0) { switch (ri(0, 3)) { case 0: cv = 0; break; case 1: cv = (rnd() & 1) ? lim : -lim; break; default: cv = (long long)(rnd() % (uint64_t)(2 * lim + 1)) - lim; } }
      if (s78 == 8 && cv != 0) { while ((cv < 0 ? -cv : cv) >= (1ll << 24) && (cv & 1)) cv /= 2; }  // keep c representable
      float cf = (float)cv; // exact when |cv| <= 2^24 or even-enough; then scale by 2^(EA+EB) exactly
      uint32_t u = f2u(cf); if (cv != 0) u += (uint32_t)(EA + EB) << 23; c.c = u; push(c, s78);
    }

  // ---- round-2 sets (appended; earlier streams unchanged) ----
  // lattice operand: integer magnitude with <= 8 significant bits, < 2^(8+s) (or == 2^(8+s) when top), times 2^E
  auto lat = [&](int s, int E, int kind) -> uint16_t {
    int sg = (int)(rnd() & 1); int sh = ri(0, s); int sig = 128 | (int)(rnd() & 127);
    if (kind == 1) { sig = 255; sh = s; }            // largest 8-bit value on the lattice
    if (kind == 2) { sig = 128; sh = s + 1; }        // exactly 2^(8+s)
    if (kind == 3 && ri(0, 2) == 0) return 0;
    if (kind == 4) { sig = ri(1, 255); sh = ri(0, s); int top = 31 - __builtin_clz((unsigned)sig); return bfb(sg, E + sh + top, (sig << (7 - top)) & 127); }
    return bfb(sg, E + sh + 7, sig & 127);
  };
  auto latval = [&](uint16_t v, int E) -> long long {  // integer value in units of 2^E
    if (!(v & 0x7FFF)) return 0; int e = ((v >> 7) & 255) - 127 - 7 - E; long long m = 128 | (v & 127); m = e >= 0 ? m << e : m >> (-e); return (v >> 15) ? -m : m;
  };
  auto cfrom = [&](long long cv, int E) -> uint32_t { if (cv == 0) return 0; uint32_t u = f2u((float)cv); return u + ((uint32_t)E << 23); };
  // 9: T1 holds (sA + sB <= 4, |c| + sum |p| <= 2^24 units).  10: T1 violated slightly.
  for (int s910 = 9; s910 <= 10; ++s910)
    for (int i = 0; i < counts[s910]; ++i) {
      Case c; int sA = ri(0, 4), sB = 4 - sA; const int EA = ri(-20, 20), EB = ri(-20, 20); const int kind = ri(0, 4), same = ri(0, 2) == 0;
      int viol = 0;
      if (s910 == 10) { viol = ri(1, 3); if (viol == 1) { if (rnd() & 1) sA += ri(1, 3); else sB += ri(1, 3); } }
      long long sabs = 0, ssum = 0;
      for (int k = 0; k < 16; ++k) {
        c.a[k] = lat(sA, EA, kind); c.b[k] = lat(sB, EB, kind);
        if (same) { c.a[k] &= 0x7FFF; c.b[k] &= 0x7FFF; }
        const long long p = latval(c.a[k], EA) * latval(c.b[k], EB); sabs += p < 0 ? -p : p; ssum += p;
      }
      long long lim = (1ll << 24) - sabs, cv = 0;
      if (viol == 2) lim = (1ll << 24) + ri(1, 1 << 20);                       // c + products may leave [-2^24, 2^24]
      if (viol == 3) lim = (1ll << 24) - (ssum < 0 ? -ssum : ssum);            // total fits, partial sums may not
      if (lim > 0) { const int m = ri(0, 3); cv = m == 0 ? 0 : m == 1 ? lim : m == 2 ? -lim : (long long)(rnd() % (uint64_t)(2 * lim + 1)) - lim; }
      if (viol == 3 && lim > 0 && ri(0, 1)) cv = ssum < 0 ? lim : -lim;
      while ((cv < 0 ? -cv : cv) > (1ll << 24)) cv /= 2;
      c.c = cfrom(cv, EA + EB); push(c, s910);
    }
  // 11: T2 (single-rounding) family: c = 0, only k < 8 used, product exponents within a 24-bit window (sA + sB <= 10)
  // 12: same but all 16 slots used (class A rounds the intermediate)
  for (int s1112 = 11; s1112 <= 12; ++s1112)
    for (int i = 0; i < counts[s1112]; ++i) {
      Case c; memset(&c, 0, sizeof c); const int sA = ri(0, 10), sB = 10 - sA, EA = ri(-20, 20), EB = ri(-20, 20), kind = ri(0, 4) == 1 ? 1 : (ri(0, 1) ? 0 : 4);
      for (int k = 0; k < (s1112 == 11 ? 8 : 16); ++k) { c.a[k] = lat(sA, EA, kind); c.b[k] = lat(sB, EB, kind); }
      push(c, s1112);
    }
  // 13: signed zeros: every product is +-0 (or all -0), c = +-0
  for (int i = 0; i < counts[13]; ++i) {
    Case c; const int allneg = ri(0, 1), cneg = ri(0, 1);
    for (int k = 0; k < 16; ++k) {
      const int za = ri(0, 1); const uint16_t z = (uint16_t)((allneg ? 0 : (rnd() & 1)) << 15), x = bfb(allneg ? 1 : (int)(rnd() & 1), ri(-5, 5), mant7());
      c.a[k] = za ? z : x; c.b[k] = za ? (uint16_t)(x & 0x7FFF) : (uint16_t)(z ^ (allneg ? 0x8000 : 0));
      if (allneg && !za) { c.a[k] = x; c.b[k] = 0; }   // (-x) * (+0) = -0
    }
    c.c = (uint32_t)cneg << 31; push(c, 13);
  }
}
}  // namespace gen
