#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "model.h"
static int ARGS_RTL = 1, OR_RTL = 0;
static uint64_t rnd(); static uint16_t bf_bits(int sign, int e, int m7);
template <class F> static uint16_t bf3_(F ef) { int s, e, m; if (ARGS_RTL) { m = (int)(rnd() & 127); e = ef(); s = rnd() & 1; } else { s = rnd() & 1; e = ef(); m = (int)(rnd() & 127); } return bf_bits(s, e, m); }
template <class F> static uint32_t or3_(F ef) { uint32_t s, e, m; if (OR_RTL) { m = rnd() & 0x7FFFFF; e = ef(); s = rnd() & 1; } else { s = rnd() & 1; e = ef(); m = rnd() & 0x7FFFFF; } return (s << 31) | (e << 23) | m; }
#define BF3(E) bf3_([&]() { return (int)(E); })
#define OR3(E) or3_([&]() { return (uint32_t)(E); })
// ------------------------------------------------------------------ host ----
static uint64_t g_state;
static uint64_t rnd() {  // splitmix64: identical stream on every machine
  uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static int rint_in(int lo, int hi) { return lo + (int)(rnd() % (uint64_t)(hi - lo + 1)); }
static uint16_t f2bf(float f) {  // round to nearest even
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static uint16_t bf_bits(int sign, int e, int m7) {  // (-1)^sign * 2^e * (1 + m7 / 128)
  return (uint16_t)((sign << 15) | ((127 + e) << 7) | (m7 & 127));
}
static float gauss() {
  double s = 0;
  for (int i = 0; i < 4; ++i) s += (double)(rnd() >> 32) / 4294967296.0;
  return (float)((s - 2.0) * 1.7320508075688772);
}
static uint64_t fnv(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  const unsigned char* p = (const unsigned char*)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
static size_t mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += memcmp(&a[i], &b[i], 4) != 0;
  return n;
}

struct Data {
  int tiles, T;
  std::vector<uint16_t> X, W;
  std::vector<float> C;
};

static Data make(int which) {
  Data d;
  g_state = 0x1234 + which;
  const int tiles[4] = {256, 512, 4096, 4096}, Ts[4] = {8, 64, 1, 1};
  d.tiles = tiles[which];
  d.T = Ts[which];
  d.X.resize((size_t)d.tiles * d.T * 1024);
  d.W.resize((size_t)d.tiles * d.T * 128);
  d.C.assign((size_t)d.tiles * 512, 0.f);
  if (which == 0) {
    for (auto& v : d.X) v = f2bf((float)rint_in(-8, 8));
    for (auto& v : d.W) v = f2bf((float)rint_in(-8, 8));
    for (auto& v : d.C) v = (float)rint_in(-1000, 1000);
  } else if (which == 1) {
    for (auto& v : d.X) v = f2bf(gauss());
    for (auto& v : d.W) v = f2bf(0.02f * gauss());
  } else if (which == 2) {
    for (auto& v : d.X) v = BF3(rint_in(-10, 10));
    for (auto& v : d.W) v = BF3(rint_in(-10, 10));
    for (auto& v : d.C) {
      const uint32_t u = OR3(127 + rint_in(-10, 10));
      memcpy(&v, &u, 4);
    }
  } else {
    for (int t = 0; t < d.tiles; ++t) {
      uint16_t* x = &d.X[(size_t)t * 1024];
      uint16_t* w = &d.W[(size_t)t * 128];
      const int e1 = rint_in(-4, 4), e2 = rint_in(-4, 4);
      const int k1 = rint_in(0, 15);
      int k2 = rint_in(0, 14);
      if (k2 >= k1) ++k2;
      for (int r = 0; r < 64; ++r)
        for (int k = 0; k < 16; ++k)
          x[r * 16 + k] = BF3(e1 - rint_in(5, 14));
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k)
          w[n * 16 + k] = BF3(e2 - rint_in(5, 14));
      for (int r = 0; r < 64; ++r) {  // a[k2] b[k2] = -a[k1] b[k1] exactly, for every (r, n)
        const int s = rnd() & 1, m = (int)(rnd() & 127);
        x[r * 16 + k1] = bf_bits(s, e1, m);
        x[r * 16 + k2] = bf_bits(s ^ 1, e1, m);
      }
      for (int n = 0; n < 8; ++n) {
        const uint16_t b = BF3(e2);
        w[n * 16 + k1] = b;
        w[n * 16 + k2] = b;
      }
      if (t & 1)
        for (int i = 0; i < 512; ++i) {
          const uint32_t u = OR3(127 + e1 + e2 - rint_in(8, 26));
          memcpy(&d.C[(size_t)t * 512 + i], &u, 4);
        }
    }
  }
  return d;
}


static uint64_t run(const Data& d, const MP& p) {
  const size_t nout = (size_t)d.tiles * 512; std::vector<float> out(nout);
  for (int t = 0; t < d.tiles; ++t) for (int r = 0; r < 64; ++r) for (int n = 0; n < 8; ++n) {
    uint32_t c; memcpy(&c, &d.C[(size_t)t * 512 + r * 8 + n], 4);
    for (int q = 0; q < d.T; ++q) c = mma_k16(p, c, &d.X[((size_t)t * d.T + q) * 1024 + r * 16], &d.W[((size_t)t * d.T + q) * 128 + n * 16]);
    memcpy(&out[(size_t)t * 512 + r * 8 + n], &c, 4);
  }
  return fnv(out);
}
int main() {
  const char* nm[4] = {"D0", "D1", "D2", "D3"};
  const uint64_t L4_full[4] = {0, 0x95d55763cd362b42ull, 0xbbcbd13b05e08634ull, 0xa346b7687ea9accfull};
  const uint64_t H_full[4] = {0xd5ecf65eefcfa186ull, 0xed7cc0eb29d47334ull, 0x5c8fdeae63acb81eull, 0xb8452bbef665ce5bull};
  const uint64_t H_split[4] = {0xd5ecf65eefcfa186ull, 0xfe23239ffc54aad2ull, 0xbc29fb1e8f82b103ull, 0x9785a88cd76999d3ull};
  for (int which = 0; which < 4; ++which) for (ARGS_RTL = 1; ARGS_RTL >= 0; --ARGS_RTL) for (OR_RTL = 0; OR_RTL < 2; ++OR_RTL) {
    Data d = make(which);
    const uint64_t a = run(d, MP{24, 0, 0, 0, 8}), h = run(d, MP{25, 0, 0, 0, 16}), hs = run(d, MP{25, 0, 0, 0, 8});
    printf("%s args_rtl %d or_rtl %d | model A FULL %016llx %s | model H FULL %016llx %s | model H SPLIT8 %016llx %s\n", nm[which], ARGS_RTL, OR_RTL,
           (unsigned long long)a, which && a == L4_full[which] ? "== L4" : "", (unsigned long long)h, h == H_full[which] ? "== H100" : "",
           (unsigned long long)hs, hs == H_split[which] ? "== H100" : "");
    if (which < 2) { ARGS_RTL = -1; break; }
  }
}
