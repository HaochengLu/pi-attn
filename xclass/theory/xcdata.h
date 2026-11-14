// xcdata.h -- deterministic GEMM-shaped data for the XC program test. X [tiles][T][64][16], W [tiles][T][8][16].
#pragma once
#include <cmath>
#include "gen.h"
struct XData { int tiles, T; std::vector<uint16_t> X, W; };
static inline XData xc_make(int which, int tiles, int T) {
  XData d; d.tiles = tiles; d.T = T; gen::g_state = 0xC0FFEE + which;
  d.X.resize((size_t)tiles * T * 1024); d.W.resize((size_t)tiles * T * 128);
  if (which == 0) {            // Gaussian activations, small Gaussian weights
    for (auto& v : d.X) v = gen::f2bf(gen::gauss());
    for (auto& v : d.W) v = gen::f2bf(0.02f * gen::gauss());
  } else if (which == 1) {     // heavy-tailed activations: Gaussian times 2^[-6, 6], weights Gaussian times 2^[-3, 3]
    for (auto& v : d.X) { const float g = gen::gauss(); const int e = gen::ri(-6, 6); v = gen::f2bf(std::ldexp(g, e)); }
    for (auto& v : d.W) { const float g = 0.02f * gen::gauss(); const int e = gen::ri(-3, 3); v = gen::f2bf(std::ldexp(g, e)); }
  } else if (which == 2) {     // wide exponents: uniform exponent in [-10, 10], random mantissa and sign
    for (auto& v : d.X) v = gen::bfb_R(-10, 10);
    for (auto& v : d.W) v = gen::bfb_R(-10, 10);
  } else {                     // sparse / ReLU-like with exact zeros and a few huge outliers
    for (auto& v : d.X) { const float g = gen::gauss(); const int o = gen::ri(0, 99); v = g < 0 ? 0 : gen::f2bf(o == 0 ? g * 300.f : g); }
    for (auto& v : d.W) v = gen::f2bf(0.02f * gen::gauss());
  }
  return d;
}
