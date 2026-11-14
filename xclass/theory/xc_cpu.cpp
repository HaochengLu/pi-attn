// xc_cpu.cpp -- CPU side: reference hashes for the XC programs, model hashes for native MMA on class A / H,
// accuracy vs fp64, and (optionally) comparison with a GPU dump xc_out.bin.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "xc.h"
#include "xcdata.h"
#include "model.h"
static uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 1469598103934665603ull; const unsigned char* p = (const unsigned char*)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
static double bfd(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char** argv) {
  const int tiles = atoi(argv[1]), T = atoi(argv[2]);
  const int prog[][5] = {{0, 0, 0, 0, 0}, {1, 1, 2, 2, 0}, {2, 1, 2, 2, 0}, {2, 2, 2, 2, 1}, {2, 2, 2, 2, 0}, {1, 1, 5, 5, 2}};
  std::vector<FILE*> dumps; for (int i = 3; i < argc; ++i) dumps.push_back(fopen(argv[i], "rb"));
  for (int which = 0; which < 4; ++which) {
    XData d = xc_make(which, tiles, T); const size_t nout = (size_t)tiles * 512;
    std::vector<double> ex(nout); double nrm = 0;
    for (int t = 0; t < tiles; ++t) for (int r = 0; r < 64; ++r) for (int n = 0; n < 8; ++n) {
      long double s = 0; for (int q = 0; q < T; ++q) for (int k = 0; k < 16; ++k) s += (long double)(bfd(d.X[((size_t)t * T + q) * 1024 + r * 16 + k]) * bfd(d.W[((size_t)t * T + q) * 128 + n * 16 + k]));
      ex[(size_t)t * 512 + r * 8 + n] = (double)s; nrm += (double)(s * s);
    }
    for (auto& pr : prog) {
      std::vector<std::vector<uint32_t>> outs; std::vector<const char*> names;
      if (pr[0] == 0) {
        const MP A{24, 0, 0, 0, 8}, H{25, 0, 0, 0, 16};
        for (int cls = 0; cls < 2; ++cls) {
          std::vector<uint32_t> o(nout);
          for (int t = 0; t < tiles; ++t) for (int r = 0; r < 64; ++r) for (int n = 0; n < 8; ++n) {
            uint32_t c = 0; for (int q = 0; q < T; ++q) c = mma_k16(cls ? H : A, c, &d.X[((size_t)t * T + q) * 1024 + r * 16], &d.W[((size_t)t * T + q) * 128 + n * 16]);
            o[(size_t)t * 512 + r * 8 + n] = c;
          }
          outs.push_back(o); names.push_back(cls ? "native, model H" : "native, model A");
        }
      } else {
        std::vector<uint32_t> o(nout);
        for (int t = 0; t < tiles; ++t) xc_reference_tile(&d.X[(size_t)t * T * 1024], &d.W[(size_t)t * T * 128], T, pr[0], pr[1], pr[2], pr[3], pr[4], (float*)&o[(size_t)t * 512]);
        outs.push_back(o); names.push_back(pr[4] == 2 ? "XR CPU reference" : pr[4] == 1 ? "XC-tri CPU ref" : "XC CPU reference");
      }
      std::vector<std::vector<uint32_t>> g;
      for (FILE* f : dumps) { std::vector<uint32_t> o(nout); if (fread(o.data(), 4, nout, f) != nout) { printf("short dump\n"); return 1; } g.push_back(o); }
      for (size_t v = 0; v < outs.size(); ++v) {
        double err = 0; for (size_t i = 0; i < nout; ++i) { float fv; memcpy(&fv, &outs[v][i], 4); const double e = fv - ex[i]; err += e * e; }
        printf("data %d prog SA%d SB%d sA%d sB%d mode%d  %-18s hash %016llx  rel-L2 err vs exact %.3e", which, pr[0], pr[1], pr[2], pr[3], pr[4], names[v], (unsigned long long)fnv(outs[v]), std::sqrt(err / nrm));
        for (size_t q = 0; q < g.size(); ++q) { size_t mm = 0; for (size_t i = 0; i < nout; ++i) mm += g[q][i] != outs[v][i]; printf("  | GPU dump %zu: %zu mismatches", q, mm); }
        printf("\n");
      }
    }
  }
}
