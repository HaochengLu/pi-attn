// predict.cpp -- model-only: per set, how often do model A, model H and the exact sum agree? (and vs GPU dumps if given)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gen.h"
#include "model.h"
// exact c + sum a b as integer * 2^lsb, then: representable in fp32? and RZ value
static bool exact_sum(const Case& c, int n, uint32_t* rz, bool* repr) {
  Term t[17]; int nt = 0; t[nt++] = term_c(c.c); for (int k = 0; k < n; ++k) t[nt++] = term_p(c.a[k], c.b[k], 0);
  int lo = 100000; for (int i = 0; i < nt; ++i) if (t[i].nz && t[i].lsb < lo) lo = t[i].lsb;
  if (lo == 100000) { *rz = 0; *repr = true; return true; }
  i128 S = 0; for (int i = 0; i < nt; ++i) if (t[i].nz) { if (t[i].lsb - lo > 80) return false; i128 v = (i128)t[i].m << (t[i].lsb - lo); S += t[i].s ? -v : v; }
  if (S == 0) { *rz = 0; *repr = true; return true; }
  i128 mag = S < 0 ? -S : S; *rz = round_f32(S < 0, mag, lo, 0); *repr = (*rz == round_f32(S < 0, mag, lo, 3)); return true;
}
int main(int argc, char** argv) {
  // predict seed n1..n13 [-- outA.bin outH.bin]
  int counts[16] = {0}; uint64_t seed = strtoull(argv[1], 0, 0); int i = 2;
  for (; i < argc && argv[i][0] != '-'; ++i) counts[i - 1] = atoi(argv[i]);
  std::vector<Case> cs; std::vector<uint8_t> set; gen::make(cs, set, seed, counts);
  std::vector<uint32_t> gA, gH, gHs, gHw;
  auto rd = [&](const char* fn, int arr, std::vector<uint32_t>& o) { FILE* f = fopen(fn, "rb"); uint64_t h[2]; fread(h, 8, 2, f); if (h[0] != cs.size() || (int)h[1] <= arr) { printf("bad file %s\n", fn); exit(1); } o.resize(cs.size()); fseek(f, 16 + (long)arr * 4 * (long)cs.size(), SEEK_SET); fread(o.data(), 4, cs.size(), f); fclose(f); };
  if (i < argc) { rd(argv[i + 1], 0, gA); rd(argv[i + 2], 0, gH); rd(argv[i + 2], 1, gHs); }
  const MP A{24, 0, 0, 0, 8}, H{25, 0, 0, 0, 16}, Hs{25, 0, 0, 0, 8};
  long tot[16] = {0}, ah[16] = {0}, aex[16] = {0}, hex_[16] = {0}, rep[16] = {0}, arz[16] = {0}, hrz[16] = {0}, gbadA[16] = {0}, gbadH[16] = {0}, gbadHs[16] = {0}, gAH[16] = {0};
  for (size_t q = 0; q < cs.size(); ++q) {
    const int s = set[q]; const uint32_t a = mma_k16(A, cs[q].c, cs[q].a, cs[q].b), h = mma_k16(H, cs[q].c, cs[q].a, cs[q].b);
    uint32_t rz = 0; bool r = false; const bool ok = exact_sum(cs[q], 16, &rz, &r);
    ++tot[s]; ah[s] += a == h; rep[s] += ok && r; aex[s] += ok && r && a == rz; hex_[s] += ok && r && h == rz; arz[s] += ok && a == rz; hrz[s] += ok && h == rz;
    if (!gA.empty()) { gbadA[s] += gA[q] != a; gbadH[s] += gH[q] != h; gbadHs[s] += gHs[q] != mma_k16(Hs, cs[q].c, cs[q].a, cs[q].b); gAH[s] += gA[q] == gH[q]; }
  }
  for (int s = 1; s < 16; ++s) if (tot[s]) {
    printf("set %2d n %7ld | model: A==H %7ld  exact-representable %7ld (A exact %7ld, H exact %7ld)  A==RZ(exact) %7ld  H==RZ(exact) %7ld", s, tot[s], ah[s], rep[s], aex[s], hex_[s], arz[s], hrz[s]);
    if (!gA.empty()) printf(" | GPU: L4!=modelA %ld  H100!=modelH %ld  H100split8!=model %ld  L4==H100 %ld", gbadA[s], gbadH[s], gbadHs[s], gAH[s]);
    printf("\n");
  }
}
