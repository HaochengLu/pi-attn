#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gen.h"
#include "model.h"
int main(int argc, char** argv) {
  // fit out.bin arrayidx group seed n1..n8
  const char* fn = argv[1]; int arr = atoi(argv[2]); int group = atoi(argv[3]);
  int counts[16] = {0}; uint64_t seed = strtoull(argv[4], 0, 0);
  for (int i = 5; i < argc; ++i) counts[i - 4] = atoi(argv[i]);
  std::vector<Case> cs; std::vector<uint8_t> set; gen::make(cs, set, seed, counts);
  FILE* f = fopen(fn, "rb"); uint64_t hdr[2]; fread(hdr, 8, 2, f);
  if (hdr[0] != cs.size()) { printf("size mismatch %llu %zu\n", (unsigned long long)hdr[0], cs.size()); return 1; }
  std::vector<uint32_t> o(cs.size()); fseek(f, 16 + (long)arr * 4 * (long)cs.size(), SEEK_SET); fread(o.data(), 4, cs.size(), f);
  if (getenv("SHOW")) {
    MP p{atoi(getenv("F")), 0, 0, 0, group}; int want = atoi(getenv("SHOW")), shown = 0;
    for (size_t i = 0; i < cs.size() && shown < 40; ++i) if (set[i] == want) {
      uint32_t m = mma_k16(p, cs[i].c, cs[i].a, cs[i].b); if (m == o[i]) continue; ++shown;
      printf("case %zu c %08x gpu %08x model %08x :", i, cs[i].c, o[i], m);
      for (int k = 0; k < 16; ++k) if ((cs[i].a[k] & 0x7FFF) || (cs[i].b[k] & 0x7FFF)) printf(" [%d] %04x*%04x", k, cs[i].a[k], cs[i].b[k]);
      printf("\n");
    }
    return 0;
  }
  for (int F = 22; F <= 28; ++F) for (int am = 0; am < 2; ++am) for (int fm = 0; fm < 5; ++fm) for (int pn = 0; pn < 2; ++pn) {
    MP p{F, am, fm, pn, group}; long bad[16] = {0}, tot[16] = {0};
    for (size_t i = 0; i < cs.size(); ++i) { ++tot[set[i]]; bad[set[i]] += mma_k16(p, cs[i].c, cs[i].a, cs[i].b) != o[i]; }
    long b15 = bad[1] + bad[2] + bad[3] + bad[4] + bad[5];
    { printf("%ld F %d am %d fm %d pn %d :", b15, F, am, fm, pn); for (int s = 1; s <= 13; ++s) if (tot[s]) printf(" s%d %ld/%ld", s, bad[s], tot[s]); printf("\n"); }
  }
}
