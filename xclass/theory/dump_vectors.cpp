// dump_vectors.cpp -- text test vectors for the Python references (tc_models.py, xc_ref.py)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "xc.h"
#include "xcdata.h"
int main() {
  { // MMA cases with measured GPU outputs (cal2): every 97th case
    int counts[16] = {0, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 60000, 60000, 60000, 60000, 10000};
    std::vector<Case> cs; std::vector<uint8_t> set; gen::make(cs, set, 2, counts);
    std::vector<uint32_t> gA(cs.size()), gH(cs.size());
    FILE* f = fopen("out_cal2_L4.bin", "rb"); fseek(f, 16, SEEK_SET); fread(gA.data(), 4, cs.size(), f); fclose(f);
    f = fopen("out_cal2_H100.bin", "rb"); fseek(f, 16, SEEK_SET); fread(gH.data(), 4, cs.size(), f); fclose(f);
    FILE* o = fopen("vectors_mma.txt", "w");
    for (size_t i = 0; i < cs.size(); i += 97) {
      fprintf(o, "%d %08x %08x %08x", set[i], cs[i].c, gA[i], gH[i]);
      for (int k = 0; k < 16; ++k) fprintf(o, " %04x", cs[i].a[k]); for (int k = 0; k < 16; ++k) fprintf(o, " %04x", cs[i].b[k]); fprintf(o, "\n");
    }
    fclose(o);
  }
  { // XC program vectors: 1 tile, T = 8 (K = 128), 4 datasets, with the C++ reference outputs
    const int prog[][5] = {{1, 1, 2, 2, 0}, {2, 1, 2, 2, 0}, {2, 2, 2, 2, 1}, {2, 2, 2, 2, 0}, {1, 1, 5, 5, 2}};
    FILE* o = fopen("vectors_xc.txt", "w");
    for (int which = 0; which < 4; ++which) {
      XData d = xc_make(which, 1, 8);
      fprintf(o, "DATA %d 8\nX", which); for (auto v : d.X) fprintf(o, " %04x", v); fprintf(o, "\nW"); for (auto v : d.W) fprintf(o, " %04x", v); fprintf(o, "\n");
      for (auto& pr : prog) {
        std::vector<uint32_t> out(512); xc_reference_tile(d.X.data(), d.W.data(), 8, pr[0], pr[1], pr[2], pr[3], pr[4], (float*)out.data());
        fprintf(o, "PROG %d %d %d %d %d", pr[0], pr[1], pr[2], pr[3], pr[4]); for (auto v : out) fprintf(o, " %08x", v); fprintf(o, "\n");
      }
    }
    fclose(o);
  }
}
