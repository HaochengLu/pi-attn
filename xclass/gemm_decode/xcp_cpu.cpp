// xcp_cpu.cpp -- CPU side of the portable GEMM: (1) the FP slicing used on the GPU == the integer XC slicing,
// exhaustively; (2) the reference == round-2 xc.h reference; (3) the real kernel source under emulation ==
// reference, across G / XR / M / scheduling; (4) reference hashes of the verification cases (compare with GPUs).
//   clang++ -O2 -std=c++17 -pthread xcp_cpu.cpp -o xcp_cpu && ./xcp_cpu [hashes]
#include "xcp_kernel.cuh"
#include "../../round2/theory/xc.h"
#include <string>
static float bfv(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static long check_slicing() {
  long bad = 0, n = 0;
  for (int s : {0, 2, 3, 4}) for (int E = 60; E <= 176; ++E) for (int big = 0; big < 2; ++big) {
    uint16_t grp[16] = {0}; grp[1] = (uint16_t)((E << 7) | (big ? 127 : 0)); grp[2] = 0x8000; grp[3] = (uint16_t)(0x8000 | (E << 7) | 5);
    for (uint32_t x = 0; x < 65536; ++x) {
      const int Ef = (x >> 7) & 255;
      if (Ef > E) continue;
      grp[0] = (uint16_t)x; grp[5] = (uint16_t)(x ^ 0x8000);
      int16_t n0[16], n1[16]; uint16_t s0[16], s1[16];
      const int U0 = xcp_ref_slice(grp, s, n0, n1);
      xcp_host_fp_slice(grp, s, s0, s1);
      for (int k : {0, 1, 3, 5}) {
        const float w0 = U0 ? std::ldexp((float)n0[k], U0 - 134) : 0.f, w1 = U0 ? std::ldexp((float)n1[k], U0 - 8 - s - 134) : 0.f;
        ++n;
        if (bfv(s0[k]) != w0 || (w0 == 0 && s0[k] != 0)) ++bad;
        if (s <= 3 && (bfv(s1[k]) != w1 || (w1 == 0 && s1[k] != 0))) ++bad;
      }
      if (s == 2 && E >= XC_FLUSH && Ef >= 8) {  // round-2 xc.h (its encoder is undefined below exponent field 8)
        uint16_t r0[16], r1[16]; xc_slice(grp, 16, 2, r0, r1);
        for (int k : {0, 1, 3, 5}) if (r0[k] != s0[k] || r1[k] != s1[k]) ++bad;
      }
    }
  }
  printf("slicing: %ld element checks (fp trick vs integer spec vs xc.h), %ld mismatches, %ld non-bf16 slices\n", n, bad, g_xcp_violations.load());
  return bad + g_xcp_violations.load();
}
static long check_vs_xc() {  // one segment (K = 512): same fold as the round-2 reference
  long bad = 0;
  for (int kind = 0; kind < 4; ++kind) {
    const int T = 32, K = 16 * T; std::vector<uint16_t> X, W; xcp_fill(X, 0, kind, 7 + kind, 64, K); xcp_fill(W, 1, kind, 9 + kind, 8, K);
    std::vector<uint16_t> X2((size_t)T * 1024), W2((size_t)T * 128);
    for (int t = 0; t < T; ++t) { for (int r = 0; r < 64; ++r) for (int k = 0; k < 16; ++k) X2[(size_t)t * 1024 + r * 16 + k] = X[(size_t)r * K + t * 16 + k];
                                  for (int n = 0; n < 8; ++n) for (int k = 0; k < 16; ++k) W2[(size_t)t * 128 + n * 16 + k] = W[(size_t)n * K + t * 16 + k]; }
    const int cfg[3][6] = {{0, 2, 2, 2, 2, 1}, {2, 2, 1, 0, 4, 0}, {3, 1, 1, 2, 2, 0}};  // prog, SA, SB, sA, sB, mode
    for (auto& c : cfg) {
      std::vector<float> a(512), b(512);
      xcp_reference(X.data(), K, W.data(), 64, 8, K, c[0], 32, a.data(), 2);
      xc_reference_tile(X2.data(), W2.data(), T, c[1], c[2], c[3], c[4], c[5], b.data());
      bad += memcmp(a.data(), b.data(), 2048) != 0;
    }
  }
  printf("reference vs round-2 xc_reference_tile (TRI, P21, P11; 4 data kinds): %ld mismatching configs\n", bad);
  return bad;
}
template <int G, int XR, int NW, int PROG, int CG, int R = 4, int TN = 1>
static long emu_one(xcp::Device& d, const std::vector<uint16_t>& X, const std::vector<uint16_t>& W, int M, int N, int K, int segb, int rsf, int gridf, const char* tag) {
  std::vector<float> y((size_t)M * N, -777.f), ref((size_t)M * N);
  xcp::Plan pl;
  const cudaError_t e = xcp::run<G, XR, NW, PROG, CG, R, TN>(d, X.data(), K, W.data(), y.data(), N, M, N, K, 0, segb, rsf, gridf, nullptr, &pl);
  if (e != cudaSuccess) { printf("  emu %s: refused\n", tag); return 1; }
  xcp_reference(X.data(), K, W.data(), M, N, K, PROG, segb, ref.data(), 4);
  long mm = 0; for (size_t i = 0; i < y.size(); ++i) mm += memcmp(&y[i], &ref[i], 4) != 0;
  std::vector<uint16_t> yb((size_t)M * N); long mb = 0;
  xcp::run<G, XR, NW, PROG, CG, R, TN>(d, X.data(), K, W.data(), yb.data(), N, M, N, K, 1, segb, rsf, gridf, nullptr, nullptr);
  for (size_t i = 0; i < y.size(); ++i) mb += yb[i] != emu_f2bf(ref[i]);
  printf("  emu TN%d R%d G%d XR%-2d NW%d %-4s CG%d M=%-2d N=%-4d K=%-4d segb=%d grid=%d RS=%d/%d CB=%d %s: %ld mismatches (bf16 out %ld)%s\n", TN, R, G, XR, NW, kXcpProgs[PROG].name, CG, M, N, K, segb,
         pl.grid, pl.RS, pl.S, pl.CB, tag, mm, mb, mm || mb ? "  <-- FAIL" : "");
  return mm + mb;
}
template <int BMW, int BNW, int WR, int WC, int PROG>
static long emu_tile(xcp::Device& d, const std::vector<uint16_t>& X, const std::vector<uint16_t>& W, int M, int N, int K, int segb) {
  std::vector<float> y((size_t)M * N, -777.f), ref((size_t)M * N);
  if (xcp::run_tile<BMW, BNW, WR, WC, PROG>(d, X.data(), K, W.data(), y.data(), N, M, N, K, 0, segb, nullptr) != cudaSuccess) { printf("  emu tile refused\n"); return 1; }
  xcp_reference(X.data(), K, W.data(), M, N, K, PROG, segb, ref.data(), 4);
  long mm = 0; for (size_t i = 0; i < y.size(); ++i) mm += memcmp(&y[i], &ref[i], 4) != 0;
  printf("  emu tile %dx%d (%d warps) %-4s M=%-3d N=%-4d K=%-4d segb=%d: %ld mismatches%s\n", 16 * BMW * WR, 8 * BNW * WC, WR * WC, kXcpProgs[PROG].name, M, N, K, segb, mm, mm ? "  <-- FAIL" : "");
  return mm;
}
static long check_emu() {
  emu::g_smem_max = 200 * 1024;
  long bad = 0; xcp::Device d; xcp::device_init(d, 1 << 22);
  for (int kind : {0, 2, 3}) {
    const int K1 = 1088, K2 = 2560; std::vector<uint16_t> X1, W1, X2, W2;
    xcp_fill(X1, 0, kind, 11, 64, K1); xcp_fill(W1, 1, kind, 12, 256, K1); xcp_fill(X2, 0, kind, 13, 64, K2); xcp_fill(W2, 1, kind, 14, 64, K2);
    printf(" data kind %d\n", kind);
    emu::g_sms = 5;
    bad += emu_one<1, 8, 2, 0, 0>(d, X1, W1, 8, 64, K1, 32, 0, 0, "auto");
    bad += emu_one<1, 8, 2, 1, 1, 16>(d, X1, W1, 8, 64, K1, 32, 0, 0, "cg r16");
    bad += emu_one<1, 8, 2, 1, 1, 4, 2>(d, X1, W1, 8, 256, K1, 32, 0, 0, "TN2");
    bad += emu_one<1, 8, 2, 1, 1, 4, 4>(d, X1, W1, 3, 256, K1, 32, 1, 0, "TN4 rs1");
    bad += emu_one<1, 16, 3, 0, 1, 2, 2>(d, X2, W2, 13, 64, K2, 32, 2, 0, "TN2 rs2");
    bad += emu_one<1, 16, 2, 2, 1, 8, 4>(d, X2, W2, 16, 64, K2, 32, 0, 3, "TN4 auto");
    bad += emu_one<2, 16, 2, 1, 1, 4, 2>(d, X1, W1, 27, 256, K1, 32, 1, 0, "G2 TN2 rs1");
    bad += emu_one<4, 16, 2, 3, 1, 4, 2>(d, X2, W2, 64, 64, K2, 32, 0, 0, "G4 TN2");
    bad += emu_one<1, 8, 24, 1, 1, 4, 2>(d, X1, W1, 5, 256, K1, 32, 0, 1, "nw24 TN2 (NB = 1)");
    bad += emu_one<1, 8, 24, 1, 1, 4>(d, X1, W1, 6, 256, K1, 32, 0, 1, "nw24 (NB = 2)");
    bad += emu_one<1, 16, 24, 2, 0, 4>(d, X1, W1, 16, 256, K1, 32, 1, 1, "nw24 rs1 (NB = 2)");
    bad += emu_one<1, 16, 3, 0, 1, 2>(d, X2, W2, 13, 64, K2, 32, 1, 0, "cg r2 rs1");
    bad += emu_one<2, 16, 2, 2, 1, 32>(d, X2, W2, 29, 64, K2, 32, 2, 2, "cg r32 rs2");
    bad += emu_one<1, 8, 2, 1, 0>(d, X1, W1, 1, 64, K1, 32, 0, 0, "auto");
    bad += emu_one<1, 8, 3, 1, 0>(d, X1, W1, 5, 256, K1, 32, 1, 0, "rs1");
    bad += emu_one<1, 8, 3, 1, 1>(d, X1, W1, 5, 256, K1, 32, 99, 0, "local");
    bad += emu_one<1, 16, 2, 0, 0>(d, X1, W1, 16, 64, K1, 32, 2, 3, "rs2");
    bad += emu_one<1, 16, 2, 1, 1>(d, X2, W2, 11, 64, K2, 32, 1, 0, "rs1");
    bad += emu_one<1, 16, 2, 1, 0>(d, X2, W2, 16, 64, K2, 16, 3, 0, "segb16 rs3");
    bad += emu_one<1, 16, 2, 2, 0>(d, X2, W2, 16, 64, K2, 32, 0, 2, "auto");
    bad += emu_one<1, 16, 2, 3, 0>(d, X2, W2, 9, 64, K2, 32, 1, 1, "rs1");
    bad += emu_one<2, 16, 2, 1, 0>(d, X1, W1, 32, 64, K1, 32, 0, 0, "auto");
    bad += emu_one<2, 16, 2, 0, 0>(d, X1, W1, 23, 64, K1, 32, 1, 4, "rs1");
    bad += emu_one<4, 16, 2, 1, 0>(d, X1, W1, 64, 64, K1, 32, 0, 0, "auto");
    bad += emu_one<4, 16, 2, 2, 0>(d, X2, W2, 50, 64, K2, 32, 2, 3, "rs2");
    {  // canonical weights: same function, no W slicing in the kernel
      std::vector<uint16_t> W12 = W1, W10 = W2, Wt = W1; std::vector<float> a(8 * 256), b(8 * 256);
      const long c12 = xcp_canon_weights(W12.data(), 256, K1, 4), c10 = xcp_canon_weights(W10.data(), 64, K2, 2);
      Wt = W12; const long again = xcp_canon_weights(Wt.data(), 256, K1, 4);
      xcp_reference(X1.data(), K1, W1.data(), 8, 256, K1, 2, 32, a.data()); xcp_reference(X1.data(), K1, W12.data(), 8, 256, K1, 2, 32, b.data());
      const long d12 = memcmp(a.data(), b.data(), a.size() * 4) != 0;
      printf("  canonical weights: %ld / %ld elements moved (12-bit / 10-bit lattice); idempotent: %s; P21(X, W) == P21(X, canon W): %s\n", c12, c10, again ? "NO" : "yes", d12 ? "NO" : "yes");
      bad += again + d12;
      bad += emu_one<1, 8, 2, 5, 0, 8>(d, X1, W12, 7, 64, K1, 32, 0, 0, "canon auto");
      bad += emu_one<1, 16, 3, 5, 0, 2>(d, X1, W12, 16, 256, K1, 32, 1, 0, "canon rs1");
      bad += emu_one<4, 16, 2, 6, 0, 4>(d, X2, W10, 40, 64, K2, 32, 2, 0, "canon rs2");
      bad += emu_one<1, 8, 2, 1, 1, 8>(d, X1, W1, 3, 64, K1, 32, 0, 0, "R6 auto");
      bad += emu_tile<2, 1, 1, 2, 5>(d, X1, W12, 20, 64, K1, 32);
      bad += emu_tile<1, 2, 2, 1, 6>(d, X2, W10, 30, 64, K2, 32);
    }
    bad += emu_tile<1, 1, 1, 2, 1>(d, X1, W1, 16, 64, K1, 32);
    bad += emu_tile<2, 1, 1, 2, 0>(d, X1, W1, 50, 64, K1, 32);
    bad += emu_tile<1, 2, 2, 1, 1>(d, X2, W2, 64, 64, K2, 32);
    bad += emu_tile<2, 2, 1, 2, 2>(d, X2, W2, 37, 64, K2, 16);
    bad += emu_tile<1, 2, 2, 2, 3>(d, X1, W1, 33, 256, K1, 32);
    emu::g_smem_max = 20 * 1024;  // forces small X chunks (CB = 4 / 8)
    bad += emu_one<1, 16, 2, 1, 0>(d, X2, W2, 16, 64, K2, 32, 0, 0, "small smem");
    bad += emu_one<1, 8, 2, 0, 0>(d, X1, W1, 8, 64, K1, 32, 1, 0, "small smem rs1");
    emu::g_smem_max = 200 * 1024;
  }
  printf("emulation: T1 violations inside the emulated tensor core %ld, reference step violations %ld\n", emu::t1_violations.load(), g_xcp_violations.load());
  return bad + emu::t1_violations.load() + g_xcp_violations.load();
}
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "hashes") {
    for (int ci = 0; ci < kXcpNumCases; ++ci) {
      const XcpCase c = kXcpCases[ci]; std::vector<uint16_t> X, W; xcp_fill(X, 0, c.kind, 1000 + ci, 64, c.K); xcp_fill(W, 1, c.kind, 2000 + ci, c.N, c.K);
      for (int prog = 0; prog < 4; ++prog) {
        std::vector<float> y((size_t)kXcpRefRows * c.N); xcp_reference(X.data(), c.K, W.data(), kXcpRefRows, c.N, c.K, prog, 32, y.data());
        printf("XCPHASH case %d N=%d K=%d kind=%d %s cpu=%016llx\n", ci, c.N, c.K, c.kind, kXcpProgs[prog].name, (unsigned long long)xcp_fnv(y.data(), y.size() * 4));
        fflush(stdout);
      }
    }
    printf("reference step violations %ld\n", g_xcp_violations.load());
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "vectors") {  // small vectors for the independent Python reference
    const int M = 2, N = 3, K = 1088;
    for (int kind = 0; kind < 4; ++kind) {
      std::vector<uint16_t> X, W; xcp_fill(X, 0, kind, 50 + kind, M, K); xcp_fill(W, 1, kind, 60 + kind, N, K);
      printf("DATA %d %d %d\nX", M, N, K); for (auto v : X) printf(" %04x", v); printf("\nW"); for (auto v : W) printf(" %04x", v); printf("\n");
      for (int prog = 0; prog < 4; ++prog) {
        std::vector<float> y((size_t)M * N); xcp_reference(X.data(), K, W.data(), M, N, K, prog, 32, y.data(), 1);
        printf("PROG %d", prog); for (float f : y) { uint32_t u; memcpy(&u, &f, 4); printf(" %08x", u); } printf("\n");
      }
    }
    return 0;
  }
  long bad = check_slicing();
  bad += check_vs_xc();
  bad += check_emu();
  printf("CPU CHECK %s (%ld)\n", bad ? "FAILED" : "PASSED", bad);
  return bad != 0;
}
