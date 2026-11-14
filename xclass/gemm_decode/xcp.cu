// xcp.cu -- portable GEMM: GPU verification (vs the exact CPU reference in xcp.h) and benchmarks vs cuBLAS and
// PI-Attn's own (class-dependent) kernels.
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_XX [-DWITH_PI -I/pi/kernels -I<cutlass>/include] xcp.cu -o xcp -lcublas
//   ./xcp verify [case ...]      ./xcp bench [quick]
#include "xcp_kernel.cuh"
#include <cublas_v2.h>
#include <string>
#ifdef WITH_PI
#include "gemm_dispatch.cuh"
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

static xcp::Device g_dev;
static cublasHandle_t g_cb;
// prog: 0 TRI, 1 TRIF, 2 P21, 3 P11, 4 native (class-dependent skeleton), 5 P21C, 6 P11C (canonical weights).
// Scheduling only (no effect on bits): nw warps per CTA, r ring depth, cg, tn n8 tiles per warp, rs segments per run, grid, gsel
// (0 auto, 8: (G1, XR8), 1 / 2 / 4: G with XR16), tile (0: small-M kernel; 1: 128x64 tile kernel; 2: 128x128).
struct Opt { int prog = 1, nw = 16, cg = 0, r = 4, tn = 1, segb = 32, rs = 0, grid = 0, gsel = 0, tile = 0; };

static cudaError_t xcp_launch(const Opt& o, const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N, int K, int out_bf16,
                              xcp::Plan* pl = nullptr) {
  if (o.tile) {
#define XCP_T(PR_) \
    if (o.prog == PR_ && o.tile == 1) return xcp::run_tile<4, 2, 2, 4, PR_>(g_dev, x, ldx, w, y, ldy, M, N, K, out_bf16, o.segb, 0); \
    if (o.prog == PR_ && o.tile == 2) return xcp::run_tile<4, 2, 2, 8, PR_>(g_dev, x, ldx, w, y, ldy, M, N, K, out_bf16, o.segb, 0);
    XCP_T(0) XCP_T(1) XCP_T(2) XCP_T(3) XCP_T(4) XCP_T(5) XCP_T(6)
    return cudaErrorInvalidValue;
  }
  const int G = o.gsel == 0 ? (M <= 16 ? 1 : M <= 32 ? 2 : 4) : (o.gsel == 8 ? 1 : o.gsel);
  const int XR = o.gsel == 0 ? (M <= 8 ? 8 : 16) : (o.gsel == 8 ? 8 : 16);
#define XCP_V(G_, XR_, NW_, PR_, CG_, R_, TN_) \
  if (G == G_ && XR == XR_ && o.nw == NW_ && o.prog == PR_ && o.cg == CG_ && o.r == R_ && o.tn == TN_) \
    return xcp::run<G_, XR_, NW_, PR_, CG_, R_, TN_>(g_dev, x, ldx, w, y, ldy, M, N, K, out_bf16, o.segb, o.rs, o.grid, 0, pl);
#define XCP_G1(NW_, PR_, CG_, R_, TN_) XCP_V(1, 8, NW_, PR_, CG_, R_, TN_) XCP_V(1, 16, NW_, PR_, CG_, R_, TN_)
#define XCP_FULL(PR_) XCP_G1(16, PR_, 0, 4, 1) XCP_G1(16, PR_, 1, 4, 1) XCP_G1(24, PR_, 1, 4, 1) XCP_G1(32, PR_, 1, 4, 1) \
  XCP_G1(8, PR_, 1, 4, 2) XCP_G1(12, PR_, 1, 4, 2) XCP_G1(16, PR_, 1, 4, 2) XCP_G1(24, PR_, 1, 4, 2) XCP_G1(8, PR_, 1, 8, 2) \
  XCP_G1(4, PR_, 1, 4, 4) XCP_G1(8, PR_, 1, 4, 4) XCP_G1(12, PR_, 1, 4, 4) XCP_G1(4, PR_, 1, 8, 4)
  XCP_FULL(1) XCP_FULL(5)
  XCP_G1(16, 0, 0, 4, 1) XCP_G1(16, 2, 0, 4, 1) XCP_G1(16, 3, 0, 4, 1)
#define XCP_LITE(PR_) XCP_G1(16, PR_, 1, 4, 1) XCP_G1(8, PR_, 1, 4, 2) XCP_G1(16, PR_, 1, 4, 2) XCP_G1(8, PR_, 1, 4, 4)
  XCP_LITE(6) XCP_LITE(4) XCP_G1(16, 6, 0, 4, 1) XCP_G1(16, 4, 0, 4, 1)
#define XCP_BIG(PR_) XCP_V(2, 16, 16, PR_, 0, 4, 1) XCP_V(4, 16, 16, PR_, 0, 4, 1)
  XCP_BIG(0) XCP_BIG(1) XCP_BIG(2) XCP_BIG(3) XCP_BIG(4) XCP_BIG(5) XCP_BIG(6)
#define XCP_BIG2(PR_) XCP_V(2, 16, 16, PR_, 1, 4, 1) XCP_V(4, 16, 16, PR_, 1, 4, 1) XCP_V(2, 16, 8, PR_, 1, 4, 2) XCP_V(4, 16, 8, PR_, 1, 4, 2)
  XCP_BIG2(1) XCP_BIG2(5) XCP_BIG2(6) XCP_BIG2(4)
  return cudaErrorInvalidValue;
}
// any M. Small-M kernel: 64 rows per launch (M > 64 re-streams W per 64 rows; use the tile kernel there).
static cudaError_t xcp_gemm(const Opt& o, const void* x, int64_t ldx, const void* w, void* y, int64_t ldy, int M, int N, int K, int out_bf16,
                            xcp::Plan* pl = nullptr) {
  if (o.tile) return xcp_launch(o, x, ldx, w, y, ldy, M, N, K, out_bf16);
  for (int m0 = 0; m0 < M; m0 += 64) {
    const cudaError_t e = xcp_launch(o, (const uint16_t*)x + (int64_t)m0 * ldx, ldx, w, (char*)y + (int64_t)m0 * ldy * (out_bf16 ? 2 : 4), ldy,
                                     std::min(64, M - m0), N, K, out_bf16, m0 == 0 ? pl : nullptr);
    if (e != cudaSuccess) return e;
  }
  return cudaSuccess;
}
static cublasStatus_t cublas_gemm(const void* x, const void* w, void* y, int M, int N, int K) {  // Y^T = W X^T, bf16 out, fp32 compute
  const float one = 1.f, zero = 0.f;
  return cublasGemmEx(g_cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &zero, y, CUDA_R_16BF, N,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
#ifdef WITH_PI
static pi_gemm::GemmDevice g_pidev;
static cudaError_t pi_run(int id, const void* x, const void* w, void* y, int M, int N, int K) {
  const cudaError_t e = pi_gemm::run_gemm<__nv_bfloat16>(id, x, K, w, (__nv_bfloat16*)y, N, M, N, K, pi_gemm::gemm_segments(N, K), g_pidev, 0);
  if (e != cudaSuccess) cudaGetLastError();
  return e;
}
static cudaError_t pi_small_fp32(const void* x, const void* w, float* y, int M, int N, int K) {  // id 1 with fp32 output
  using namespace pi_gemm;
  const size_t smem = (size_t)kSmallStages * (16 + 32) * (kSmallBK + 8) * sizeof(__nv_bfloat16);
  cudaFuncSetAttribute(gemm_small_kernel<1, 4, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gemm_small_kernel<1, 4, float><<<dim3(N / 32, (M + 15) / 16), 128, smem>>>((const __nv_bfloat16*)x, K, (const __nv_bfloat16*)w, y, N, M, N, K,
                                                                         gemm_segments(N, K));
  return cudaGetLastError();
}
#endif

// f(i) launches GEMM number i; i selects the weight copy (cold regime: copies cycle so every launch streams from DRAM)
template <class F> static double time_us(F&& f, double target_ms, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f(i);
  CK(cudaDeviceSynchronize());
  float ms = 0;
  CK(cudaEventRecord(a)); for (int i = 0; i < 4; ++i) f(i); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
  const int iters = std::max(4, std::min(400, (int)(target_ms / (ms / 4 + 1e-6))));
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a)); for (int i = 0; i < iters; ++i) f(i + 4); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, (double)ms / iters);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  return best * 1000;
}

static long diff(const float* got, const float* ref, size_t n) { long d = 0; for (size_t i = 0; i < n; ++i) d += memcmp(got + i, ref + i, 4) != 0; return d; }
static float bfv(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static int verify(const std::vector<int>& which) {
  long total_bad = 0;
  for (int ci = 0; ci < kXcpNumCases; ++ci) {
    if (!which.empty() && std::find(which.begin(), which.end(), ci) == which.end()) continue;
    const XcpCase c = kXcpCases[ci]; const int N = c.N, K = c.K, R = kXcpRefRows;
    std::vector<uint16_t> X, W, X2(16 * (size_t)K);
    xcp_fill(X, 0, c.kind, 1000 + ci, 64, K); xcp_fill(W, 1, c.kind, 2000 + ci, N, K);
    std::vector<uint16_t> W12 = W, W10 = W;
    const long ch12 = xcp_canon_weights(W12.data(), N, K, 4), ch10 = xcp_canon_weights(W10.data(), N, K, 2);
    memcpy(X2.data(), X.data(), (size_t)K * 2); memcpy(X2.data() + K, X.data() + (size_t)33 * K, (size_t)15 * K * 2);  // row 0 + other rows
    uint16_t *dX, *dX2, *dW, *dW12, *dW10, *dYb; float* dY;
    CK(cudaMalloc(&dX, X.size() * 2)); CK(cudaMalloc(&dX2, X2.size() * 2)); CK(cudaMalloc(&dW, W.size() * 2)); CK(cudaMalloc(&dW12, W.size() * 2)); CK(cudaMalloc(&dW10, W.size() * 2));
    CK(cudaMalloc(&dY, (size_t)64 * N * 4)); CK(cudaMalloc(&dYb, (size_t)64 * N * 2));
    CK(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX2, X2.data(), X2.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW12, W12.data(), W.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW10, W10.data(), W.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> got((size_t)64 * N), ref((size_t)R * N);
    std::vector<double> ex((size_t)R * N); xcp_exact64(X.data(), K, W.data(), R, N, K, ex.data());
    double nrm = 0; for (double v : ex) nrm += v * v;
    auto rel = [&](const float* y) { double e = 0; for (size_t i = 0; i < (size_t)R * N; ++i) { const double d = y[i] - ex[i]; e += d * d; } return std::sqrt(e / nrm); };
    double acc[7] = {0, 0, 0, 0, 0, 0, 0}, acc_bf = 0, acc_bf21 = 0;
    for (int prog : {0, 1, 2, 3, 5, 6}) {
      const uint16_t* dWp = prog == 5 ? dW12 : prog == 6 ? dW10 : dW;
      // run rows [r0, r0 + M) of `src` and compare the first `cmp` rows with reference rows [ref0, ...). -1: refused on this GPU
      auto test = [&](Opt o, const uint16_t* src, int r0, int M, int ref0, int cmp) -> long {
        CK(cudaMemset(dY, 0xFF, (size_t)M * N * 4));
        const cudaError_t e = xcp_gemm(o, src + (size_t)r0 * K, K, dWp, dY, N, M, N, K, 0);
        if (e != cudaSuccess) { cudaGetLastError(); return -1; }
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got.data(), dY, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
        return diff(got.data(), ref.data() + (size_t)ref0 * N, (size_t)cmp * N);
      };
      xcp_reference(X.data(), K, W.data(), R, N, K, prog, 32, ref.data());  // always from the ORIGINAL weights
      Opt o; o.prog = prog;
      std::vector<long> mm; std::string names;
      auto add = [&](const char* nm, long v) { mm.push_back(v); names += nm; names += ' '; };
      add("M16", test(o, dX, 0, 16, 0, 16));
      const uint64_t hg = xcp_fnv(got.data(), (size_t)R * N * 4), hc = xcp_fnv(ref.data(), (size_t)R * N * 4);
      acc[prog] = rel(got.data());
      if (prog >= 5) {  // the reference on canonical weights is the same function
        std::vector<float> r2((size_t)R * N); xcp_reference(X.data(), K, prog == 5 ? W12.data() : W10.data(), R, N, K, prog, 32, r2.data());
        add("ref(canonW)", diff(r2.data(), ref.data(), (size_t)R * N));
        Opt p = o; p.prog = prog == 5 ? 2 : 3; add("online-slicing(canonW)", test(p, dX, 0, 16, 0, 16));
      }
      add("M8", test(o, dX, 0, 8, 0, 8)); add("M1r0", test(o, dX, 0, 1, 0, 1)); add("M1r5", test(o, dX, 5, 1, 5, 1)); add("M3r9", test(o, dX, 9, 3, 9, 3));
      add("M16+junk", test(o, dX2, 0, 16, 0, 1));
      { Opt p = o; p.rs = 1; add("rs1", test(p, dX, 0, 16, 0, 16)); p.rs = 2; add("rs2", test(p, dX, 0, 8, 0, 8)); p.rs = 999; add("local", test(p, dX, 0, 16, 0, 16)); }
      { Opt p = o; p.grid = 7; add("grid7", test(p, dX, 0, 16, 0, 16)); p.grid = 1; p.rs = 2; add("grid1", test(p, dX, 3, 5, 3, 5)); }
      add("M32", test(o, dX, 0, 32, 0, 16)); add("M64", test(o, dX, 0, 64, 0, 16));
      { Opt p = o; p.gsel = 4; add("M13/G4", test(p, dX, 2, 13, 2, 13)); p.gsel = 1; add("M2/XR16", test(p, dX, 7, 2, 7, 2)); p.gsel = 2; p.rs = 1; add("M20/G2rs1", test(p, dX, 0, 20, 0, 16)); }
      if (prog == 1 || prog == 5) {
        Opt p = o; p.cg = 1; add("cg16", test(p, dX, 0, 16, 0, 16)); add("cg16M3", test(p, dX, 9, 3, 9, 3)); p.rs = 1; p.nw = 24; add("cg24rs1", test(p, dX, 0, 16, 0, 16)); add("cg24M8rs1", test(p, dX, 0, 8, 0, 8));
        p.rs = 0; p.nw = 32; add("cg32", test(p, dX, 0, 16, 0, 16)); p.gsel = 4; p.nw = 16; add("cgG4M64", test(p, dX, 0, 64, 0, 16));
        if (N % 32 == 0) {
          p = o; p.cg = 1; p.tn = 2; add("tn2", test(p, dX, 0, 16, 0, 16)); p.nw = 8; p.rs = 1; add("tn2nw8M8rs1", test(p, dX, 0, 8, 0, 8)); p.r = 8; p.rs = 999; add("tn2r8local", test(p, dX, 0, 16, 0, 16));
          p = o; p.cg = 1; p.tn = 2; p.nw = 12; p.rs = 2; add("tn2nw12rs2", test(p, dX, 0, 8, 0, 8)); p.nw = 24; p.rs = 0; add("tn2nw24", test(p, dX, 0, 16, 0, 16));
          p = o; p.cg = 1; p.tn = 4; p.nw = 8; add("tn4", test(p, dX, 0, 16, 0, 16)); p.nw = 4; p.rs = 1; add("tn4nw4M3rs1", test(p, dX, 9, 3, 9, 3)); p.r = 8; p.rs = 0; add("tn4r8M8", test(p, dX, 0, 8, 0, 8)); p.r = 4; p.nw = 12; add("tn4nw12", test(p, dX, 0, 16, 0, 16));
          p = o; p.cg = 1; p.tn = 2; p.nw = 8; p.gsel = 4; add("tn2G4M64", test(p, dX, 0, 64, 0, 16)); p.gsel = 2; p.rs = 1; add("tn2G2M32rs1", test(p, dX, 0, 32, 0, 16));
        }
      }
      if (N % 128 == 0) { Opt p = o; p.tile = 1; add("tile128x64/M64", test(p, dX, 0, 64, 0, 16)); add("tile/M5r9", test(p, dX, 9, 5, 9, 5)); p.tile = 2; add("tile128x128/M37", test(p, dX, 0, 37, 0, 16)); }
      {  // bf16 output == RNE(fp32 reference)
        CK(xcp_gemm(o, dX, K, dWp, dYb, N, 16, N, K, 1)); CK(cudaDeviceSynchronize());
        std::vector<uint16_t> yb((size_t)16 * N); CK(cudaMemcpy(yb.data(), dYb, yb.size() * 2, cudaMemcpyDeviceToHost));
        long d = 0; for (size_t i = 0; i < yb.size(); ++i) d += yb[i] != xcp_f2bf(ref[i]);
        add("bf16out", d);
        if (prog == 1 || prog == 5) { std::vector<float> f(yb.size()); for (size_t i = 0; i < f.size(); ++i) f[i] = bfv(yb[i]); (prog == 1 ? acc_bf : acc_bf21) = rel(f.data()); }
      }
      if (prog == 1 && ci < 2) {  // alternative canonical segment length (a different program: its own reference)
        std::vector<float> r16((size_t)R * N); xcp_reference(X.data(), K, W.data(), R, N, K, prog, 16, r16.data());
        Opt p = o; p.segb = 16; CK(xcp_gemm(p, dX, K, dW, dY, N, 16, N, K, 0)); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dY, (size_t)16 * N * 4, cudaMemcpyDeviceToHost));
        add("segb16", diff(got.data(), r16.data(), (size_t)16 * N));
      }
      long bad = 0, na = 0; std::string s;
      for (long v : mm) { bad += v > 0; na += v < 0; s += std::to_string(v) + " "; }
      bad += mm[0] < 0;  // the default configuration must run everywhere
      total_bad += bad + (hg != hc);
      printf("XCPHASH case %d N=%d K=%d kind=%d %s gpu=%016llx cpu=%016llx %s checks=%zu refused=%ld | mismatches [%s] (%s)\n", ci, N, K, c.kind, kXcpProgs[prog].name,
             (unsigned long long)hg, (unsigned long long)hc, hg == hc && !bad ? "OK" : "FAIL", mm.size(), na, s.c_str(), names.c_str());
      fflush(stdout);
    }
    double acc_nat = -1, acc_cb = -1; uint64_t hn = 0, hb = 0;
    {
      const cublasStatus_t st = cublas_gemm(dX, dW, dYb, 16, N, K); CK(cudaDeviceSynchronize());
      std::vector<uint16_t> yb((size_t)16 * N); CK(cudaMemcpy(yb.data(), dYb, yb.size() * 2, cudaMemcpyDeviceToHost));
      if (st == CUBLAS_STATUS_SUCCESS) { std::vector<float> f(yb.size()); for (size_t i = 0; i < f.size(); ++i) f[i] = bfv(yb[i]); acc_cb = rel(f.data()); hb = xcp_fnv(yb.data(), yb.size() * 2); }
    }
#ifdef WITH_PI
    if (N % 32 == 0) {
      CK(pi_small_fp32(dX, dW, dY, 16, N, K)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got.data(), dY, (size_t)16 * N * 4, cudaMemcpyDeviceToHost));
      acc_nat = rel(got.data()); hn = xcp_fnv(got.data(), (size_t)16 * N * 4);
    }
#endif
    printf("ACC case %d N=%d K=%d kind=%d rel-L2 vs fp64 of the original weights (16 rows): TRI %.3e TRIF %.3e P21=P21C %.3e P11=P11C %.3e | PI native fp32 %.3e | bf16 output: TRIF %.3e P21C %.3e cuBLAS %.3e | canonical weights moved %.2f%% (12-bit) %.2f%% (10-bit) of elements\n",
           ci, N, K, c.kind, acc[0], acc[1], acc[5], acc[6], acc_nat, acc_bf, acc_bf21, acc_cb, 100.0 * ch12 / W.size(), 100.0 * ch10 / W.size());
    printf("CLASSHASH case %d (class-dependent kernels, for contrast) PI-native-fp32=%016llx cuBLAS-bf16=%016llx\n", ci, (unsigned long long)hn, (unsigned long long)hb);
    fflush(stdout);
    cudaFree(dX); cudaFree(dX2); cudaFree(dW); cudaFree(dW12); cudaFree(dW10); cudaFree(dY); cudaFree(dYb);
  }
  printf("VERIFY %s: %ld failing checks, %ld reference step violations\n", total_bad || g_xcp_violations.load() ? "FAILED" : "PASSED", total_bad, g_xcp_violations.load());
  return total_bad != 0;
}

static void bench(const std::string& set) {
  struct Shape { int N, K; const char* name; };
  const std::vector<Shape> shapes = set == "quick" ? std::vector<Shape>{{4096, 2048, "1.7B qkv"}, {2048, 6144, "1.7B down"}}
      : set == "a" ? std::vector<Shape>{{4096, 2048, "1.7B qkv"}, {2048, 2048, "1.7B o"}, {12288, 2048, "1.7B gateup"}, {2048, 6144, "1.7B down"}}
                   : std::vector<Shape>{{6144, 4096, "8B qkv"}, {4096, 4096, "8B o"}, {24576, 4096, "8B gateup"}, {4096, 12288, "8B down"}};
  const std::vector<int> Ms = set == "quick" ? std::vector<int>{1, 16, 64, 256} : std::vector<int>{1, 8, 16, 32, 64, 256, 1024};
  struct Var { std::string grp, name; Opt o; int min_m, max_m; };
  std::vector<Var> vars;
  auto addv = [&](const char* grp, const std::string& name, Opt o, int lo, int hi) { vars.push_back({grp, name, o, lo, hi}); };
  const char* gname[7] = {"TRI", "TRIF", "P21", "P11", "SKEL", "P21C", "P11C"};
  auto vname = [](const Opt& o) { return std::string(o.cg ? "cg" : "ca") + std::to_string(o.nw) + "r" + std::to_string(o.r) + (o.tn > 1 ? "t" + std::to_string(o.tn) : "") + (o.rs == 0 ? "" : o.rs == 999 ? "local" : "rs" + std::to_string(o.rs)); };
  for (int prog : {1, 5}) {  // scheduling sweep (same bits within a program): {warps, cg, ring depth, tiles per warp} x {auto, rs1, local, rs2}
    const int cfg[13][4] = {{16, 0, 4, 1}, {16, 1, 4, 1}, {24, 1, 4, 1}, {32, 1, 4, 1}, {8, 1, 4, 2}, {12, 1, 4, 2}, {16, 1, 4, 2}, {24, 1, 4, 2}, {8, 1, 8, 2},
                            {4, 1, 4, 4}, {8, 1, 4, 4}, {12, 1, 4, 4}, {4, 1, 8, 4}};
    for (auto& q : cfg) {
      Opt o; o.prog = prog; o.nw = q[0]; o.cg = q[1]; o.r = q[2]; o.tn = q[3];
      for (int rs : {0, 1, 999, 2}) { if (rs == 2 && q[3] == 1) continue; o.rs = rs; addv(gname[prog], vname(o), o, 1, 16); }
    }
  }
  for (int prog : {6, 4}) { const int cfg[4][4] = {{16, 1, 4, 1}, {8, 1, 4, 2}, {16, 1, 4, 2}, {8, 1, 4, 4}}; for (auto& q : cfg) { Opt o; o.prog = prog; o.nw = q[0]; o.cg = q[1]; o.r = q[2]; o.tn = q[3]; addv(gname[prog], vname(o), o, 1, 16); o.rs = 1; addv(gname[prog], vname(o), o, 1, 16); } }
  for (int prog : {0, 2, 3}) { Opt o; o.prog = prog; addv(gname[prog], vname(o), o, 1, 16); }
  for (int prog : {1, 5, 6, 0, 2, 3, 4}) {  // M = 17 .. 64: G = 2 / 4 kernels; M >= 64: tile kernels
    Opt o; o.prog = prog; addv(gname[prog], "G", o, 17, 64); o.rs = 1; addv(gname[prog], "Grs1", o, 17, 64); o.rs = 999; addv(gname[prog], "Glocal", o, 17, 64); o.rs = 0;
    if (prog == 1 || prog == 5 || prog == 6 || prog == 4) {
      o.cg = 1; addv(gname[prog], "Gcg", o, 17, 64); o.rs = 1; addv(gname[prog], "Gcgrs1", o, 17, 64); o.rs = 999; addv(gname[prog], "Gcglocal", o, 17, 64);
      o.tn = 2; o.nw = 8; o.rs = 0; addv(gname[prog], "Gt2", o, 17, 64); o.rs = 1; addv(gname[prog], "Gt2rs1", o, 17, 64); o.rs = 999; addv(gname[prog], "Gt2local", o, 17, 64); o = Opt(); o.prog = prog;
    }
    o.tile = 1; addv(gname[prog], "tile128x64", o, 64, 1 << 30); o.tile = 2; addv(gname[prog], "tile128x128", o, 64, 1 << 30);
  }
  printf("# us per GEMM, bf16 output, launches back to back. HOT: one weight buffer (sits in L2 when it fits). COLD: weight copies (>= 300 MB in total) cycle,\n");
  printf("# so every launch streams its weight from DRAM, as a decode step does. native = best PI-Attn kernel id for this (shape, M, regime).\n");
  printf("# per program: best scheduling (same bits). TRI / TRIF / P21 slice W on the fly; P21C / P11C assume canonical weights; SKEL = class-dependent chained MMA in the same skeleton.\n");
  for (const Shape& sh : shapes) {
    const int N = sh.N, K = sh.K, MX = 1024;
    std::vector<uint16_t> X, W; xcp_fill(X, 0, 0, 5, MX, K); xcp_fill(W, 1, 0, 6, N, K);
    const size_t wb = W.size() * 2;
    const int copies = (int)std::max<size_t>(2, (((size_t)300 << 20) + wb - 1) / wb);
    uint16_t *dX, *dY; std::vector<uint16_t*> dW(copies);
    CK(cudaMalloc(&dX, X.size() * 2)); CK(cudaMalloc(&dY, (size_t)MX * N * 2));
    CK(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice));
    for (int i = 0; i < copies; ++i) { CK(cudaMalloc(&dW[i], wb)); if (i == 0) CK(cudaMemcpy(dW[0], W.data(), wb, cudaMemcpyHostToDevice)); else CK(cudaMemcpy(dW[i], dW[0], wb, cudaMemcpyDeviceToDevice)); }
    for (int M : Ms) {
      const bool cold = M <= 64;
      double h_cb = -1, c_cb = -1, h_nat = -1, c_nat = -1; int hid = -1, cid = -1;
      if (cublas_gemm(dX, dW[0], dY, M, N, K) == CUBLAS_STATUS_SUCCESS) {
        h_cb = time_us([&](int) { cublas_gemm(dX, dW[0], dY, M, N, K); }, 12, 3);
        if (cold) c_cb = time_us([&](int i) { cublas_gemm(dX, dW[i % copies], dY, M, N, K); }, 12, 3);
      }
#ifdef WITH_PI
      std::vector<int> ids;
      if (M <= 64) ids = pi_gemm::small_kernels();
      ids.insert(ids.end(), pi_gemm::large_kernels().begin(), pi_gemm::large_kernels().end());
      for (int id : ids) {
        if (pi_run(id, dX, dW[0], dY, M, N, K) != cudaSuccess) continue;
        CK(cudaDeviceSynchronize());
        const double t = time_us([&](int) { pi_run(id, dX, dW[0], dY, M, N, K); }, 4, 2);
        if (h_nat < 0 || t < h_nat) { h_nat = t; hid = id; }
        if (cold) { const double tc = time_us([&](int i) { pi_run(id, dX, dW[i % copies], dY, M, N, K); }, 4, 2); if (c_nat < 0 || tc < c_nat) { c_nat = tc; cid = id; } }
      }
      if (hid >= 0) h_nat = time_us([&](int) { pi_run(hid, dX, dW[0], dY, M, N, K); }, 12, 3);
      if (cid >= 0) c_nat = time_us([&](int i) { pi_run(cid, dX, dW[i % copies], dY, M, N, K); }, 12, 3);
#endif
      std::vector<double> th(vars.size(), -1), tc(vars.size(), -1);
      for (size_t i = 0; i < vars.size(); ++i) {
        const Var& v = vars[i];
        if (M < v.min_m || M > v.max_m || xcp_gemm(v.o, dX, K, dW[0], dY, N, M, N, K, 1) != cudaSuccess) { cudaGetLastError(); continue; }
        CK(cudaDeviceSynchronize());
        th[i] = time_us([&](int) { xcp_gemm(v.o, dX, K, dW[0], dY, N, M, N, K, 1); }, 5, 2);
        if (cold) tc[i] = time_us([&](int j) { xcp_gemm(v.o, dX, K, dW[j % copies], dY, N, M, N, K, 1); }, 5, 2);
      }
      for (int regime = 0; regime < (cold ? 2 : 1); ++regime) {
        const std::vector<double>& t = regime ? tc : th; const double cb = regime ? c_cb : h_cb, nat = regime ? c_nat : h_nat;
        printf("BENCH %-11s N=%5d K=%5d M=%4d %s cuBLAS %8.1f native %8.1f (k%d) |", sh.name, N, K, M, regime ? "COLD" : "HOT ", cb, nat, regime ? cid : hid);
        for (const char* g : {"TRIF", "P21C", "P11C", "TRI", "P21", "P11", "SKEL"}) {
          double best = -1; int bi = -1;
          for (size_t i = 0; i < vars.size(); ++i) if (vars[i].grp == g && t[i] > 0 && (best < 0 || t[i] < best)) { best = t[i]; bi = (int)i; }
          if (bi >= 0) printf(" %s %.1f (%s) x%.2f nat x%.2f cb |", g, best, vars[bi].name.c_str(), nat > 0 ? best / nat : -1, cb > 0 ? best / cb : -1);
        }
        printf(" ALL:");
        for (size_t i = 0; i < vars.size(); ++i) if (t[i] > 0) printf(" %s/%s %.1f", vars[i].grp.c_str(), vars[i].name.c_str(), t[i]);
        printf("\n");
      }
      fflush(stdout);
    }
    cudaFree(dX); cudaFree(dY); for (auto p : dW) cudaFree(p);
  }
}

int main(int argc, char** argv) {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  CK(xcp::device_init(g_dev));
  if (cublasCreate(&g_cb) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 2; }
#ifdef WITH_PI
  g_pidev.sms = g_dev.sms; g_pidev.smem_optin = g_dev.smem_max;
#endif
  printf("GPU %s sm_%d%d SMs %d smem-optin %d\n", p.name, p.major, p.minor, g_dev.sms, g_dev.smem_max);
  const std::string mode = argc > 1 ? argv[1] : "verify";
  if (mode == "verify") { std::vector<int> w; for (int i = 2; i < argc; ++i) w.push_back(atoi(argv[i])); return verify(w); }
  if (mode == "bench") { bench(argc > 2 ? argv[2] : "a"); return 0; }
  printf("usage: xcp verify [case ...] | bench [quick|a|b]\n");
  return 1;
}
