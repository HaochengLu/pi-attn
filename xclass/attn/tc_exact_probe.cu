// tc_exact_probe.cu -- where does a bf16 -> fp32 tensor-core step stop being exact?
// Each case is d = c + sum over T k16 steps of a_k b_k with integer-valued bf16 operands (8- or 4-bit
// mantissas times shared per-k powers of two) and integer c, built so that |c| + sum |a_k b_k| < 2^W.
// For W <= 24 every partial sum is an integer below 2^24: if the tensor core never truncates such a sum,
// the result equals the exact integer on every GPU class (errors == 0 and equal hashes everywhere).
// For W >= 25 the result is not representable and the hash is expected to depend on the class.
// Also: exhaustive hash of ex2.approx.ftz.f32 over all 2^32 inputs (is the softmax exponential
// class-independent?).
//   nvcc -O2 -std=c++17 -arch=sm_80 tc_exact_probe.cu -o p && ./p        (any GPU, mma.sync)
//   nvcc -O2 -std=c++17 -arch=sm_90a -DWGMMA tc_exact_probe.cu -o p && ./p   (H100: adds wgmma)
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__);          \
      exit(2);                                                                         \
    }                                                                                  \
  } while (0)

__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) {
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                      uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                      uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// X [tiles][T][64][16], W [tiles][T][8][16], C [tiles][64][8], out [tiles][64][8].
// One block of 128 threads per tile; warp w owns rows 16 w .. 16 w + 15.
__global__ void k_mma(const uint16_t* X, const uint16_t* W, const float* C, float* out, int T,
                      int mode) {
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, k0 = (lane & 3) * 2;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const float* c = C + (size_t)tile * 512;
  float d0 = c[r0 * 8 + k0], d1 = c[r0 * 8 + k0 + 1], d2 = c[r1 * 8 + k0], d3 = c[r1 * 8 + k0 + 1];
  for (int t = 0; t < T; ++t) {
    const uint16_t* x = X + ((size_t)tile * T + t) * 1024;
    const uint16_t* w = W + ((size_t)tile * T + t) * 128;
    const uint32_t a0 = pk(x[r0 * 16 + k0], x[r0 * 16 + k0 + 1]);
    const uint32_t a1 = pk(x[r1 * 16 + k0], x[r1 * 16 + k0 + 1]);
    const uint32_t a2 = pk(x[r0 * 16 + k0 + 8], x[r0 * 16 + k0 + 9]);
    const uint32_t a3 = pk(x[r1 * 16 + k0 + 8], x[r1 * 16 + k0 + 9]);
    const uint32_t b0 = pk(w[g * 16 + k0], w[g * 16 + k0 + 1]);
    const uint32_t b1 = pk(w[g * 16 + k0 + 8], w[g * 16 + k0 + 9]);
    if (mode == 0) {
      mma16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    } else {
      mma16(d0, d1, d2, d3, a0, a1, 0u, 0u, b0, 0u);
      mma16(d0, d1, d2, d3, 0u, 0u, a2, a3, 0u, b1);
    }
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = d0;
  o[r0 * 8 + k0 + 1] = d1;
  o[r1 * 8 + k0] = d2;
  o[r1 * 8 + k0 + 1] = d3;
}

#ifdef WGMMA
// K-major, no swizzle: 8-row x 16-byte core matrices. Element (r, k) of an operand lives at
// (r / 8) * sbo + (k / 8) * lbo + (r % 8) * 16 + (k % 8) * 2 bytes from the operand start.
__device__ __forceinline__ int smem_off(int r, int k, int lbo, int sbo) {
  return (r >> 3) * sbo + (k >> 3) * lbo + (r & 7) * 16 + (k & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc(const void* p, int lbo, int sbo, int swap) {
  const uint64_t addr = (uint64_t)__cvta_generic_to_shared(p);
  const uint64_t l = (uint64_t)((swap ? sbo : lbo) >> 4) & 0x3FFF;
  const uint64_t s = (uint64_t)((swap ? lbo : sbo) >> 4) & 0x3FFF;
  return ((addr >> 4) & 0x3FFF) | (l << 16) | (s << 32);  // bits 62..63 = 0: no swizzle
}

__global__ void __launch_bounds__(128)
    k_wgmma(const uint16_t* X, const uint16_t* W, const float* C, float* out, int T, int lbo,
            int sbo, int swap) {
  __shared__ __align__(128) unsigned char sA[2048];
  __shared__ __align__(128) unsigned char sB[2048];
  const int tile = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, k0 = (lane & 3) * 2;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const float* c = C + (size_t)tile * 512;
  float d0 = c[r0 * 8 + k0], d1 = c[r0 * 8 + k0 + 1], d2 = c[r1 * 8 + k0], d3 = c[r1 * 8 + k0 + 1];
  const uint64_t da = make_desc(sA, lbo, sbo, swap), db = make_desc(sB, lbo, sbo, swap);
  for (int t = 0; t < T; ++t) {
    const uint16_t* x = X + ((size_t)tile * T + t) * 1024;
    const uint16_t* w = W + ((size_t)tile * T + t) * 128;
    for (int i = threadIdx.x; i < 1024; i += 128)
      *(uint16_t*)(sA + smem_off(i >> 4, i & 15, lbo, sbo)) = x[i];
    for (int i = threadIdx.x; i < 128; i += 128)
      *(uint16_t*)(sB + smem_off(i >> 4, i & 15, lbo, sbo)) = w[i];
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");  // own smem writes ordered before the barrier
    __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, "
        "0, 0;\n"
        "}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(da), "l"(db), "r"(1));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    __syncthreads();
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = d0;
  o[r0 * 8 + k0 + 1] = d1;
  o[r1 * 8 + k0] = d2;
  o[r1 * 8 + k0 + 1] = d3;
}
#endif


__global__ void k_ex2(unsigned long long* acc) {
  const uint32_t hi = blockIdx.x * 1024u + threadIdx.x;  // 2^20 threads x 4096 inputs
  unsigned long long h = 0;
  for (uint32_t lo = 0; lo < 4096u; ++lo) {
    const uint32_t x = (hi << 12) | lo;
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(__uint_as_float(x)));
    unsigned long long z = ((unsigned long long)x << 32 | __float_as_uint(y)) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 29)) * 0xBF58476D1CE4E5B9ull;
    h += z ^ (z >> 32);
  }
  atomicAdd(acc, h);
}

// ------------------------------------------------------------------ host ----
static uint64_t g_state;
static uint64_t rnd() {
  uint64_t z = (g_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
static int rint_in(int lo, int hi) { return lo + (int)(rnd() % (uint64_t)(hi - lo + 1)); }
static uint16_t f2bf_exact(float f) {  // f must be bf16-representable
  uint32_t u;
  memcpy(&u, &f, 4);
  if (u & 0xFFFF) { printf("generator bug: %g not bf16\n", f); exit(3); }
  return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static uint64_t fnv(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  const unsigned char* p = (const unsigned char*)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
struct Data {
  int tiles, T;
  std::vector<uint16_t> X, W;
  std::vector<float> C;
};
// |c| < 2^(Wb-1), sum over all steps and k of |a b| < 2^(Wb-1).
static Data make(int Wb, int T, int tiles) {
  Data d;
  g_state = 0xABC000 + Wb * 100 + T;
  d.tiles = tiles;
  d.T = T;
  d.X.resize((size_t)tiles * T * 1024);
  d.W.resize((size_t)tiles * T * 128);
  d.C.assign((size_t)tiles * 512, 0.f);
  int lg = 4;  // log2(16 T)
  for (int t = T; t > 1; t >>= 1) ++lg;
  const int twmax = Wb - 1 - lg;  // per-term bit budget when all terms are equal
  const int mbits = twmax >= 16 ? 8 : twmax / 2;
  if (mbits < 1) { printf("W too small\n"); exit(3); }
  const int mmax = (1 << mbits) - 1, pb = 2 * mbits;
  for (int tl = 0; tl < tiles; ++tl) {
    const int style = tl & 3;
    const int big_step = rint_in(0, T - 1), big_k = rint_in(0, 15);
    for (int q = 0; q < T; ++q) {
      uint16_t* x = &d.X[((size_t)tl * T + q) * 1024];
      uint16_t* w = &d.W[((size_t)tl * T + q) * 128];
      for (int k = 0; k < 16; ++k) {
        int tw;
        if (style == 0 || style == 3) tw = twmax;
        else if (style == 1) tw = rint_in(pb, twmax);
        else tw = (q == big_step && k == big_k) ? Wb - 2 : rint_in(pb, twmax);
        const int sh = tw - pb, sa = rint_in(0, sh), sb = sh - sa;
        for (int r = 0; r < 64; ++r) {
          const int m = style == 3 ? mmax : rint_in(0, mmax);
          const float v = (float)ldexp((double)m, sa) * ((style != 3 && (rnd() & 1)) ? -1.f : 1.f);
          x[r * 16 + k] = f2bf_exact(v);
        }
        for (int n = 0; n < 8; ++n) {
          const int m = style == 3 ? mmax : rint_in(0, mmax);
          const float v = (float)ldexp((double)m, sb) * ((style != 3 && (rnd() & 1)) ? -1.f : 1.f);
          w[n * 16 + k] = f2bf_exact(v);
        }
      }
    }
    for (int i = 0; i < 512; ++i) {
      int64_t c;
      if (style == 3) c = ((int64_t)1 << (Wb - 1)) - 1;
      else c = (int64_t)(rnd() % ((uint64_t)1 << (Wb - 1))) >> rint_in(style == 2 ? 1 : 0, Wb - 2);
      if (style != 3 && (rnd() & 1)) c = -c;
      d.C[(size_t)tl * 512 + i] = (float)c;  // rounds when c has more than 24 bits (W >= 26)
    }
  }
  return d;
}
static void check(const Data& d, const std::vector<float>& out, size_t* inexact, size_t* negzero) {
  size_t bad = 0, nz = 0;
  for (int t = 0; t < d.tiles; ++t)
    for (int r = 0; r < 64; ++r)
      for (int n = 0; n < 8; ++n) {
        double s = d.C[(size_t)t * 512 + r * 8 + n];
        for (int q = 0; q < d.T; ++q)
          for (int k = 0; k < 16; ++k)
            s += (double)bf2f(d.X[((size_t)t * d.T + q) * 1024 + r * 16 + k]) *
                 (double)bf2f(d.W[((size_t)t * d.T + q) * 128 + n * 16 + k]);
        const float o = out[(size_t)t * 512 + r * 8 + n];
        uint32_t u;
        memcpy(&u, &o, 4);
        bad += (double)o != s;
        nz += u == 0x80000000u;
      }
  *inexact = bad;
  *negzero = nz;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s sm_%d%d\n", p.name, p.major, p.minor);
  {
    unsigned long long* acc;
    CK(cudaMalloc(&acc, 8));
    CK(cudaMemset(acc, 0, 8));
    k_ex2<<<1024, 1024>>>(acc);
    CK(cudaDeviceSynchronize());
    unsigned long long h;
    CK(cudaMemcpy(&h, acc, 8, cudaMemcpyDeviceToHost));
    printf("ex2.approx.ftz.f32 exhaustive (2^32 inputs) hash %016llx\n", h);
  }
  int wg_lbo = 0, wg_sbo = 0, wg_swap = 0, wg_ok = 0;
  const int Ts[2] = {1, 8};
  for (int ti = 0; ti < 2; ++ti)
    for (int Wb = 20; Wb <= 27; ++Wb) {
      const int T = Ts[ti], tiles = T == 1 ? 2048 : 512;
      Data d = make(Wb, T, tiles);
      const size_t nout = (size_t)d.tiles * 512;
      uint16_t *dX, *dW;
      float *dC, *dO;
      CK(cudaMalloc(&dX, d.X.size() * 2));
      CK(cudaMalloc(&dW, d.W.size() * 2));
      CK(cudaMalloc(&dC, nout * 4));
      CK(cudaMalloc(&dO, nout * 4));
      CK(cudaMemcpy(dX, d.X.data(), d.X.size() * 2, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dW, d.W.data(), d.W.size() * 2, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dC, d.C.data(), nout * 4, cudaMemcpyHostToDevice));
      std::vector<float> full(nout);
      k_mma<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(full.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
      size_t bad, nz;
      check(d, full, &bad, &nz);
      printf("W=%2d T=%d cases %7zu | mma.sync: not-exact %7zu  -0 %zu  hash %016llx", Wb, T, nout, bad, nz,
             (unsigned long long)fnv(full));
#ifdef WGMMA
      std::vector<float> wg(nout);
      if (!wg_ok) {
        const int cand[4][3] = {{128, 256, 0}, {1024, 128, 0}, {128, 256, 1}, {1024, 128, 1}};
        for (int v = 0; v < 2 && !wg_ok; ++v) {
          printf("\n  try wgmma layout lbo=%d sbo=%d swap=%d ...", cand[v][0], cand[v][1], cand[v][2]);
          fflush(stdout);
          k_wgmma<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, cand[v][0], cand[v][1], cand[v][2]);
          CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(wg.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
          size_t e, z;
          check(d, wg, &e, &z);
          printf(" not-exact %zu\n", e);
          {
            int shown = 0, by_style[4] = {0, 0, 0, 0}, by_r[64] = {0}, by_n[8] = {0}, vs_full = 0;
            for (int t = 0; t < d.tiles; ++t)
              for (int r = 0; r < 64; ++r)
                for (int n = 0; n < 8; ++n) {
                  double s = d.C[(size_t)t * 512 + r * 8 + n];
                  for (int k = 0; k < 16; ++k)
                    s += (double)bf2f(d.X[(size_t)t * 1024 + r * 16 + k]) * (double)bf2f(d.W[(size_t)t * 128 + n * 16 + k]);
                  const float o = wg[(size_t)t * 512 + r * 8 + n];
                  if ((double)o == s) continue;
                  ++by_style[t & 3]; ++by_r[r]; ++by_n[n];
                  vs_full += o != full[(size_t)t * 512 + r * 8 + n];
                  if (shown++ < 10) printf("    tile %d r %d n %d: exact %.1f wgmma %.1f c %.1f\n", t, r, n, s, (double)o, (double)d.C[(size_t)t * 512 + r * 8 + n]);
                }
            printf("    by style %d %d %d %d | differing from mma.sync %d | rows:", by_style[0], by_style[1], by_style[2], by_style[3], vs_full);
            for (int r = 0; r < 64; ++r) if (by_r[r]) printf(" %d:%d", r, by_r[r]);
            printf(" | cols:");
            for (int n = 0; n < 8; ++n) printf(" %d", by_n[n]);
            printf("\n");
          }
          if (e == 0) {
            wg_ok = 1;
            wg_lbo = cand[v][0];
            wg_sbo = cand[v][1];
            wg_swap = cand[v][2];
          }
        }
      }
      if (wg_ok) {
        k_wgmma<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, wg_lbo, wg_sbo, wg_swap);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(wg.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
        size_t e, z, mm = 0;
        check(d, wg, &e, &z);
        for (size_t i = 0; i < nout; ++i) mm += memcmp(&wg[i], &full[i], 4) != 0;
        printf(" | wgmma: not-exact %7zu  vs mma.sync %7zu  hash %016llx", e, mm, (unsigned long long)fnv(wg));
      } else {
        printf(" | wgmma: no layout gave exact results");
      }
#endif
      printf("\n");
      cudaFree(dX);
      cudaFree(dW);
      cudaFree(dC);
      cudaFree(dO);
    }
  return 0;
}
