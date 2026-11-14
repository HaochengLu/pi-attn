// tc_class_probe.cu -- which tensor-core "arithmetic class" is this GPU in, and is a bf16->fp32
// k16 step bit-identical between mma.sync and wgmma on the same GPU?
//
// Build + run (Ampere / Ada / any GPU, mma.sync only):
//   nvcc -O2 -std=c++17 -arch=sm_80 tc_class_probe.cu -o tc_class_probe && ./tc_class_probe
// Build + run on H100 (adds the wgmma path):
//   nvcc -O2 -std=c++17 -arch=sm_90a -DWGMMA tc_class_probe.cu -o tc_class_probe && ./tc_class_probe
// B200: -arch=sm_100a (try -DWGMMA; drop it if ptxas rejects wgmma for that target).
//
// What it does. Each test case is one dot product with accumulator, d = c + sum_{k<16} a_k b_k,
// chained over T k16 steps. A tile is X[64][16], W[8][16], C[64][8] = 512 independent cases.
//   path FULL  : one mma.sync.m16n8k16 per step (the PI-Attn canonical step)
//   path SPLIT8: two mma.sync per step, k 0..7 then k 8..15, the other half zero-padded
//   path WGMMA : one wgmma.mma_async.m64n8k16 per step (only with -DWGMMA)
// Datasets: D0 small integers (every result exact: validates fragment / smem layouts),
//           D1 Gaussian activations x N(0,1), w N(0,0.02), chain of 64 steps from zero (K = 1024),
//           D2 wide exponent spread, random accumulator, 1 step,
//           D3 exact cancellation of two large products plus terms 2^-10..2^-28 below them, 1 step
//              (sensitive to the fused-group size L and to the alignment width F).
//
// Pass / fail criteria
//   C0  D0 must be exact on every path. If not, the probe itself is wrong: ignore the rest.
//   H1  (H100, -DWGMMA) "WGMMA vs FULL mismatches" == 0 on D1, D2, D3
//         => wgmma executes the canonical k16 step with mma.sync's bits on Hopper.
//   H2  the "hash FULL" lines are identical on A10G, L4, L40S, A100 (class A: L=8, F=24) and
//       identical on H100, B200 (class H: L=16, F=25), and differ between the two groups.
//         Literature prediction (MMA-Sim arXiv 2511.10909, Hawkeye arXiv 2603.20421). If H100's
//         hashes equal A100's instead, cross-class identity is free and the prediction is wrong.
//   H3  "SPLIT8 vs FULL mismatches": expected 0 on class A if a k16 step is two chained k8 fused
//       sums with an fp32 result in between (hypothesis; if > 0 the intermediate is kept wider),
//       expected > 0 on class H on D2 / D3 (one 16-way fused sum).
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

#if 1
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
    for (auto& v : d.X) v = bf_bits(rnd() & 1, rint_in(-10, 10), (int)(rnd() & 127));
    for (auto& v : d.W) v = bf_bits(rnd() & 1, rint_in(-10, 10), (int)(rnd() & 127));
    for (auto& v : d.C) {
      const uint32_t u = ((uint32_t)(rnd() & 1) << 31) | ((uint32_t)(127 + rint_in(-10, 10)) << 23) |
                         (uint32_t)(rnd() & 0x7FFFFF);
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
          x[r * 16 + k] = bf_bits(rnd() & 1, e1 - rint_in(5, 14), (int)(rnd() & 127));
      for (int n = 0; n < 8; ++n)
        for (int k = 0; k < 16; ++k)
          w[n * 16 + k] = bf_bits(rnd() & 1, e2 - rint_in(5, 14), (int)(rnd() & 127));
      for (int r = 0; r < 64; ++r) {  // a[k2] b[k2] = -a[k1] b[k1] exactly, for every (r, n)
        const int s = rnd() & 1, m = (int)(rnd() & 127);
        x[r * 16 + k1] = bf_bits(s, e1, m);
        x[r * 16 + k2] = bf_bits(s ^ 1, e1, m);
      }
      for (int n = 0; n < 8; ++n) {
        const uint16_t b = bf_bits(rnd() & 1, e2, (int)(rnd() & 127));
        w[n * 16 + k1] = b;
        w[n * 16 + k2] = b;
      }
      if (t & 1)
        for (int i = 0; i < 512; ++i) {
          const uint32_t u = ((uint32_t)(rnd() & 1) << 31) |
                             ((uint32_t)(127 + e1 + e2 - rint_in(8, 26)) << 23) |
                             (uint32_t)(rnd() & 0x7FFFFF);
          memcpy(&d.C[(size_t)t * 512 + i], &u, 4);
        }
    }
  }
  return d;
}

static float bf2f(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static size_t exact_errors(const Data& d, const std::vector<float>& out) {  // dataset 0 only
  size_t bad = 0;
  for (int t = 0; t < d.tiles; ++t)
    for (int r = 0; r < 64; ++r)
      for (int n = 0; n < 8; ++n) {
        double s = d.C[(size_t)t * 512 + r * 8 + n];
        for (int q = 0; q < d.T; ++q)
          for (int k = 0; k < 16; ++k)
            s += (double)bf2f(d.X[((size_t)t * d.T + q) * 1024 + r * 16 + k]) *
                 (double)bf2f(d.W[((size_t)t * d.T + q) * 128 + n * 16 + k]);
        bad += (double)out[(size_t)t * 512 + r * 8 + n] != s;
      }
  return bad;
}


// variant bits: 1 = second barrier before the fence, 2 = B stored before A, 4 = no trailing barrier,
//               8 = 32-bit smem stores, 16 = wgmma.fence executed by the storing warp BEFORE the barrier
__global__ void __launch_bounds__(128)
    k_wg2(const uint16_t* X, const uint16_t* W, const float* C, float* out, int T, int lbo, int sbo, int variant, int tile0) {
  __shared__ __align__(128) unsigned char sA[2048];
  __shared__ __align__(128) unsigned char sB[2048];
  const int tile = blockIdx.x + tile0, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, k0 = (lane & 3) * 2;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const float* c = C + (size_t)tile * 512;
  float d0 = c[r0 * 8 + k0], d1 = c[r0 * 8 + k0 + 1], d2 = c[r1 * 8 + k0], d3 = c[r1 * 8 + k0 + 1];
  const uint64_t da = make_desc(sA, lbo, sbo, 0), db = make_desc(sB, lbo, sbo, 0);
  for (int t = 0; t < T; ++t) {
    const uint16_t* x = X + ((size_t)tile * T + t) * 1024;
    const uint16_t* w = W + ((size_t)tile * T + t) * 128;
    if (variant & 8) {
      for (int i = threadIdx.x; i < 512; i += 128) *(uint32_t*)(sA + smem_off(i >> 3, (i & 7) * 2, lbo, sbo)) = pk(x[2 * i], x[2 * i + 1]);
      if (threadIdx.x < 64) { const int i = threadIdx.x; *(uint32_t*)(sB + smem_off(i >> 3, (i & 7) * 2, lbo, sbo)) = pk(w[2 * i], w[2 * i + 1]); }
    } else {
      if (variant & 2) for (int i = threadIdx.x; i < 128; i += 128) *(uint16_t*)(sB + smem_off(i >> 4, i & 15, lbo, sbo)) = w[i];
      for (int i = threadIdx.x; i < 1024; i += 128) *(uint16_t*)(sA + smem_off(i >> 4, i & 15, lbo, sbo)) = x[i];
      if (!(variant & 2)) for (int i = threadIdx.x; i < 128; i += 128) *(uint16_t*)(sB + smem_off(i >> 4, i & 15, lbo, sbo)) = w[i];
    }
    if (variant & 16) asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    __syncthreads();
    if (variant & 1) __syncthreads();
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db), "r"(1));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
    if (!(variant & 4)) __syncthreads();
  }
  float* o = out + (size_t)tile * 512;
  o[r0 * 8 + k0] = d0; o[r0 * 8 + k0 + 1] = d1; o[r1 * 8 + k0] = d2; o[r1 * 8 + k0 + 1] = d3;
}
static double dot(const Data& d, int ta, int r, int tb, int n, int klo, int khi) {
  double s = 0;
  for (int k = klo; k < khi; ++k) s += (double)bf2f(d.X[(size_t)ta * 1024 + r * 16 + k]) * (double)bf2f(d.W[(size_t)tb * 128 + n * 16 + k]);
  return s;
}
int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s sm_%d%d  SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
  Data d = make(2);
  const size_t nout = (size_t)d.tiles * 512;
  uint16_t *dX, *dW; float *dC, *dO;
  CK(cudaMalloc(&dX, d.X.size() * 2)); CK(cudaMalloc(&dW, d.W.size() * 2)); CK(cudaMalloc(&dC, nout * 4)); CK(cudaMalloc(&dO, nout * 4));
  CK(cudaMemcpy(dX, d.X.data(), d.X.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dW, d.W.data(), d.W.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, d.C.data(), nout * 4, cudaMemcpyHostToDevice));
  std::vector<float> full(nout), wg(nout);
  k_mma<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, 0); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(full.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
  printf("D2 hash FULL %016llx\n", (unsigned long long)fnv(full));
  const int variants[] = {8, 24, 101, 0, 16, 8, 24};  // 100: chunks of 128 blocks, 101: orig kernel, 102: chunk 1024
  for (int v : variants) {
    size_t total = 0; int explained_B = 0, explained_A = 0, looked = 0;
    for (int rep = 0; rep < 40; ++rep) {
      CK(cudaMemset(dO, 0, nout * 4));
      if (v == 101) k_wgmma<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, 128, 256, 0);
      else if (v >= 100) { const int ch = v == 100 ? 128 : 1024; for (int t0 = 0; t0 < d.tiles; t0 += ch) { k_wg2<<<ch, 128>>>(dX, dW, dC, dO, d.T, 128, 256, 0, t0); CK(cudaDeviceSynchronize()); } }
      else k_wg2<<<d.tiles, 128>>>(dX, dW, dC, dO, d.T, 128, 256, v, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(wg.data(), dO, nout * 4, cudaMemcpyDeviceToHost));
      total += mismatches(full, wg);
      int lastkey = -1;
      for (size_t i = 0; i < nout && looked < 6; ++i) if (memcmp(&full[i], &wg[i], 4)) {
        const int tile = (int)(i / 512), r = (int)(i % 512) / 8, n = (int)(i % 8);
        if (tile * 8 + n / 2 == lastkey) continue;
        lastkey = tile * 8 + n / 2; ++looked;
        // count mismatching (r, n) pattern in this tile
        int cnt = 0, colmask = 0; for (int q = 0; q < 512; ++q) if (memcmp(&full[(size_t)tile * 512 + q], &wg[(size_t)tile * 512 + q], 4)) { ++cnt; colmask |= 1 << (q % 8); }
        // stale-B search: wg ~= c + A[tile][r] . B[t'][n] (whole column, or only one k half stale)
        int foundB = -1, kind = -1, foundA = -1;
        for (int tp = 0; tp < d.tiles && foundB < 0; ++tp) if (tp != tile) {
          const double c0 = d.C[i];
          const double cand[3] = {c0 + dot(d, tile, r, tp, n, 0, 16), c0 + dot(d, tile, r, tp, n, 0, 8) + dot(d, tile, r, tile, n, 8, 16),
                                  c0 + dot(d, tile, r, tile, n, 0, 8) + dot(d, tile, r, tp, n, 8, 16)};
          for (int q = 0; q < 3; ++q) if (fabs(cand[q] - wg[i]) <= 1e-5 * fabs(wg[i]) + 1e-30) { foundB = tp; kind = q; }
          if (fabs(c0 + dot(d, tp, r, tile, n, 0, 16) - wg[i]) <= 1e-5 * fabs(wg[i]) + 1e-30) foundA = tp;
        }
        explained_B += foundB >= 0; explained_A += foundA >= 0;
        printf("    v%d rep %d tile %d: %d wrong, colmask %02x, r %d n %d full %.9g wg %.9g | stale-B source tile %d (kind %d) stale-A source %d\n",
               v, rep, tile, cnt, colmask, r, n, full[i], wg[i], foundB, kind, foundA);
      }
    }
    printf("variant %3d: total mismatches over 40 reps %zu (looked %d, stale-B explained %d, stale-A %d)\n", v, total, looked, explained_B, explained_A);
  }
  return 0;
}
