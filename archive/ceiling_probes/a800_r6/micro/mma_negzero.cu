// Can mma.sync m16n8k16 (bf16 x bf16 + f32) return -0 from an accumulator that is not -0?
// Directed: every pair of special values (one product, or all 16 products of an element equal), accumulator +0.
// Random: A and B from raw bit patterns restricted to small magnitudes (products underflow), 30% special values;
// accumulator +0, or a random float that is not -0. Counts outputs equal to 0x80000000.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ uint32_t rng(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
// mode 0: directed, all 16 products = va * vb; mode 1: directed, one product va * vb, the rest +0 * +0
// mode 2: random small magnitudes, acc +0; mode 3: random, acc random non -0; mode 4: as 2 with acc = -0 (control)
__global__ void k(int mode, const uint16_t* sp, int nsp, uint32_t seed, unsigned long long* cnt) {
  uint32_t s = seed * 2654435761u + (blockIdx.x * blockDim.x + threadIdx.x) * 40503u + 1;
  rng(s); rng(s);
  const int lane = threadIdx.x & 31;
  uint32_t a[4] = {0, 0, 0, 0}, b0 = 0, b1 = 0;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  if (mode <= 1) {
    const int idx = blockIdx.x;  // one (va, vb) pair per block, one warp
    const uint32_t va = sp[idx / nsp], vb = sp[idx % nsp];
    if (mode == 0) { for (int i = 0; i < 4; ++i) a[i] = va | (va << 16); b0 = b1 = vb | (vb << 16); }
    else { if ((lane & 3) == 0) a[0] = va; if ((lane & 3) == 0) b0 = vb; }
  } else {
    auto val = [&]() -> uint32_t {
      const uint32_t r = rng(s);
      if ((r & 15) < 5) return sp[(r >> 8) % nsp];
      const uint32_t expo = (r >> 4) % 60;  // exponent field 0..59: magnitudes below 2^-67
      return ((r >> 31) << 15) | (expo << 7) | ((r >> 12) & 0x7f);
    };
    for (int i = 0; i < 4; ++i) a[i] = val() | (val() << 16);
    b0 = val() | (val() << 16); b1 = val() | (val() << 16);
    if (mode == 3) for (int i = 0; i < 4; ++i) { uint32_t r = rng(s); if (r == 0x80000000u) r = 0; d[i] = (r & 7) < 3 ? 0.f : __uint_as_float(r); }
    if (mode == 4) for (int i = 0; i < 4; ++i) d[i] = -0.f;
  }
  mma(d, a, b0, b1);
  unsigned long long neg0 = 0, tot = 4;
  for (int i = 0; i < 4; ++i) neg0 += __float_as_uint(d[i]) == 0x80000000u;
  atomicAdd(cnt, neg0); atomicAdd(cnt + 1, tot);
}
int main() {
  std::vector<uint16_t> sp;
  for (uint16_t sign : {0, 0x8000}) for (uint16_t m : {0x0000, 0x0001, 0x007f, 0x0080, 0x0081, 0x00ff, 0x0100, 0x1000, 0x2000, 0x3f80, 0x4000, 0x7f7f, 0x7f80, 0x7fc0}) sp.push_back(sign | m);
  const int nsp = sp.size();
  uint16_t* dsp; cudaMalloc(&dsp, nsp * 2); cudaMemcpy(dsp, sp.data(), nsp * 2, cudaMemcpyHostToDevice);
  unsigned long long* cnt; cudaMalloc(&cnt, 16);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  const char* names[] = {"directed, 16 equal products, acc +0", "directed, one product, acc +0", "random small/special, acc +0", "random small/special, acc random (not -0)", "control: random small/special, acc -0"};
  unsigned long long sum = 0;
  for (int mode = 0; mode < 5; ++mode) {
    cudaMemset(cnt, 0, 16);
    if (mode <= 1) k<<<nsp * nsp, 32>>>(mode, dsp, nsp, 1, cnt);
    else for (int r = 0; r < 200; ++r) k<<<1 << 16, 256>>>(mode, dsp, nsp, 1000 * mode + r, cnt);
    unsigned long long h[2]; cudaMemcpy(h, cnt, 16, cudaMemcpyDeviceToHost);
    printf("NEGZERO %s | %s: %llu of %llu outputs are -0\n", p.name, names[mode], h[0], h[1]);
    if (mode < 4) sum += h[0];
  }
  printf("NEGZERO TOTAL (accumulator not -0): %llu\n", sum);
  return 0;
}
