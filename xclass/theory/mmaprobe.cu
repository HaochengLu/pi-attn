// mmaprobe.cu -- run every generated case (gen.h) through mma.sync m16n8k16 bf16->fp32 (FULL), through two
// zero-padded k8 halves (SPLIT8) and, with -DWGMMA, through wgmma m64n8k16; dump raw outputs to out.bin.
// Each tile carries 8 independent cases on a diagonal: case j uses A row rowbase+j and B column j.
//   nvcc -O2 -std=c++17 -arch=sm_80 mmaprobe.cu -o mmaprobe          (class A)
//   nvcc -O2 -std=c++17 -arch=sm_90a -DWGMMA mmaprobe.cu -o mmaprobe (Hopper)
// usage: ./mmaprobe seed n1 n2 ... n8
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "gen.h"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

__device__ __forceinline__ uint32_t pk(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }
__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1, uint32_t a2,
                                      uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint16_t XA(const Case* cs, int rb, int r, int k) { const int j = r - rb; return (j >= 0 && j < 8) ? cs[j].a[k] : (uint16_t)0; }
__device__ __forceinline__ float CC(const Case* cs, int rb, int r, int n) { const int j = r - rb; return (j >= 0 && j < 8 && j == n) ? __uint_as_float(cs[j].c) : 0.f; }

// mode 0 FULL, 1 SPLIT8 (lo then hi), 2 SPLIT8 reversed (hi then lo), 3 SWAP halves (k -> k^8) FULL
__global__ void k_mma(const Case* all, float* out, int mode) {
  const int t = blockIdx.x, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2, rb = (t & 1) * 8;
  const Case* cs = all + (size_t)t * 8;
  const int r0 = g, r1 = g + 8;
  float d0 = CC(cs, rb, r0, k0), d1 = CC(cs, rb, r0, k0 + 1), d2 = CC(cs, rb, r1, k0), d3 = CC(cs, rb, r1, k0 + 1);
  const int x = (mode == 3) ? 8 : 0;
  const uint32_t a0 = pk(XA(cs, rb, r0, (k0) ^ x), XA(cs, rb, r0, (k0 + 1) ^ x)), a1 = pk(XA(cs, rb, r1, (k0) ^ x), XA(cs, rb, r1, (k0 + 1) ^ x));
  const uint32_t a2 = pk(XA(cs, rb, r0, (k0 + 8) ^ x), XA(cs, rb, r0, (k0 + 9) ^ x)), a3 = pk(XA(cs, rb, r1, (k0 + 8) ^ x), XA(cs, rb, r1, (k0 + 9) ^ x));
  const uint32_t b0 = pk(cs[g].b[(k0) ^ x], cs[g].b[(k0 + 1) ^ x]), b1 = pk(cs[g].b[(k0 + 8) ^ x], cs[g].b[(k0 + 9) ^ x]);
  if (mode == 0 || mode == 3) mma16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
  else if (mode == 1) { mma16(d0, d1, d2, d3, a0, a1, 0u, 0u, b0, 0u); mma16(d0, d1, d2, d3, 0u, 0u, a2, a3, 0u, b1); }
  else { mma16(d0, d1, d2, d3, 0u, 0u, a2, a3, 0u, b1); mma16(d0, d1, d2, d3, a0, a1, 0u, 0u, b0, 0u); }
  const int j = g;  // this thread owns case j = g iff column j is one of its two columns
  if ((j & ~1) == k0) out[(size_t)t * 8 + j] = rb ? ((j & 1) ? d3 : d2) : ((j & 1) ? d1 : d0);
}

#ifdef WGMMA
__device__ __forceinline__ int smem_off(int r, int k, int lbo, int sbo) { return (r >> 3) * sbo + (k >> 3) * lbo + (r & 7) * 16 + (k & 7) * 2; }
__device__ __forceinline__ uint64_t make_desc(const void* p, int lbo, int sbo, int swap) {
  const uint64_t addr = (uint64_t)__cvta_generic_to_shared(p);
  const uint64_t l = (uint64_t)((swap ? sbo : lbo) >> 4) & 0x3FFF, s = (uint64_t)((swap ? lbo : sbo) >> 4) & 0x3FFF;
  return ((addr >> 4) & 0x3FFF) | (l << 16) | (s << 32);
}
__global__ void __launch_bounds__(128) k_wgmma(const Case* all, float* out, int lbo, int sbo, int swap) {
  __shared__ __align__(128) unsigned char sA[2048];
  __shared__ __align__(128) unsigned char sB[2048];
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, k0 = (lane & 3) * 2, rb = (t & 7) * 8;
  const Case* cs = all + (size_t)t * 8;
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  float d0 = CC(cs, rb, r0, k0), d1 = CC(cs, rb, r0, k0 + 1), d2 = CC(cs, rb, r1, k0), d3 = CC(cs, rb, r1, k0 + 1);
  const uint64_t da = make_desc(sA, lbo, sbo, swap), db = make_desc(sB, lbo, sbo, swap);
  for (int i = threadIdx.x; i < 1024; i += 128) *(uint16_t*)(sA + smem_off(i >> 4, i & 15, lbo, sbo)) = XA(cs, rb, i >> 4, i & 15);
  for (int i = threadIdx.x; i < 128; i += 128) *(uint16_t*)(sB + smem_off(i >> 4, i & 15, lbo, sbo)) = cs[i >> 4].b[i & 15];
  __syncthreads();
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db), "r"(1));
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  asm volatile("" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) :: "memory");
  const int j = r0 - rb, j2 = r1 - rb;
  if (j >= 0 && j < 8 && (j & ~1) == k0) out[(size_t)t * 8 + j] = (j & 1) ? d1 : d0;
  if (j2 >= 0 && j2 < 8 && (j2 & ~1) == k0) out[(size_t)t * 8 + j2] = (j2 & 1) ? d3 : d2;
}
#endif

static uint64_t fnv(const std::vector<uint32_t>& v) {
  uint64_t h = 1469598103934665603ull; const unsigned char* p = (const unsigned char*)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
int main(int argc, char** argv) {
  int counts[16] = {0}; uint64_t seed = strtoull(argv[1], 0, 0);
  for (int i = 2; i < argc && i < 17; ++i) counts[i - 1] = atoi(argv[i]);
  std::vector<Case> cs; std::vector<uint8_t> set; gen::make(cs, set, seed, counts);
  const size_t n = cs.size(); while (cs.size() % 64) { Case z; memset(&z, 0, sizeof z); cs.push_back(z); }
  const size_t np = cs.size(); const int tiles = (int)(np / 8);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU %s sm_%d%d cases %zu\n", p.name, p.major, p.minor, n);
  Case* dC; float* dO; CK(cudaMalloc(&dC, np * sizeof(Case))); CK(cudaMalloc(&dO, np * 4));
  CK(cudaMemcpy(dC, cs.data(), np * sizeof(Case), cudaMemcpyHostToDevice));
  FILE* f = fopen("out.bin", "wb"); uint64_t hdr[2] = {n, 0};
  std::vector<std::vector<uint32_t>> outs;
  for (int mode = 0; mode < 4; ++mode) {
    std::vector<uint32_t> o(np); CK(cudaMemset(dO, 0xFF, np * 4));
    k_mma<<<tiles, 32>>>(dC, dO, mode); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(o.data(), dO, np * 4, cudaMemcpyDeviceToHost)); o.resize(n);
    printf("mode %d hash %016llx\n", mode, (unsigned long long)fnv(o)); outs.push_back(o);
  }
#ifdef WGMMA
  {
    const int cand[2][3] = {{128, 256, 0}, {1024, 128, 0}};
    size_t best = (size_t)-1; std::vector<uint32_t> bo;
    for (int v = 0; v < 2; ++v) {
      std::vector<uint32_t> o(np); CK(cudaMemset(dO, 0xFF, np * 4));
      k_wgmma<<<tiles, 128>>>(dC, dO, cand[v][0], cand[v][1], cand[v][2]); CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(o.data(), dO, np * 4, cudaMemcpyDeviceToHost)); o.resize(n);
      size_t mm = 0; for (size_t i = 0; i < n; ++i) mm += o[i] != outs[0][i];
      printf("wgmma layout lbo=%d sbo=%d swap=%d: mismatches vs FULL %zu hash %016llx\n", cand[v][0], cand[v][1], cand[v][2], mm, (unsigned long long)fnv(o));
      if (mm < best) { best = mm; bo = o; }
    }
    outs.push_back(bo);
  }
#endif
  // dump only distinct arrays; print which mode maps to which dumped array
  std::vector<int> keep;
  for (size_t i = 0; i < outs.size(); ++i) {
    int dup = -1;
    for (size_t q = 0; q < keep.size(); ++q) if (outs[keep[q]] == outs[i]) dup = (int)q;
    if (dup < 0) { keep.push_back((int)i); dup = (int)keep.size() - 1; }
    printf("DUMPMAP mode %zu -> array %d\n", i, dup);
  }
  hdr[1] = keep.size(); fwrite(hdr, 8, 2, f);
  for (int q : keep) fwrite(outs[q].data(), 4, n, f);
  fclose(f);
  return 0;
}
