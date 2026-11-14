// sm_90a (Hopper) kernels of the canonical GEMM program (gemm_inv.cuh), with wgmma and TMA. Compiled only
// when the build targets sm_90a (PI_SM90A; gemm_dispatch.cuh registers the ids on compute capability 9.0).
//   y[m, n] = round((((+0 + p_0) + p_1) + ...) + p_{S-1}),  p_s = one fp32 chain from +0 over the segment's
//   k16 steps in ascending order: wgmma.m64nNk16 bf16 -> fp32 with scale_d = 1 on every step.
// On Hopper one wgmma k16 step returns the bits of mma.sync.m16n8k16 for every element (measured; the
// kernels here are checked bitwise against the mma.sync ids by kernels/test_gemm.py on the device).
// From CUTLASS 3.x / 4.x: the sm90 TMA / GMMA warp-specialized collective for shared storage, the TMA
// descriptors and the producer (load). The consumer loop and the epilogue are ours:
//  - the first step accumulates into an explicit +0 (CUTLASS's mma() issues it with ScaleOut::Zero, D = A B
//    instead of D = +0 + A B);
//  - the chain restarts and the segment totals fold with __fadd_rn at the canonical segment boundaries;
//  - no split-K or stream-K: every (m tile, n tile) is computed whole by one CTA. CTAs are persistent, CTA b
//    computes tiles b, b + gridDim.x, ... of a fixed walk (which CTA computes an element is free in the
//    contract); the producer streams the next tile's K stages while the consumers run the epilogue, which
//    writes the rounded tile to shared memory and hands it to one asynchronous TMA store. The walk goes in
//    blocks `swz` n tiles wide so that one wave of CTAs shares its X rows and W columns in L2.
// One CTA = 1 producer warp group + BM / 64 consumer warp groups (64 rows x BN columns each), BM = 64, 128 or 192.
// With a cluster the walk is over cluster tiles and CTA (cx, cy) of a cluster takes tile (cx, cy) of each.
// Decode (M below one row tile): the X stage loads are issued with a TMA box of exactly M rows. A box that
// crosses the tensor's last row costs about 3 ns per out-of-bounds row on H100 (measured: a 64 x 16 tile over
// K = 6144 took 35 us at M = 1 against 16 us at M = 64), which made every tile here slower as M shrank. The
// stage rows past M then hold stale shared memory; wgmma computes on them, row by row independently of the
// real rows, and the TMA store drops them. Ids 98..102 are tiles for this regime: N tiles narrow enough that a
// 2048-column weight still gives every SM a CTA to stream through, each element's chain still the whole K.
// What bounds a launch below M ~ 1024 (H100, Qwen3-1.7B, 28 weight copies streamed from DRAM in one graph) is the
// weight bytes it keeps in flight, N x (K columns the stages of one CTA hold) x 2: down (2048 x 6144) at M = 64
// takes 18.5 us with 2.1 MB in flight (id 99), 16.5 us with 3.1 MB, 13.4 us with 4.2 MB (id 103; cuBLAS 14.8 us).
// It is not the K chain: 384 dependent wgmma k16 steps on register-resident operands take 8.0 us (20.8 ns per
// step for one warp group at n <= 64; an SM saturates at 3.6e12 multiply-adds per second). Ids 103..112 are
// tiles for this bound and for the SM count: stages 1024 columns deep for M <= 128 (103, 104), the deepest
// that fit beside the X stages for M = 192..512 (105..110), each sized so that one wave of CTAs covers the
// output (128 or 129 tiles on 132 SMs for the Qwen3-1.7B projections), and 192-row tiles (three consumer warp
// groups) that keep a 2048-token chunk plus up to 256 decode rows in one wave (111, 112: 132 tiles of 192 x 192
// at M = 2304, N = 2048, where 128 x 256 needs 144). The X stages cannot be given a shallower pipeline of their
// own: TMA loads of one CTA complete in issue order, so an X load issued behind deep W loads waits for DRAM
// (archive/ceiling_probes/h100_gemm, run D).
// Clusters (ids 113..120): two CTAs that compute the same W columns for two adjacent row tiles form a 2 x 1
// cluster and run their K stages in step; each loads half of a W stage with one TMA multicast that lands in both
// CTAs' shared memory, so a W stage enters the pair once instead of twice (X is loaded per CTA as before). The
// operands, the k16 steps and their order are those of the tile without a cluster, so the bits are too. Measured
// (archive/ceiling_probes/h100_gemm_r4): 3-18% on o and down at M = 128..512, 4-7% on qkv at M = 256 / 512,
// 1-3% on gate_up; sharing X across column tiles (1 x 2, 2 x 2) and clusters of four (4 x 1) are slower. A cluster id refuses M that fits one row tile (the M buckets of the engine end at row tile sizes, so an
// id tuned at a bucket's largest M runs every M of the bucket). Keeping X out of shared memory instead (wgmma with A from
// registers, W stages only) does not help: with no X loads at all and twice the W stages it is no faster than
// the 2 x 1 cluster, and an X stage loaded from global memory one stage ahead costs 0.7-0.8 us per K stage.
// A launch refuses (cudaErrorInvalidValue, before launching) x, w, y that are not 16-byte aligned, row
// strides that are not multiples of 8 elements (TMA), a K remainder, and a segment count its kind does not run.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/pipeline/pipeline.hpp>

namespace pi_gemm90 {

// SEG: S > 1 kernels keep the segment totals in a second register fragment (BN <= 128). BK: K columns per stage.
// CX x CY: the CTAs of one cluster compute CX m tiles x CY n tiles in step (CX * CY <= 8; 1 x 1: no cluster).
template <int BM, int BN, int ST, bool SEG, typename OutT, int BK = 64, int CX = 1, int CY = 1>
struct Sm90Gemm {
  static_assert(BM == 64 || BM == 128 || BM == 192, "one to three consumer warp groups");
  static_assert(BK % 64 == 0, "whole 64-column (128-byte swizzled) K slabs");
  static constexpr int kBK = BK;
  using TileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<kBK>>;
  static_assert(CX >= 1 && CY >= 1 && CX * CY <= 8 && BN % (8 * CX) == 0 && BM % (8 * CY) == 0, "cluster of at most 8 CTAs");
  static constexpr int kCluster = CX * CY;
  using ClusterShape = cute::Shape<cute::Int<CX>, cute::Int<CY>, cute::_1>;
  using Schedule = std::conditional_t<BM == 64, cutlass::gemm::KernelTmaWarpSpecialized,
                                      cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,     // A = X [M, K], K contiguous
      cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8,  // B = W^T [K, N] column-major = W [N, K] row-major
      float, TileShape, ClusterShape, cutlass::gemm::collective::StageCount<ST>, Schedule>::CollectiveOp;
  // BM / 64 warp groups of the collective's wgmma atom (the builder itself tiles one or two).
  using TiledMma = decltype(cute::make_tiled_mma(typename Mainloop::TiledMma::Atom{},
                                                 cute::Layout<cute::Shape<cute::Int<BM / 64>, cute::_1, cute::_1>>{}));
  using Pipeline = typename Mainloop::MainloopPipeline;
  using PipelineState = typename Mainloop::PipelineState;
  struct SharedStorage {
    typename Mainloop::SharedStorage main;
    alignas(128) OutT epi[BM * BN];  // the output tile, row-major
  };
  static constexpr int kWG = BM / 64;
  static constexpr int kThreads = 128 * (1 + kWG);

  struct Params {
    typename Mainloop::Params mainloop;
    int M, N, K, segs, swz;
    cute::TmaDescriptor tma_y;  // [M][N] of OutT, box BM x BN, no swizzle
  };

  static __device__ __forceinline__ void store2(OutT* p, float a, float b) {
    if constexpr (sizeof(OutT) == 2) *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
    else *reinterpret_cast<float2*>(p) = make_float2(a, b);
  }

  // Tile t of the walk -> (m tile, n tile): blocks of swz n tiles (the last block takes the remainder), within
  // a block m-major. A bijection of [0, tiles_m * tiles_n).
  static __device__ __forceinline__ void tile_of(int t, int tiles_m, int tiles_n, int swz, int& mt, int& nt) {
    const int per = tiles_m * swz, full = tiles_n / swz;
    if (t < full * per) {
      const int r = t % per;
      mt = r / swz;
      nt = (t / per) * swz + r % swz;
    } else {
      const int rem = tiles_n - full * swz, r = t - full * per;
      mt = r / rem;
      nt = full * swz + r % rem;
    }
  }

  static __device__ __forceinline__ void device(Params const& params, char* smem_buf) {
    using namespace cute;
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int thread_idx = threadIdx.x, lane_idx = thread_idx % 32;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const bool producer = cutlass::canonical_warp_group_idx() == 0;
    if (warp_idx == 0 && cute::elect_one_sync()) Mainloop::prefetch_tma_descriptors(params.mainloop);

    typename Pipeline::Params pp;
    if (producer && warp_idx == 0) pp.role = Pipeline::ThreadCategory::Producer;
    if (!producer) pp.role = Pipeline::ThreadCategory::Consumer;
    pp.is_leader = (thread_idx % 128) == 0;
    pp.num_consumers = kWG * 128;
    pp.transaction_bytes = params.mainloop.tma_transaction_bytes;
    Pipeline pipeline(ss.main.pipeline, pp, ClusterShape{});
    if constexpr (kCluster > 1) {  // every CTA of the cluster has its barriers before any of them is signalled
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }

    const int M = params.M, N = params.N, K = params.K;
    Mainloop mainloop;
    auto load_inputs = mainloop.load_init(make_shape(M, N, K, 1), params.mainloop);
    Tensor gA_mkl = get<0>(load_inputs);
    const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN, swz = params.swz;
    // The walk is over cluster tiles (CX x CY tiles); CTA (cx, cy) of a cluster takes tile (cx, cy) of each. A
    // tile past the last row or column tile is run on the last one's operands and not stored (`live`).
    const int ctiles_m = (tiles_m + CX - 1) / CX, ctiles_n = (tiles_n + CY - 1) / CY;
    const int tiles = ctiles_m * ctiles_n, first = blockIdx.x / CX, step = gridDim.x / CX;
    const int cx = kCluster > 1 ? (int)cute::block_id_in_cluster().x : 0;
    const int cy = kCluster > 1 ? (int)cute::block_id_in_cluster().y : 0;
    auto tile_at = [&](int t, int& mt, int& nt) {
      tile_of(t, ctiles_m, ctiles_n, swz, mt, nt);
      mt = mt * CX + cx;
      nt = nt * CY + cy;
      const bool live = mt < tiles_m && nt < tiles_n;
      mt = mt < tiles_m ? mt : tiles_m - 1;
      nt = nt < tiles_n ? nt : tiles_n - 1;
      return live;
    };
    auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
    const int k_tile_count = K / kBK;

    if (producer) {
      if (warp_idx == 0) {  // one stream of K tiles in ascending order per output tile, segments included
        PipelineState write_state = cutlass::make_producer_start_state<Pipeline>();
        for (int t = first; t < tiles; t += step) {
          int mt, nt;
          tile_at(t, mt, nt);
          auto blk_coord = make_coord(mt, nt, _, 0);
          mainloop.load(params.mainloop, pipeline, write_state, load_inputs, blk_coord, k_tile_iter, k_tile_count,
                        lane_idx, cx + cy * CX, ss.main.tensors);
          write_state.advance(k_tile_count);
        }
        mainloop.load_tail(pipeline, write_state);
      }
      return;
    }

    // ---- consumer: the canonical chain ----
    const int mma_thread = thread_idx - 128;
    TiledMma tiled_mma;  // accumulate_ == ScaleOut::One: every step is D = D + A * B
    Tensor accum = partition_fragment_C(tiled_mma, take<0, 2>(TileShape{}));
    Tensor sA = make_tensor(make_smem_ptr(ss.main.tensors.smem_A.data()), typename Mainloop::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(ss.main.tensors.smem_B.data()), typename Mainloop::SmemLayoutB{});
    Layout wg_layout = make_layout(Int<kWG>{}, Int<128>{});
    const int wg = __shfl_sync(0xFFFFFFFF, mma_thread / 128, 0);
    auto wg_mma = tiled_mma.get_slice(wg_layout(wg));
    Tensor tCsA = wg_mma.partition_A(sA);
    Tensor tCsB = wg_mma.partition_B(sB);
    Tensor tCrA = wg_mma.make_fragment_A(tCsA);
    Tensor tCrB = wg_mma.make_fragment_B(tCsB);
    static_assert(decltype(size<2>(tCrA))::value == kBK / 16, "k16 steps per stage");

    const int segs = SEG ? params.segs : 1, seg_tiles = k_tile_count / segs;
    float total[SEG ? decltype(size(accum))::value : 1];
    auto thr_mma = tiled_mma.get_thread_slice(mma_thread);
    Tensor tCcD = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));  // tile-relative (m, n)
    PipelineState read_state, release_state;
    bool stored = false;
    for (int t = first; t < tiles; t += step) {
      int mt, nt;
      const bool live = tile_at(t, mt, nt);
      if constexpr (SEG) {
        CUTE_UNROLL
        for (int i = 0; i < size(accum); ++i) total[i] = 0.f;
      }
      for (int s = 0; s < segs; ++s) {
        clear(accum);  // +0
        warpgroup_fence_operand(accum);
        bool pending = false;
        CUTLASS_PRAGMA_NO_UNROLL
        for (int kt = 0; kt < seg_tiles; ++kt) {
          auto barrier_token = pipeline.consumer_try_wait(read_state);
          pipeline.consumer_wait(read_state, barrier_token);
          const int read_stage = read_state.index();
          warpgroup_fence_operand(accum);
          warpgroup_arrive();
          CUTE_UNROLL
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)  // ascending k16 steps
            cute::gemm(tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accum);
          warpgroup_commit_batch();
          if (pending) {  // the previous stage's steps are done: free its buffer
            warpgroup_wait<1>();
            warpgroup_fence_operand(accum);
            pipeline.consumer_release(release_state);
            ++release_state;
          }
          pending = true;
          ++read_state;
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(accum);
        pipeline.consumer_release(release_state);
        ++release_state;
        if constexpr (SEG) {
          CUTE_UNROLL
          for (int i = 0; i < size(accum); ++i) total[i] = __fadd_rn(total[i], accum(i));
        }
      }

      // ---- epilogue: round to nearest even; elements i, i + 1 are one row's adjacent columns (n even) ----
      if (!live) continue;
      // The previous tile's TMA store has read epi (issued a whole mainloop ago; the wait is a formality).
      if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");  // the consumers only
      CUTE_UNROLL
      for (int i = 0; i < size(accum); i += 2) {
        auto c = tCcD(i);
        OutT* p = ss.epi + get<0>(c) * BN + get<1>(c);
        if constexpr (SEG) store2(p, total[i], total[i + 1]);
        else store2(p, accum(i), accum(i + 1));
      }
      cute::tma_store_fence();
      asm volatile("bar.sync %0, %1;" :: "r"(1), "r"(kWG * 128) : "memory");
      if (mma_thread == 0) {  // out-of-range rows / columns of an edge tile are not written
        cute::SM90_TMA_STORE_2D::copy(&params.tma_y, ss.epi, nt * BN, mt * BM);
        cute::tma_store_arrive();
      }
      stored = true;
    }
    if (mma_thread == 0 && stored) cute::tma_store_wait<0>();
  }

  template <typename SetSmem>
  static cudaError_t run(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                         int segs, int sms, SetSmem&& set_smem, cudaStream_t stream);
};

// The TMA descriptors live in the kernel parameters: they must be __grid_constant__ (a thread-local copy of
// a descriptor is not a legal TMA operand).
template <class G>
__global__ void __launch_bounds__(G::kThreads) sm90_gemm_kernel(CUTLASS_GRID_CONSTANT typename G::Params const params) {
  extern __shared__ char smem_buf[];
  G::device(params, smem_buf);
}

// set_smem(kernel, bytes): the caller's once-per-device cudaFuncSetAttribute (gemm_dispatch.cuh).
template <int BM, int BN, int ST, bool SEG, typename OutT, int BK, int CX, int CY>
template <typename SetSmem>
cudaError_t Sm90Gemm<BM, BN, ST, SEG, OutT, BK, CX, CY>::run(const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy,
                                                 int M, int N, int K, int segs, int sms, SetSmem&& set_smem,
                                                 cudaStream_t stream) {
  using Self = Sm90Gemm<BM, BN, ST, SEG, OutT, BK, CX, CY>;
  // A cluster needs CX row tiles and CY column tiles to share operands across (and the M-row X box below is
  // for one CTA's own loads).
  if (kCluster > 1 && (M <= BM * (CX - 1) || N <= BN * (CY - 1))) return cudaErrorInvalidValue;
  if (segs < 1 || K % (kBK * segs) || (!SEG && segs != 1) || (SEG && segs < 2) || N % 8 || ldy % 8 || ldx % 8 ||
      ((uintptr_t)x & 15) || ((uintptr_t)w & 15) || ((uintptr_t)y & 15) || sms < 1)
    return cudaErrorInvalidValue;
  using EA = cutlass::bfloat16_t;
  typename Mainloop::Arguments margs{(const EA*)x, cute::make_stride(int64_t(ldx), cute::Int<1>{}, int64_t(0)),
                                     (const EA*)w, cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(0))};
  Params params{Mainloop::to_underlying_arguments(cute::make_shape(M, N, K, 1), margs, nullptr), M, N, K, segs, 1, {}};
  const int smem = (int)sizeof(SharedStorage);
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  if (M < BM) {
    // X's descriptor again, as cute builds it (make_tma_copy_desc: 16-bit elements, dims (K, M, L = 1) since the
    // collective loads through the 3-d instruction, one 64-column 128-byte-swizzled slab per instruction) but
    // with a box of M rows, and the stage's byte count to match.
    cute::TmaDescriptor desc{};
    const cuuint64_t gdim[3] = {(cuuint64_t)K, (cuuint64_t)M, 1}, gstride[2] = {(cuuint64_t)ldx * 2, 0};
    const cuuint32_t box[3] = {64, (cuuint32_t)M, 1}, estride[3] = {1, 1, 1};
    if (cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, const_cast<void*>(x), gdim, gstride, box, estride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
      return cudaErrorInvalidValue;
    int driver = 0;
    if (cudaDriverGetVersion(&driver) == cudaSuccess && driver <= 13010 && (uint64_t)M * ldx * 2 < 131072)
      reinterpret_cast<uint64_t*>(&desc)[1] &= ~(1llu << 21);  // cute's workaround for small tensors on old drivers
    *const_cast<cute::TmaDescriptor*>(params.mainloop.tma_load_a.get_tma_descriptor()) = desc;
    params.mainloop.tma_transaction_bytes = (uint32_t)((M + BN) * kBK * 2);
  }
  // Persistent CTAs: as many as are resident at once (shared memory and threads per SM); more would only queue.
  const int per_sm = std::max(1, std::min((228 << 10) / smem, 2048 / kThreads));
  const int ctiles_m = (tiles_m + CX - 1) / CX, ctiles_n = (tiles_n + CY - 1) / CY;  // cluster tiles
  const int grid = std::min(ctiles_m * ctiles_n, std::max(1, sms * per_sm / kCluster));  // clusters
  // One wave = (grid / swz) x swz cluster tiles; its X rows + W columns are fewest at BM CX a = BN CY b.
  params.swz = std::max(1, std::min(ctiles_n, (int)std::lround(std::sqrt((double)grid * (BM * CX) / (BN * CY)))));
  const cuuint64_t gdim[2] = {(cuuint64_t)N, (cuuint64_t)M}, gstride[1] = {(cuuint64_t)ldy * sizeof(OutT)};
  const cuuint32_t box[2] = {BN, BM}, estride[2] = {1, 1};
  if (cuTensorMapEncodeTiled(&params.tma_y, sizeof(OutT) == 2 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                             2, y, gdim, gstride, box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return cudaErrorInvalidValue;
  set_smem(sm90_gemm_kernel<Self>, (size_t)smem);
  if constexpr (kCluster == 1) {
    sm90_gemm_kernel<Self><<<grid, kThreads, smem, stream>>>(params);
    return cudaGetLastError();
  } else {
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(CX * grid, CY, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CX;
    attribute[0].val.clusterDim.y = CY;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, sm90_gemm_kernel<Self>, params);
  }
}

// Ids 90..94: S = 1 tiles 128x256, 128x128, 64x128, 64x256, 64x64; 95..97: S > 1 tiles 128x128, 64x128, 64x64;
// 98..102: S = 1 decode tiles 64x32, 64x16 with 64-column stages and 64x64, 64x32, 64x16 with 128-column stages.
// Stage counts leave room for the epilogue tile (2 bytes per element for bf16 output, 4 for fp32).
// 103..112: S = 1, bf16 output only (an fp32 launch is refused): 64x16, 64x32 (8 stages) and 64x64 (6 stages) with
// 128-column stages; 128x64 (8 stages), 128x96 (6), 128x192 (4), 192x32 (7), 192x96 (4), 192x192 (3), 192x128 (4)
// with 64-column stages.
// 113..120: S = 1, bf16 output only, 2 x 1 clusters (W multicast to two row tiles; M above one row tile) of the tiles
// of 104, 105, 106, 107, 91, 108, 90 and 111: 64x32 (8 stages), 64x64 (6) with 128-column stages; 128x64 (8),
// 128x96 (6), 128x128 (5), 128x192 (4), 128x256 (3), 192x192 (3) with 64-column stages.
constexpr int kSm90First = 90, kSm90Last = 120;
inline bool is_sm90_kernel(int k) { return k >= kSm90First && k <= kSm90Last; }
inline int sm90_kernel_bk(int k) { return (k >= 100 && k <= 105) || k == 113 || k == 114 ? 128 : 64; }
template <typename OutT, typename SetSmem>
cudaError_t run_gemm_sm90(int k, const void* x, int64_t ldx, const void* w, OutT* y, int64_t ldy, int M, int N, int K,
                          int segs, int sms, SetSmem&& ss, cudaStream_t st) {
  constexpr bool h = sizeof(OutT) == 2;
  switch (k) {
    case 90: return Sm90Gemm<128, 256, h ? 3 : 2, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 91: return Sm90Gemm<128, 128, h ? 5 : 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 92: return Sm90Gemm<64, 128, 3, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 93: return Sm90Gemm<64, 256, h ? 4 : 3, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 94: return Sm90Gemm<64, 64, 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 95: return Sm90Gemm<128, 128, h ? 5 : 4, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 96: return Sm90Gemm<64, 128, 3, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 97: return Sm90Gemm<64, 64, 4, true, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 98: return Sm90Gemm<64, 32, 6, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 99: return Sm90Gemm<64, 16, 8, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 100: return Sm90Gemm<64, 64, 3, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 101: return Sm90Gemm<64, 32, 4, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    case 102: return Sm90Gemm<64, 16, 4, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
    default: break;
  }
  if constexpr (h) {
    switch (k) {
      case 103: return Sm90Gemm<64, 16, 8, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 104: return Sm90Gemm<64, 32, 8, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 105: return Sm90Gemm<64, 64, 6, false, OutT, 128>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 106: return Sm90Gemm<128, 64, 8, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 107: return Sm90Gemm<128, 96, 6, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 108: return Sm90Gemm<128, 192, 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 109: return Sm90Gemm<192, 32, 7, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 110: return Sm90Gemm<192, 96, 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 111: return Sm90Gemm<192, 192, 3, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 112: return Sm90Gemm<192, 128, 4, false, OutT>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 113: return Sm90Gemm<64, 32, 8, false, OutT, 128, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 114: return Sm90Gemm<64, 64, 6, false, OutT, 128, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 115: return Sm90Gemm<128, 64, 8, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 116: return Sm90Gemm<128, 96, 6, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 117: return Sm90Gemm<128, 128, 5, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 118: return Sm90Gemm<128, 192, 4, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 119: return Sm90Gemm<128, 256, 3, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      case 120: return Sm90Gemm<192, 192, 3, false, OutT, 64, 2, 1>::run(x, ldx, w, y, ldy, M, N, K, segs, sms, ss, st);
      default: break;
    }
  }
  return cudaErrorInvalidValue;
}

}  // namespace pi_gemm90
