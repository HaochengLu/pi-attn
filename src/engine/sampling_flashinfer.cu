#include <cuda_runtime.h>
#include <flashinfer/sampling.cuh>

extern "C" const char* sglang_sampling_native(
    float* probs, int64_t* output, bool* valid, int32_t rows,
    int32_t vocab, int64_t top_k, float top_p, uint64_t seed, uint64_t offset, void* stream) {
  using namespace flashinfer::sampling;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t status;
  if (top_k < vocab) {
    status = TopKTopPSamplingFromProb<float,int64_t>(probs,nullptr,nullptr,
        output,valid,nullptr,rows,top_k,top_p,vocab,true,
        nullptr,seed,nullptr,offset,cuda_stream);
  } else if (top_p < 1.f) {
    status = TopPSamplingFromProb<float,int64_t>(probs,output,valid,nullptr,nullptr,
        rows,top_p,vocab,true,nullptr,seed,nullptr,offset,cuda_stream);
  } else {
    status = SamplingFromProb<float,int64_t>(probs,output,valid,nullptr,
        rows,vocab,true,nullptr,seed,nullptr,offset,cuda_stream);
  }
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}

// ---------------------------------------------------------------------------
// Deterministic-mode batched sampling. Fork of FlashInfer 0.6.18
// TopKTopPSamplingFromProbKernel (Apache-2.0, Copyright FlashInfer team) with one
// change: row bx draws from its own Philox stream (seed[bx], subsequence 0,
// offset[bx]) instead of (seed, subsequence = bx, offset), so a request's sample
// does not depend on its position in the batch. Per-row top-k / top-p as upstream.
namespace flashinfer {
namespace sampling {

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK_THREADS) void RowSeededTopKTopPSamplingKernel(
    DType* probs, IdType* top_k_arr, float* top_p_arr, IdType* output, bool* valid, uint32_t d,
    const uint64_t* seed_arr, const uint64_t* offset_arr) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed_arr[bx], 0, offset_arr[bx], &state);
  const uint32_t row_idx = bx;
  const uint32_t k = top_k_arr[row_idx];
  const float p = top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  float low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    temp_storage.last_valid_id = -1;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      sampled_id = temp_storage.last_valid_id;
      if (temp_storage.last_valid_id == -1) {
        if (tx == 0) {
          output[bx] = 0;
          valid[bx] = false;
        }
        return;
      }
    }
    float pivot_0 = probs[row_idx * d + sampled_id];
    float pivot_1 = ieee_mul(ieee_add(pivot_0, high), 0.5f);

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_0{0, 0};
    ValueCount<float> threadlocal_aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        threadlocal_aggregate_gt_pivot_0 += probs_gt_pivot_0[j];
        threadlocal_aggregate_gt_pivot_1 += probs_gt_pivot_1[j];
      }
    }
    aggregate_gt_pivot_0 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_0);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
    }
    __syncthreads();
    aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

    aggregate_gt_pivot_1 +=
        BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
            .Sum(threadlocal_aggregate_gt_pivot_1);
    if (tx == 0) {
      temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
    }
    __syncthreads();
    aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
    valid[bx] = true;
  }
}

}  // namespace sampling
}  // namespace flashinfer

extern "C" const char* sglang_sampling_native_rows(
    float* probs, int64_t* output, bool* valid, int32_t rows, int32_t vocab, int64_t* top_k,
    float* top_p, const uint64_t* seeds, const uint64_t* offsets, void* stream) {
  using namespace flashinfer;
  using namespace flashinfer::sampling;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (rows == 0) return nullptr;
  uint32_t d = vocab;
  const uint32_t vec_size = std::gcd(16 / sizeof(float), d);
  auto compute_capacity = GetCudaComputeCapability();
  cudaError_t status = cudaSuccess;
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = RowSeededTopKTopPSamplingKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                    true, float, int64_t>;
      status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      if (status == cudaSuccess) {
        kernel<<<rows, BLOCK_THREADS, smem_size, cuda_stream>>>(probs, top_k, top_p, output, valid, d,
                                                                 seeds, offsets);
        status = cudaGetLastError();
      }
    });
  });
  return status == cudaSuccess ? nullptr : cudaGetErrorString(status);
}
