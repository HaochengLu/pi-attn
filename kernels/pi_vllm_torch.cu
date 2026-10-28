// PyTorch binding of the engine's GEMM entry points (pi_launch.cu, compiled in whole) for
// vLLM: linear() is the engine's pi_linear with the engine's tuner in front of it (pi.rs
// tune(), ported here), so vLLM runs exactly the tile choice and kernels the engine runs.
// Every tile gives identical bits; tuning only changes speed.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <set>

#include "pi_launch.cu"

namespace {

std::mutex attempted_mutex;
std::set<std::tuple<int, int, int>> attempted;  // (N, K, bucket) tuned or being tuned
int64_t heuristic_calls = 0;                      // calls that ran an untuned bucket's fixed tile
std::set<std::pair<int, int>> seen;               // (N, K) of every weight linear() ran

void check(const char* error) { TORCH_CHECK(error == nullptr, error); }

// pi.rs tune(): the first time (N, K, bucket) is seen outside graph capture, time every
// candidate on scratch input (and, for buckets <= 512, over weight copies in a 2x-L2 flush
// buffer) and remember the fastest. Inside capture the bucket keeps the fixed heuristic.
void tune(const at::Tensor& w, int N, int K, int M, cudaStream_t st) {
  const int bucket = pi_linear_bucket(M);
  if (pi_linear_tuned_kernel(N, K, bucket) != 0) return;
  if (pi_stream_capturing(st)) {
    ++heuristic_calls;
    return;
  }
  {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    if (!attempted.insert({N, K, bucket}).second) return;
  }
  auto x = at::empty({bucket, K}, w.options());
  auto y = at::empty({bucket, N}, w.options());
  const int64_t flush_bytes = bucket <= 512 ? 2 * pi_l2_bytes() : 0;
  at::Tensor flush;
  if (flush_bytes) flush = at::empty({flush_bytes}, w.options().dtype(at::kByte));
  check(pi_linear_tune(w.data_ptr(), N, K, bucket, x.data_ptr(), y.data_ptr(),
                       flush_bytes ? flush.data_ptr() : nullptr, flush_bytes, st));
}

}  // namespace

// y [M, N] bf16 = x [M, K] w [N, K]^T (x rows contiguous, any row stride; w contiguous).
at::Tensor linear(at::Tensor x, at::Tensor w) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1) && w.is_contiguous() && x.stride(1) == 1);
  const c10::cuda::CUDAGuard guard(x.device());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  auto y = at::empty({M, N}, x.options());
  if (M == 0) return y;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    seen.insert({N, K});
  }
  tune(w, N, K, M, st);
  check(pi_linear(x.data_ptr(), x.stride(0), w.data_ptr(), y.data_ptr(), M, N, K, 0, st));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear);
  m.def("bucket", &pi_linear_bucket, "M bucket the tuner keys on");
  m.def("tuned_kernel", &pi_linear_tuned_kernel, "kernel id tuned for (N, K, bucket), 0 = not tuned");
  m.def("shapes", [] {
    std::lock_guard<std::mutex> lock(attempted_mutex);
    return std::vector<std::pair<int, int>>(seen.begin(), seen.end());
  }, "(N, K) of every weight linear() ran");
  m.def("heuristic_calls", [] { return heuristic_calls; }, "calls inside graph capture on an untuned bucket");
}
