// The serving engine's own row-wise layer kernels (llm_infer_rust/src/layers/fused_ops.cu, through
// its ATen bridge) and the PI token log-probability (pi_logprob.cuh), as torch ops, so a PyTorch
// forward (engine_forward.py) runs exactly the engine's kernels.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "pi_logprob.cuh"

extern "C" {
const char* sglang_layers_error();
at::Tensor* sglang_layers_rms_norm(const at::Tensor* x, const at::Tensor* weight, double eps);
bool sglang_layers_add_rms_norm_inplace(const at::Tensor* x, const at::Tensor* residual, const at::Tensor* weight,
                                        double eps);
bool sglang_layers_qk_norm_inplace(const at::Tensor* q, const at::Tensor* k, const at::Tensor* qw,
                                   const at::Tensor* kw, double eps);
bool sglang_layers_rope_inplace(const at::Tensor* q, const at::Tensor* k, const at::Tensor* positions,
                                const at::Tensor* cos, const at::Tensor* sin);
at::Tensor* sglang_layers_silu_and_mul(const at::Tensor* gate_up);
}

static at::Tensor take(at::Tensor* t) {
  TORCH_CHECK(t, sglang_layers_error());
  at::Tensor out = *t;
  delete t;
  return out;
}

at::Tensor rms_norm(at::Tensor x, at::Tensor weight, double eps) { return take(sglang_layers_rms_norm(&x, &weight, eps)); }

// x <- rms_norm(x + residual) * weight, residual <- x + residual (rounded), in place.
void add_rms_norm_(at::Tensor x, at::Tensor residual, at::Tensor weight, double eps) {
  TORCH_CHECK(sglang_layers_add_rms_norm_inplace(&x, &residual, &weight, eps), sglang_layers_error());
}

void qk_norm_(at::Tensor q, at::Tensor k, at::Tensor qw, at::Tensor kw, double eps) {
  TORCH_CHECK(sglang_layers_qk_norm_inplace(&q, &k, &qw, &kw, eps), sglang_layers_error());
}

void rope_(at::Tensor q, at::Tensor k, at::Tensor positions, at::Tensor cos, at::Tensor sin) {
  TORCH_CHECK(sglang_layers_rope_inplace(&q, &k, &positions, &cos, &sin), sglang_layers_error());
}

at::Tensor silu_and_mul(at::Tensor gate_up) { return take(sglang_layers_silu_and_mul(&gate_up)); }

// log softmax(logits[r])[tokens[r]] in f32 for logits [rows, vocab] (bf16 or f32), tokens [rows] int64.
at::Tensor token_logprob(at::Tensor logits, at::Tensor tokens) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.stride(1) == 1, "logits: 2-D CUDA rows");
  TORCH_CHECK(logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kFloat, "logits: bf16 or f32");
  TORCH_CHECK(tokens.scalar_type() == at::kLong && tokens.numel() == logits.size(0), "tokens: one int64 per row");
  tokens = tokens.contiguous();
  auto out = torch::empty({logits.size(0)}, logits.options().dtype(at::kFloat));
  const cudaError_t err = pi_logprob::token_logprob(
      logits.data_ptr(), logits.scalar_type() == at::kFloat, logits.stride(0), (int)logits.size(0),
      (int)logits.size(1), tokens.data_ptr<int64_t>(), out.data_ptr<float>(), at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm);
  m.def("add_rms_norm_", &add_rms_norm_);
  m.def("qk_norm_", &qk_norm_);
  m.def("rope_", &rope_);
  m.def("silu_and_mul", &silu_and_mul);
  m.def("token_logprob", &token_logprob);
}
