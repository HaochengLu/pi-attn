// Current libtorch CUDA stream, so PI kernels are ordered with the rest of a forward.
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

extern "C" void* pi_current_stream(int64_t device) {
  return c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device)).stream();
}
