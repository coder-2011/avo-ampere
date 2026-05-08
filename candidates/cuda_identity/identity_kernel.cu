#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void identity_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                int64_t numel) {
  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numel) {
    output[index] = input[index];
  }
}

}  // namespace

void identity_cuda(torch::Tensor input, torch::Tensor output) {
  const int64_t numel = input.numel();
  if (numel == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((numel + threads - 1) / threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "identity_cuda",
      [&] {
        identity_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
