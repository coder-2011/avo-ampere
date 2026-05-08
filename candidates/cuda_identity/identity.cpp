#include <torch/extension.h>

void identity_cuda(torch::Tensor input, torch::Tensor output);

torch::Tensor identity(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "identity input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "identity input must be contiguous");
  auto output = torch::empty_like(input);
  identity_cuda(input, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("identity", &identity, "Copy a CUDA tensor through a custom kernel");
}
