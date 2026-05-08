#include <torch/extension.h>

torch::Tensor attention_cuda(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             bool causal);

torch::Tensor attention(torch::Tensor q,
                        torch::Tensor k,
                        torch::Tensor v,
                        bool causal) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape");
  TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have the same shape");
  TORCH_CHECK(q.dim() == 4, "q, k, and v must have shape (batch, heads, seq, dim)");
  return attention_cuda(q, k, v, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention, "Tiny tiled CUDA scaled dot-product attention");
}
