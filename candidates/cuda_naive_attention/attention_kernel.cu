#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

template <typename scalar_t>
__global__ void naive_attention_kernel(const scalar_t* __restrict__ q,
                                       const scalar_t* __restrict__ k,
                                       const scalar_t* __restrict__ v,
                                       scalar_t* __restrict__ output,
                                       int batch,
                                       int heads,
                                       int seq_len,
                                       int head_dim,
                                       bool causal,
                                       float scale) {
  const int row = blockIdx.x;
  const int query = row % seq_len;
  const int head = (row / seq_len) % heads;
  const int batch_idx = row / (seq_len * heads);
  const int base = ((batch_idx * heads + head) * seq_len) * head_dim;
  const int q_offset = base + query * head_dim;
  const int key_limit = causal ? query + 1 : seq_len;

  float row_max = -std::numeric_limits<float>::infinity();
  for (int key = 0; key < key_limit; ++key) {
    const int k_offset = base + key * head_dim;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_offset + dim]) * static_cast<float>(k[k_offset + dim]);
    }
    row_max = fmaxf(row_max, score * scale);
  }

  float denom = 0.0f;
  for (int key = 0; key < key_limit; ++key) {
    const int k_offset = base + key * head_dim;
    float score = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      score += static_cast<float>(q[q_offset + dim]) * static_cast<float>(k[k_offset + dim]);
    }
    denom += expf(score * scale - row_max);
  }

  const int out_offset = q_offset;
  for (int dim = 0; dim < head_dim; ++dim) {
    float acc = 0.0f;
    for (int key = 0; key < key_limit; ++key) {
      const int k_offset = base + key * head_dim;
      const int v_offset = base + key * head_dim;
      float score = 0.0f;
      for (int dot_dim = 0; dot_dim < head_dim; ++dot_dim) {
        score += static_cast<float>(q[q_offset + dot_dim]) *
                 static_cast<float>(k[k_offset + dot_dim]);
      }
      const float weight = expf(score * scale - row_max) / denom;
      acc += weight * static_cast<float>(v[v_offset + dim]);
    }
    output[out_offset + dim] = static_cast<scalar_t>(acc);
  }
}

}  // namespace

torch::Tensor attention_cuda(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             bool causal) {
  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int seq_len = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  auto output = torch::empty_like(q);
  const int rows = batch * heads * seq_len;
  if (rows == 0) {
    return output;
  }

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "naive_attention_cuda",
      [&] {
        naive_attention_kernel<scalar_t><<<rows, 1, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            heads,
            seq_len,
            head_dim,
            causal,
            scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
