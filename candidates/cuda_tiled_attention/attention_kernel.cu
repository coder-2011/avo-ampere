#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 128;
constexpr int kTileKeys = 32;

template <typename scalar_t>
__global__ void tiled_attention_kernel(const scalar_t* __restrict__ q,
                                       const scalar_t* __restrict__ k,
                                       const scalar_t* __restrict__ v,
                                       scalar_t* __restrict__ output,
                                       int batch,
                                       int heads,
                                       int seq_len,
                                       int head_dim,
                                       bool causal,
                                       float scale) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* reduce = shared + kTileKeys;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int query = row % seq_len;
  const int head = (row / seq_len) % heads;
  const int batch_idx = row / (seq_len * heads);
  const int base = ((batch_idx * heads + head) * seq_len) * head_dim;
  const int q_offset = base + query * head_dim;
  const int key_limit = causal ? query + 1 : seq_len;

  float row_max = -std::numeric_limits<float>::infinity();
  float row_sum = 0.0f;
  float output_acc = 0.0f;

  for (int tile_start = 0; tile_start < key_limit; tile_start += kTileKeys) {
    const int tile_keys = min(kTileKeys, key_limit - tile_start);

    float local_max = -std::numeric_limits<float>::infinity();
    if (tid < tile_keys) {
      const int key = tile_start + tid;
      const int k_offset = base + key * head_dim;
      float score = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        score += static_cast<float>(q[q_offset + dim]) * static_cast<float>(k[k_offset + dim]);
      }
      score *= scale;
      scores[tid] = score;
      local_max = score;
    }
    reduce[tid] = local_max;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      }
      __syncthreads();
    }
    const float tile_max = reduce[0];

    float local_sum = 0.0f;
    if (tid < tile_keys) {
      const float shifted = expf(scores[tid] - tile_max);
      scores[tid] = shifted;
      local_sum = shifted;
    }
    reduce[tid] = local_sum;
    __syncthreads();

    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      __syncthreads();
    }
    const float tile_sum = reduce[0];
    const float new_row_max = fmaxf(row_max, tile_max);
    const float old_scale = row_sum == 0.0f ? 0.0f : expf(row_max - new_row_max);
    const float tile_scale = expf(tile_max - new_row_max);

    if (tid < head_dim) {
      float tile_acc = 0.0f;
      for (int key_inner = 0; key_inner < tile_keys; ++key_inner) {
        const int key = tile_start + key_inner;
        const int v_offset = base + key * head_dim;
        tile_acc += scores[key_inner] * static_cast<float>(v[v_offset + tid]);
      }
      output_acc = output_acc * old_scale + tile_acc * tile_scale;
    }

    row_sum = row_sum * old_scale + tile_sum * tile_scale;
    row_max = new_row_max;
    __syncthreads();
  }

  if (tid < head_dim) {
    output[q_offset + tid] = static_cast<scalar_t>(output_acc / row_sum);
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
  const int shared_bytes = (kTileKeys + kThreads) * static_cast<int>(sizeof(float));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "tiled_attention_cuda",
      [&] {
        tiled_attention_kernel<scalar_t><<<rows, kThreads, shared_bytes, stream>>>(
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
