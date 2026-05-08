#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;
constexpr int kThreads = kRowsPerBlock * kWarpSize;
constexpr int kTileKeys = kWarpSize;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxLaneDims = kMaxHeadDim / kWarpSize;

template <typename scalar_t>
struct alignas(sizeof(scalar_t) * 4) ScalarPack4 {
  scalar_t values[4];
};

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ __forceinline__ float dot_product(const scalar_t* __restrict__ q_row,
                                             const scalar_t* __restrict__ k_row,
                                             int head_dim) {
  float score = 0.0f;
  if ((head_dim & 3) == 0) {
    const auto* q_pack = reinterpret_cast<const ScalarPack4<scalar_t>*>(q_row);
    const auto* k_pack = reinterpret_cast<const ScalarPack4<scalar_t>*>(k_row);
    for (int dim = 0; dim < head_dim / 4; ++dim) {
      const ScalarPack4<scalar_t> qv = q_pack[dim];
      const ScalarPack4<scalar_t> kv = k_pack[dim];
#pragma unroll
      for (int inner = 0; inner < 4; ++inner) {
        score += static_cast<float>(qv.values[inner]) * static_cast<float>(kv.values[inner]);
      }
    }
    return score;
  }

  for (int dim = 0; dim < head_dim; ++dim) {
    score += static_cast<float>(q_row[dim]) * static_cast<float>(k_row[dim]);
  }
  return score;
}

template <typename scalar_t>
__global__ void warp_rows_attention_kernel(const scalar_t* __restrict__ q,
                                           const scalar_t* __restrict__ k,
                                           const scalar_t* __restrict__ v,
                                           scalar_t* __restrict__ output,
                                           int batch,
                                           int heads,
                                           int seq_len,
                                           int head_dim,
                                           int rows,
                                           bool causal,
                                           float scale) {
  __shared__ float score_tiles[kRowsPerBlock][kTileKeys];

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int row = blockIdx.x * kRowsPerBlock + warp_id;
  if (row >= rows) {
    return;
  }

  const int query = row % seq_len;
  const int head = (row / seq_len) % heads;
  const int batch_idx = row / (seq_len * heads);
  const int base = ((batch_idx * heads + head) * seq_len) * head_dim;
  const int q_offset = base + query * head_dim;
  const int key_limit = causal ? query + 1 : seq_len;
  float* scores = score_tiles[warp_id];

  float row_max = -std::numeric_limits<float>::infinity();
  float row_sum = 0.0f;
  float output_acc[kMaxLaneDims];
#pragma unroll
  for (int slot = 0; slot < kMaxLaneDims; ++slot) {
    output_acc[slot] = 0.0f;
  }

  for (int tile_start = 0; tile_start < key_limit; tile_start += kTileKeys) {
    const int tile_keys = min(kTileKeys, key_limit - tile_start);
    float score = -std::numeric_limits<float>::infinity();
    if (lane < tile_keys) {
      const int key = tile_start + lane;
      const int k_offset = base + key * head_dim;
      score = dot_product(q + q_offset, k + k_offset, head_dim) * scale;
    }

    const float tile_max = warp_reduce_max(score);
    const float shifted = lane < tile_keys ? expf(score - tile_max) : 0.0f;
    scores[lane] = shifted;
    const float tile_sum = warp_reduce_sum(shifted);
    __syncwarp();

    const float new_row_max = fmaxf(row_max, tile_max);
    const float old_scale = row_sum == 0.0f ? 0.0f : expf(row_max - new_row_max);
    const float tile_scale = expf(tile_max - new_row_max);

#pragma unroll
    for (int slot = 0; slot < kMaxLaneDims; ++slot) {
      const int dim = lane + slot * kWarpSize;
      if (dim < head_dim) {
        float tile_acc = 0.0f;
        for (int key_inner = 0; key_inner < tile_keys; ++key_inner) {
          const int key = tile_start + key_inner;
          const int v_offset = base + key * head_dim;
          tile_acc += scores[key_inner] * static_cast<float>(v[v_offset + dim]);
        }
        output_acc[slot] = output_acc[slot] * old_scale + tile_acc * tile_scale;
      }
    }

    row_sum = row_sum * old_scale + tile_sum * tile_scale;
    row_max = new_row_max;
    __syncwarp();
  }

#pragma unroll
  for (int slot = 0; slot < kMaxLaneDims; ++slot) {
    const int dim = lane + slot * kWarpSize;
    if (dim < head_dim) {
      output[q_offset + dim] = static_cast<scalar_t>(output_acc[slot] / row_sum);
    }
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

  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "warp_rows_attention_cuda",
      [&] {
        warp_rows_attention_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            heads,
            seq_len,
            head_dim,
            rows,
            causal,
            scale);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
