#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>

namespace {

using namespace nvcuda;

constexpr int kTile = 16;
constexpr int kMaxSeqLen = 256;
constexpr int kHeadDim = 128;
constexpr int kScoreElements = kTile * kTile;
constexpr int kOutputElements = kTile * kHeadDim;
constexpr int kThreads = 256;

__global__ void mma_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                     const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v,
                                     __nv_bfloat16* __restrict__ output,
                                     int batch_heads,
                                     int seq_len,
                                     bool causal,
                                     float scale) {
  __shared__ float scores[kScoreElements];
  __shared__ __nv_bfloat16 probabilities[kScoreElements];
  __shared__ float pv_tile[kOutputElements];
  __shared__ float output_acc[kOutputElements];
  __shared__ float row_max[kTile];
  __shared__ float row_sum[kTile];
  __shared__ float old_scale[kTile];

  const int query_tiles = seq_len / kTile;
  const int bh = blockIdx.x / query_tiles;
  if (bh >= batch_heads) {
    return;
  }
  const int query_tile = blockIdx.x % query_tiles;
  const int query_start = query_tile * kTile;
  const int base = bh * seq_len * kHeadDim;

  for (int linear = threadIdx.x; linear < kOutputElements; linear += blockDim.x) {
    output_acc[linear] = 0.0f;
  }
  for (int row = threadIdx.x; row < kTile; row += blockDim.x) {
    row_max[row] = -std::numeric_limits<float>::infinity();
    row_sum[row] = 0.0f;
    old_scale[row] = 0.0f;
  }
  __syncthreads();

  for (int key_start = 0; key_start < seq_len; key_start += kTile) {
    if (threadIdx.x < warpSize) {
      wmma::fragment<wmma::accumulator, kTile, kTile, 16, float> score_frag;
      wmma::fill_fragment(score_frag, 0.0f);

      for (int chunk = 0; chunk < 8; ++chunk) {
        wmma::fragment<wmma::matrix_a,
                       kTile,
                       kTile,
                       16,
                       __nv_bfloat16,
                       wmma::row_major>
            q_frag;
        wmma::fragment<wmma::matrix_b,
                       kTile,
                       kTile,
                       16,
                       __nv_bfloat16,
                       wmma::col_major>
            k_frag;
        const int chunk_offset = chunk * 16;
        wmma::load_matrix_sync(q_frag, q + base + query_start * kHeadDim + chunk_offset, kHeadDim);
        wmma::load_matrix_sync(k_frag, k + base + key_start * kHeadDim + chunk_offset, kHeadDim);
        wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
      }

      wmma::store_matrix_sync(scores, score_frag, kTile, wmma::mem_row_major);
    }
    __syncthreads();

    for (int row = threadIdx.x; row < kTile; row += blockDim.x) {
      const int query = query_start + row;
      int valid_keys = kTile;
      if (causal) {
        valid_keys = min(kTile, max(0, query - key_start + 1));
      }

      float tile_max = -std::numeric_limits<float>::infinity();
      for (int key = 0; key < valid_keys; ++key) {
        tile_max = fmaxf(tile_max, scores[row * kTile + key] * scale);
      }

      if (valid_keys == 0) {
        old_scale[row] = 1.0f;
        for (int key = 0; key < kTile; ++key) {
          probabilities[row * kTile + key] = __float2bfloat16(0.0f);
        }
        continue;
      }

      const float new_row_max = fmaxf(row_max[row], tile_max);
      const float rescale =
          row_sum[row] == 0.0f ? 0.0f : expf(row_max[row] - new_row_max);
      old_scale[row] = rescale;

      float tile_sum = 0.0f;
      for (int key = 0; key < kTile; ++key) {
        const float weight =
            key < valid_keys ? expf(scores[row * kTile + key] * scale - new_row_max) : 0.0f;
        probabilities[row * kTile + key] = __float2bfloat16(weight);
        tile_sum += weight;
      }
      row_sum[row] = row_sum[row] * rescale + tile_sum;
      row_max[row] = new_row_max;
    }
    __syncthreads();

    for (int linear = threadIdx.x; linear < kOutputElements; linear += blockDim.x) {
      const int row = linear / kHeadDim;
      output_acc[linear] *= old_scale[row];
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
      for (int chunk = 0; chunk < 8; ++chunk) {
        wmma::fragment<wmma::matrix_a,
                       kTile,
                       16,
                       kTile,
                       __nv_bfloat16,
                       wmma::row_major>
            probability_frag;
        wmma::fragment<wmma::matrix_b,
                       kTile,
                       16,
                       kTile,
                       __nv_bfloat16,
                       wmma::row_major>
            v_frag;
        wmma::fragment<wmma::accumulator, kTile, 16, kTile, float> output_frag;

        wmma::fill_fragment(output_frag, 0.0f);
        wmma::load_matrix_sync(probability_frag, probabilities, kTile);
        const int chunk_offset = chunk * 16;
        wmma::load_matrix_sync(v_frag, v + base + key_start * kHeadDim + chunk_offset, kHeadDim);
        wmma::mma_sync(output_frag, probability_frag, v_frag, output_frag);
        wmma::store_matrix_sync(&pv_tile[chunk_offset], output_frag, kHeadDim, wmma::mem_row_major);
      }
    }
    __syncthreads();

    for (int linear = threadIdx.x; linear < kOutputElements; linear += blockDim.x) {
      output_acc[linear] += pv_tile[linear];
    }
    __syncthreads();
  }

  for (int linear = threadIdx.x; linear < kOutputElements; linear += blockDim.x) {
    const int row = linear / kHeadDim;
    output[base + (query_start * kHeadDim) + linear] =
        __float2bfloat16(output_acc[linear] / row_sum[row]);
  }
}

}  // namespace

torch::Tensor attention_cuda(torch::Tensor q,
                             torch::Tensor k,
                             torch::Tensor v,
                             bool causal) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::BFloat16, "k must be bf16");
  TORCH_CHECK(v.scalar_type() == at::ScalarType::BFloat16, "v must be bf16");
  const int seq_len = static_cast<int>(q.size(2));
  TORCH_CHECK(
      seq_len == kTile || seq_len == 2 * kTile || seq_len == 4 * kTile ||
          seq_len == 8 * kTile || seq_len == kMaxSeqLen,
      "seq_len must be 16, 32, 64, 128, or 256");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");

  auto output = torch::empty_like(q);
  const int batch_heads = static_cast<int>(q.size(0) * q.size(1));
  if (batch_heads == 0) {
    return output;
  }

  const auto* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const auto* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int query_tiles = seq_len / kTile;

  mma_attention_kernel<<<batch_heads * query_tiles, kThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, output_ptr, batch_heads, seq_len, causal, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
