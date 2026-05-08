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

constexpr int kSeqLen = 16;
constexpr int kHeadDim = 16;
constexpr int kTileElements = kSeqLen * kSeqLen;
constexpr int kThreads = 256;

__global__ void mma_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                     const __nv_bfloat16* __restrict__ k,
                                     const __nv_bfloat16* __restrict__ v,
                                     __nv_bfloat16* __restrict__ output,
                                     int batch_heads,
                                     bool causal,
                                     float scale) {
  __shared__ float scores[kTileElements];
  __shared__ __nv_bfloat16 probabilities[kTileElements];
  __shared__ float output_tile[kTileElements];

  const int bh = blockIdx.x;
  if (bh >= batch_heads) {
    return;
  }
  const int base = bh * kSeqLen * kHeadDim;

  if (threadIdx.x < warpSize) {
    wmma::fragment<wmma::matrix_a,
                   kSeqLen,
                   kSeqLen,
                   kHeadDim,
                   __nv_bfloat16,
                   wmma::row_major>
        q_frag;
    wmma::fragment<wmma::matrix_b,
                   kSeqLen,
                   kSeqLen,
                   kHeadDim,
                   __nv_bfloat16,
                   wmma::col_major>
        k_frag;
    wmma::fragment<wmma::accumulator, kSeqLen, kSeqLen, kHeadDim, float> score_frag;

    wmma::fill_fragment(score_frag, 0.0f);
    wmma::load_matrix_sync(q_frag, q + base, kHeadDim);
    wmma::load_matrix_sync(k_frag, k + base, kHeadDim);
    wmma::mma_sync(score_frag, q_frag, k_frag, score_frag);
    wmma::store_matrix_sync(scores, score_frag, kSeqLen, wmma::mem_row_major);
  }
  __syncthreads();

  for (int row = threadIdx.x; row < kSeqLen; row += blockDim.x) {
    const int key_limit = causal ? row + 1 : kSeqLen;

    float row_max = -std::numeric_limits<float>::infinity();
    for (int key = 0; key < key_limit; ++key) {
      row_max = fmaxf(row_max, scores[row * kSeqLen + key] * scale);
    }

    float denom = 0.0f;
    for (int key = 0; key < key_limit; ++key) {
      const float weight = expf(scores[row * kSeqLen + key] * scale - row_max);
      denom += weight;
    }
    for (int key = 0; key < kSeqLen; ++key) {
      const float probability =
          key < key_limit ? expf(scores[row * kSeqLen + key] * scale - row_max) / denom : 0.0f;
      probabilities[row * kSeqLen + key] = __float2bfloat16(probability);
    }
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    wmma::fragment<wmma::matrix_a,
                   kSeqLen,
                   kHeadDim,
                   kSeqLen,
                   __nv_bfloat16,
                   wmma::row_major>
        probability_frag;
    wmma::fragment<wmma::matrix_b,
                   kSeqLen,
                   kHeadDim,
                   kSeqLen,
                   __nv_bfloat16,
                   wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kSeqLen, kHeadDim, kSeqLen, float> output_frag;

    wmma::fill_fragment(output_frag, 0.0f);
    wmma::load_matrix_sync(probability_frag, probabilities, kSeqLen);
    wmma::load_matrix_sync(v_frag, v + base, kHeadDim);
    wmma::mma_sync(output_frag, probability_frag, v_frag, output_frag);
    wmma::store_matrix_sync(output_tile, output_frag, kHeadDim, wmma::mem_row_major);
  }
  __syncthreads();

  for (int linear = threadIdx.x; linear < kTileElements; linear += blockDim.x) {
    output[base + linear] = __float2bfloat16(output_tile[linear]);
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
  TORCH_CHECK(q.size(2) == kSeqLen, "seq_len must be 16");
  TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 16");

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

  mma_attention_kernel<<<batch_heads, kThreads, 0, stream>>>(
      q_ptr, k_ptr, v_ptr, output_ptr, batch_heads, causal, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
