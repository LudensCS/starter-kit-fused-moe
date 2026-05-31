#include "pure_moe.h"

#include <mma.h>

#include <cmath>

using namespace nvcuda;

namespace {

constexpr int kNumExperts = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kTopK = 8;
constexpr int kTopKGroup = 4;
constexpr int kHiddenSize = 7168;
constexpr int kIntermediateSize = 2048;
constexpr int kGemm1GridX = 32;
constexpr int kGemm2GridSize = 3584;
constexpr int kSmemPadK = 72;

__device__ __forceinline__ int min_int(int a, int b) {
  return a < b ? a : b;
}

__device__ __forceinline__ float sigmoid_pure(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_pure(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_max_pure(float val, int& src_lane) {
  int lane = threadIdx.x % 32;
  src_lane = lane;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float other = __shfl_down_sync(0xffffffff, val, offset);
    int other_lane = __shfl_down_sync(0xffffffff, src_lane, offset);
    if (other > val) {
      val = other;
      src_lane = other_lane;
    }
  }
  return val;
}

__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
  atomicAdd(address, val);
}

__global__ void routing_kernel_pure(
    const float* __restrict__ logits,
    const __nv_bfloat16* __restrict__ bias,
    float routed_scaling_factor,
    int seq_len,
    int local_expert_offset,
    int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int* __restrict__ expert_counts,
    __nv_bfloat16* __restrict__ output) {
  int token_idx = blockIdx.x;
  if (token_idx >= seq_len) {
    return;
  }
  int tid = threadIdx.x;

  int warp_id = tid / 32;
  int lane = tid % 32;

  float logit = logits[static_cast<int64_t>(token_idx) * kNumExperts + tid];
  float bias_val = bias != nullptr ? __bfloat162float(bias[tid]) : 0.0f;
  float sig = sigmoid_pure(logit);
  float score_wb = sig + bias_val;

  float max1 = score_wb;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    max1 = fmaxf(max1, __shfl_down_sync(0xffffffff, max1, offset));
  }
  max1 = __shfl_sync(0xffffffff, max1, 0);

  unsigned int mask = __ballot_sync(0xffffffff, score_wb == max1);
  int first_winner = __ffs(mask) - 1;
  float val2 = lane == first_winner ? -INFINITY : score_wb;

  float max2 = val2;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    max2 = fmaxf(max2, __shfl_down_sync(0xffffffff, max2, offset));
  }
  max2 = __shfl_sync(0xffffffff, max2, 0);

  __shared__ float group_scores[8];
  if (lane == 0) {
    group_scores[warp_id] = max1 + max2;
  }
  __syncthreads();

  __shared__ unsigned int group_mask;
  if (tid == 0) {
    float scores[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      scores[i] = group_scores[i];
    }
    unsigned int gmask = 0;
    for (int k = 0; k < kTopKGroup; ++k) {
      float best = -INFINITY;
      int best_idx = -1;
      for (int i = 0; i < 8; ++i) {
        if (scores[i] > best) {
          best = scores[i];
          best_idx = i;
        }
      }
      if (best_idx >= 0) {
        gmask |= (1u << best_idx);
        scores[best_idx] = -INFINITY;
      }
    }
    group_mask = gmask;
  }
  __syncthreads();

  if (((group_mask >> warp_id) & 1u) == 0) {
    score_wb = -INFINITY;
  }

  __shared__ int out_ids[kTopK];
  __shared__ float out_sigs[kTopK];

  for (int k = 0; k < kTopK; ++k) {
    int src_lane;
    float warp_max = warp_reduce_max_pure(score_wb, src_lane);

    __shared__ float warp_maxes[8];
    __shared__ int warp_src_lanes[8];
    if (lane == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_src_lanes[warp_id] = src_lane;
    }
    __syncthreads();

    if (tid == 0) {
      float global_max = -INFINITY;
      int global_warp = -1;
      for (int w = 0; w < 8; ++w) {
        if (warp_maxes[w] > global_max) {
          global_max = warp_maxes[w];
          global_warp = w;
        }
      }
      warp_src_lanes[0] = global_warp * 32 + warp_src_lanes[global_warp];
    }
    __syncthreads();

    int winner_tid = warp_src_lanes[0];
    if (tid == winner_tid) {
      out_ids[k] = tid;
      out_sigs[k] = sig;
      score_wb = -INFINITY;
    }
    __syncthreads();
  }

  if (tid < kTopK) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
      sum += out_sigs[i];
    }
    float weight = sum > 1.0e-20f ? out_sigs[tid] / sum : 1.0f;
    int expert = out_ids[tid];
    topk_ids[static_cast<int64_t>(token_idx) * kTopK + tid] = expert;
    topk_weights[static_cast<int64_t>(token_idx) * kTopK + tid] =
        weight * routed_scaling_factor;

    int local = expert - local_expert_offset;
    if (local >= 0 && local < kNumLocalExperts) {
      atomicAdd(&expert_counts[local], 1);
    }
  }
}

__global__ void sort_scatter_kernel_pure(
    const int* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int seq_len,
    int local_expert_offset,
    int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int* __restrict__ sorted_token_ids,
    float* __restrict__ sorted_weights) {
  __shared__ int smem_offsets[kNumLocalExperts];
  int tid = threadIdx.x;

  if (tid < kNumLocalExperts) {
    int count = expert_counts[tid];
    int sum = count;
#pragma unroll
    for (int off = 1; off < 32; off *= 2) {
      int other = __shfl_up_sync(0xffffffff, sum, off);
      if (tid >= off) {
        sum += other;
      }
    }
    int offset = sum - count;
    expert_offsets[tid] = offset;
    smem_offsets[tid] = offset;
    expert_counts[tid] = 0;
  }
  __syncthreads();

  int total_elements = seq_len * kTopK;
  for (int i = tid; i < total_elements; i += blockDim.x) {
    int expert = topk_ids[i];
    int local = expert - local_expert_offset;
    if (local >= 0 && local < kNumLocalExperts) {
      int pos = atomicAdd(&expert_counts[local], 1);
      int dest = smem_offsets[local] + pos;
      sorted_token_ids[dest] = i / kTopK;
      sorted_weights[dest] = topk_weights[i];
    }
  }
}

__global__ __launch_bounds__(128) void gemm1_kernel_pure(
    const __nv_fp8_e4m3* __restrict__ hidden_states,
    const float* __restrict__ hidden_states_scale,
    const __nv_fp8_e4m3* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    __nv_bfloat16* __restrict__ intermediate_buffer,
    int seq_len) {
  extern __shared__ char smem_buf[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
  __nv_bfloat16* smem_w_g = smem_a + 2 * 32 * kSmemPadK;
  __nv_bfloat16* smem_w_u = smem_w_g + 2 * 64 * kSmemPadK;

  int n_blk = blockIdx.x;
  int expert_idx = blockIdx.y;
  int num_tokens = expert_counts[expert_idx];
  if (num_tokens == 0) {
    return;
  }

  int gate_col_start = n_blk * 64;
  int up_col_start = gate_col_start + kIntermediateSize;
  int scale_n_g = gate_col_start / 128;
  int scale_n_u = up_col_start / 128;

  int tid = threadIdx.x;
  int warp_row = ((tid / 32) % 2) * 16;
  int warp_col = ((tid / 32) / 2) * 32;

  int m_loop_begin = 0;
  int m_loop_end = num_tokens;
  if (gridDim.z > 1) {
    m_loop_begin = static_cast<int>(blockIdx.z) * 32;
    if (m_loop_begin >= num_tokens) {
      return;
    }
    m_loop_end = min_int(m_loop_begin + 32, num_tokens);
  }

  for (int m_start = m_loop_begin; m_start < m_loop_end; m_start += 32) {
    int m_current = min_int(32, num_tokens - m_start);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_g[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_u[2];
    wmma::fill_fragment(acc_g[0], 0.0f);
    wmma::fill_fragment(acc_g[1], 0.0f);
    wmma::fill_fragment(acc_u[0], 0.0f);
    wmma::fill_fragment(acc_u[1], 0.0f);

    auto load_tile = [&](int buf_idx, int k_idx) {
      int k_base = k_idx * 64;
      int scale_idx = k_idx / 2;

      int row = tid / 4;
      int col = (tid % 4) * 16;
      if (row < 32) {
        float scale = 0.0f;
        int4 packed = make_int4(0, 0, 0, 0);
        if (row < m_current) {
          int token = sorted_token_ids[expert_offsets[expert_idx] + m_start + row];
          scale = hidden_states_scale[static_cast<int64_t>(scale_idx) * seq_len + token];
          packed = *reinterpret_cast<const int4*>(
              &hidden_states[static_cast<int64_t>(token) * kHiddenSize + k_base + col]);
        }
        __nv_fp8_e4m3 vals[16];
        *reinterpret_cast<int4*>(vals) = packed;
        __nv_bfloat16* dst =
            &smem_a[buf_idx * 32 * kSmemPadK + row * kSmemPadK + col];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst[i] = __float2bfloat16(static_cast<float>(vals[i]) * scale);
        }
      }

      int t_row = tid / 4;
      int t_col = (tid % 4) * 16;
      float scale_g = gemm1_weights_scale
          [(static_cast<int64_t>(expert_idx) * 32 + scale_n_g) * 56 + scale_idx];
      float scale_u = gemm1_weights_scale
          [(static_cast<int64_t>(expert_idx) * 32 + scale_n_u) * 56 + scale_idx];

#pragma unroll
      for (int r_off = 0; r_off < 64; r_off += 32) {
        int r = t_row + r_off;
        int c = t_col;
        int64_t base = static_cast<int64_t>(expert_idx) * 2 * kIntermediateSize * kHiddenSize;
        int64_t off_g = (static_cast<int64_t>(gate_col_start + r) * kHiddenSize) + k_base + c;
        int64_t off_u = (static_cast<int64_t>(up_col_start + r) * kHiddenSize) + k_base + c;
        int4 packed_g = *reinterpret_cast<const int4*>(&gemm1_weights[base + off_g]);
        int4 packed_u = *reinterpret_cast<const int4*>(&gemm1_weights[base + off_u]);

        __nv_fp8_e4m3 vals_g[16];
        __nv_fp8_e4m3 vals_u[16];
        *reinterpret_cast<int4*>(vals_g) = packed_g;
        *reinterpret_cast<int4*>(vals_u) = packed_u;

        __nv_bfloat16* dst_g =
            &smem_w_g[buf_idx * 64 * kSmemPadK + r * kSmemPadK + c];
        __nv_bfloat16* dst_u =
            &smem_w_u[buf_idx * 64 * kSmemPadK + r * kSmemPadK + c];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          dst_g[i] = __float2bfloat16(static_cast<float>(vals_g[i]) * scale_g);
          dst_u[i] = __float2bfloat16(static_cast<float>(vals_u[i]) * scale_u);
        }
      }
    };

    load_tile(0, 0);
    __syncthreads();

#pragma unroll 1
    for (int k_idx = 0; k_idx < 112; ++k_idx) {
      int buf_cur = k_idx % 2;
      int buf_next = (k_idx + 1) % 2;
      if (k_idx < 111) {
        load_tile(buf_next, k_idx + 1);
      }

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

#pragma unroll
      for (int ki = 0; ki < 64; ki += 16) {
        wmma::load_matrix_sync(
            fa, &smem_a[buf_cur * 32 * kSmemPadK + warp_row * kSmemPadK + ki],
            kSmemPadK);

        wmma::load_matrix_sync(
            fb, &smem_w_g[buf_cur * 64 * kSmemPadK + warp_col * kSmemPadK + ki],
            kSmemPadK);
        wmma::mma_sync(acc_g[0], fa, fb, acc_g[0]);
        wmma::load_matrix_sync(
            fb,
            &smem_w_g[buf_cur * 64 * kSmemPadK + (warp_col + 16) * kSmemPadK + ki],
            kSmemPadK);
        wmma::mma_sync(acc_g[1], fa, fb, acc_g[1]);

        wmma::load_matrix_sync(
            fb, &smem_w_u[buf_cur * 64 * kSmemPadK + warp_col * kSmemPadK + ki],
            kSmemPadK);
        wmma::mma_sync(acc_u[0], fa, fb, acc_u[0]);
        wmma::load_matrix_sync(
            fb,
            &smem_w_u[buf_cur * 64 * kSmemPadK + (warp_col + 16) * kSmemPadK + ki],
            kSmemPadK);
        wmma::mma_sync(acc_u[1], fa, fb, acc_u[1]);
      }
      __syncthreads();
    }

    float* smem_out_g = reinterpret_cast<float*>(smem_w_g);
    float* smem_out_u = reinterpret_cast<float*>(smem_w_u);
    wmma::store_matrix_sync(
        smem_out_g + warp_row * 64 + warp_col, acc_g[0], 64, wmma::mem_row_major);
    wmma::store_matrix_sync(
        smem_out_g + warp_row * 64 + warp_col + 16, acc_g[1], 64, wmma::mem_row_major);
    wmma::store_matrix_sync(
        smem_out_u + warp_row * 64 + warp_col, acc_u[0], 64, wmma::mem_row_major);
    wmma::store_matrix_sync(
        smem_out_u + warp_row * 64 + warp_col + 16, acc_u[1], 64, wmma::mem_row_major);
    __syncthreads();

#pragma unroll
    for (int i = tid; i < 2048; i += 128) {
      int r = i / 64;
      int c = i % 64;
      if (r < m_current) {
        float gate = smem_out_g[i];
        float up = smem_out_u[i];
        int dest = expert_offsets[expert_idx] + m_start + r;
        intermediate_buffer[static_cast<int64_t>(dest) * kIntermediateSize + gate_col_start + c] =
            __float2bfloat16(silu_pure(up) * gate);
      }
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(128) void gemm2_kernel_pure(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_fp8_e4m3* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const int* __restrict__ expert_counts,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ sorted_weights,
    __nv_bfloat16* __restrict__ output,
    int seq_len) {
  extern __shared__ char smem_buf[];
  __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_buf);
  __nv_bfloat16* smem_w = smem_a + 2 * 32 * kSmemPadK;

  int tid = threadIdx.x;
  int warp_row = ((tid / 32) / 2) * 16;
  int warp_col = ((tid / 32) % 2) * 32;

  for (int tile_idx = blockIdx.x; tile_idx < kGemm2GridSize; tile_idx += gridDim.x) {
    int expert_idx = tile_idx / 112;
    int n_blk = tile_idx % 112;
    int num_tokens = expert_counts[expert_idx];
    if (num_tokens == 0) {
      continue;
    }

    int col_start = n_blk * 64;
    int scale_n = col_start / 128;
    const float* scale_base =
        &gemm2_weights_scale[(static_cast<int64_t>(expert_idx) * 56 + scale_n) * 16];

    int m_loop_begin = 0;
    int m_loop_end = num_tokens;
    if (gridDim.y > 1) {
      m_loop_begin = static_cast<int>(blockIdx.y) * 32;
      if (m_loop_begin >= num_tokens) {
        continue;
      }
      m_loop_end = min_int(m_loop_begin + 32, num_tokens);
    }

    for (int m_start = m_loop_begin; m_start < m_loop_end; m_start += 32) {
      int m_current = min_int(32, num_tokens - m_start);

      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
      wmma::fill_fragment(acc[0], 0.0f);
      wmma::fill_fragment(acc[1], 0.0f);

      auto load_tile = [&](int buf_idx, int k_idx) {
        int k_base = k_idx * 64;
        int scale_idx = k_idx / 2;

        int row = tid / 4;
        int col = (tid % 4) * 16;
        if (row < 32) {
          __nv_bfloat16* dst =
              &smem_a[buf_idx * 32 * kSmemPadK + row * kSmemPadK + col];
          if (row < m_current) {
            int off = expert_offsets[expert_idx] + m_start + row;
            int4 v0 = *reinterpret_cast<const int4*>(
                &intermediate[static_cast<int64_t>(off) * kIntermediateSize + k_base + col]);
            int4 v1 = *reinterpret_cast<const int4*>(
                &intermediate[static_cast<int64_t>(off) * kIntermediateSize + k_base + col + 8]);
            *reinterpret_cast<int4*>(dst) = v0;
            *reinterpret_cast<int4*>(dst + 8) = v1;
          } else {
            *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(dst + 8) = make_int4(0, 0, 0, 0);
          }
        }

        int t_row = tid / 4;
        int t_col = (tid % 4) * 16;
        float scale = scale_base[scale_idx];

#pragma unroll
        for (int r_off = 0; r_off < 64; r_off += 32) {
          int r = t_row + r_off;
          int c = t_col;
          int64_t offset =
              (static_cast<int64_t>(expert_idx) * kHiddenSize + col_start + r) *
                  kIntermediateSize +
              k_base + c;
          int4 packed = *reinterpret_cast<const int4*>(&gemm2_weights[offset]);

          __nv_fp8_e4m3 vals[16];
          *reinterpret_cast<int4*>(vals) = packed;
          __nv_bfloat16* dst =
              &smem_w[buf_idx * 64 * kSmemPadK + r * kSmemPadK + c];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            dst[i] = __float2bfloat16(static_cast<float>(vals[i]) * scale);
          }
        }
      };

      load_tile(0, 0);
      __syncthreads();

#pragma unroll 1
      for (int k_idx = 0; k_idx < 32; ++k_idx) {
        int buf_cur = k_idx % 2;
        int buf_next = (k_idx + 1) % 2;
        if (k_idx < 31) {
          load_tile(buf_next, k_idx + 1);
        }

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;

#pragma unroll
        for (int ki = 0; ki < 64; ki += 16) {
          wmma::load_matrix_sync(
              fa, &smem_a[buf_cur * 32 * kSmemPadK + warp_row * kSmemPadK + ki],
              kSmemPadK);
          wmma::load_matrix_sync(
              fb, &smem_w[buf_cur * 64 * kSmemPadK + warp_col * kSmemPadK + ki],
              kSmemPadK);
          wmma::mma_sync(acc[0], fa, fb, acc[0]);
          wmma::load_matrix_sync(
              fb, &smem_w[buf_cur * 64 * kSmemPadK + (warp_col + 16) * kSmemPadK + ki],
              kSmemPadK);
          wmma::mma_sync(acc[1], fa, fb, acc[1]);
        }
        __syncthreads();
      }

      float* smem_out = reinterpret_cast<float*>(smem_w);
      wmma::store_matrix_sync(
          smem_out + warp_row * 64 + warp_col, acc[0], 64, wmma::mem_row_major);
      wmma::store_matrix_sync(
          smem_out + warp_row * 64 + warp_col + 16, acc[1], 64, wmma::mem_row_major);
      __syncthreads();

#pragma unroll
      for (int i = tid; i < 32 * 64; i += 128) {
        int r = i / 64;
        int c = i % 64;
        if (r < m_current) {
          int slot = expert_offsets[expert_idx] + m_start + r;
          int token = sorted_token_ids[slot];
          float weight = sorted_weights[slot];
          float val = smem_out[i];
          atomic_add_bf16(
              &output[static_cast<int64_t>(token) * kHiddenSize + col_start + c],
              __float2bfloat16(val * weight));
        }
      }
      __syncthreads();
    }
  }
}

}  // namespace

void launch_moe_pipeline_cuda(
    const float* routing_logits,
    const __nv_bfloat16* routing_bias,
    const __nv_fp8_e4m3* hidden_states,
    const float* hidden_states_scale,
    const __nv_fp8_e4m3* gemm1_weights,
    const float* gemm1_weights_scale,
    const __nv_fp8_e4m3* gemm2_weights,
    const float* gemm2_weights_scale,
    float routed_scaling_factor,
    int seq_len,
    int local_expert_offset,
    __nv_bfloat16* output,
    int* topk_ids,
    float* topk_weights,
    int* expert_counts,
    int* expert_offsets,
    int* sorted_token_ids,
    float* sorted_weights,
    __nv_bfloat16* intermediate_buffer,
    cudaStream_t stream) {
  if (seq_len == 0) {
    return;
  }

  cudaMemsetAsync(expert_counts, 0, kNumLocalExperts * sizeof(int), stream);
  cudaMemsetAsync(output, 0, static_cast<size_t>(seq_len) * kHiddenSize * sizeof(__nv_bfloat16), stream);

  routing_kernel_pure<<<seq_len, 256, 0, stream>>>(
      routing_logits,
      routing_bias,
      routed_scaling_factor,
      seq_len,
      local_expert_offset,
      topk_ids,
      topk_weights,
      expert_counts,
      output);

  sort_scatter_kernel_pure<<<1, 256, 0, stream>>>(
      topk_ids,
      topk_weights,
      seq_len,
      local_expert_offset,
      expert_counts,
      expert_offsets,
      sorted_token_ids,
      sorted_weights);

  int max_m_tiles = 1;
  if (seq_len >= 512) {
    int host_counts[kNumLocalExperts];
    cudaMemcpyAsync(
        host_counts,
        expert_counts,
        kNumLocalExperts * sizeof(int),
        cudaMemcpyDeviceToHost,
        stream);
    cudaStreamSynchronize(stream);
    int max_count = 0;
    for (int i = 0; i < kNumLocalExperts; ++i) {
      if (host_counts[i] > max_count) {
        max_count = host_counts[i];
      }
    }
    max_m_tiles = max_count > 0 ? (max_count + 31) / 32 : 1;
  }

  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(gemm1_kernel_pure, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    cudaFuncSetAttribute(gemm2_kernel_pure, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    configured = true;
  }

  dim3 grid_g1(kGemm1GridX, kNumLocalExperts, max_m_tiles);
  gemm1_kernel_pure<<<grid_g1, 128, 48000, stream>>>(
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      expert_counts,
      expert_offsets,
      sorted_token_ids,
      intermediate_buffer,
      seq_len);

  dim3 grid_g2(kGemm2GridSize, max_m_tiles);
  gemm2_kernel_pure<<<grid_g2, 128, 32000, stream>>>(
      intermediate_buffer,
      gemm2_weights,
      gemm2_weights_scale,
      expert_counts,
      expert_offsets,
      sorted_token_ids,
      sorted_weights,
      output,
      seq_len);
}
