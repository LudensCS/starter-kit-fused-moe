#include "kernel.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kTopK = 8;
constexpr int kNGroup = 8;
constexpr int kTopKGroup = 4;
constexpr int kNumExpertsGlobal = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kExpertsPerGroup = kNumExpertsGlobal / kNGroup;
constexpr int kHiddenSize = 7168;
constexpr int kIntermediateSize = 2048;
constexpr int kGemm1OutSize = 4096;
constexpr int kBlock = 128;
constexpr int kNumHiddenBlocks = kHiddenSize / kBlock;
constexpr int kNumIntermediateBlocks = kIntermediateSize / kBlock;
constexpr int kNumGemm1OutBlocks = kGemm1OutSize / kBlock;
constexpr int kNumPipelineStreams = 4;
constexpr float kEps = 1.0e-20f;
constexpr float kNegInf = -1.0e20f;

#define CUDA_CHECK(expr)                                                                     \
  do {                                                                                       \
    cudaError_t err__ = (expr);                                                              \
    TVM_FFI_ICHECK_EQ(err__, cudaSuccess) << "CUDA failure: " << cudaGetErrorString(err__); \
  } while (0)

#define CUBLAS_CHECK(expr)                                                                   \
  do {                                                                                       \
    cublasStatus_t err__ = (expr);                                                           \
    TVM_FFI_ICHECK_EQ(err__, CUBLAS_STATUS_SUCCESS) << "cuBLAS failure code: " << err__;    \
  } while (0)

struct DeviceBuffer {
  void* ptr = nullptr;
  size_t bytes = 0;

  void ensure(size_t need_bytes) {
    if (need_bytes <= bytes) {
      return;
    }
    if (ptr != nullptr) {
      CUDA_CHECK(cudaFree(ptr));
      ptr = nullptr;
      bytes = 0;
    }
    CUDA_CHECK(cudaMalloc(&ptr, need_bytes));
    bytes = need_bytes;
  }
};

struct ExpertPipeline {
  DeviceBuffer w13_buf;
  DeviceBuffer w2_buf;
  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  cublasHandle_t cublas = nullptr;
  bool initialized = false;

  void ensure_initialized() {
    if (initialized) {
      return;
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    initialized = true;
  }

  void set_stream() {
    ensure_initialized();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
  }
};

struct Workspace {
  DeviceBuffer local_ids;
  DeviceBuffer local_weights;
  DeviceBuffer counts;
  DeviceBuffer write_offsets;
  DeviceBuffer token_ids;
  DeviceBuffer pair_weights;
  DeviceBuffer a_all;
  DeviceBuffer g1_all;
  DeviceBuffer c_all;
  DeviceBuffer o_all;
  DeviceBuffer output_accum;
  cudaEvent_t route_ready_event = nullptr;
  bool route_ready_event_initialized = false;
  std::array<ExpertPipeline, kNumPipelineStreams> pipelines;

  static Workspace& instance() {
    static Workspace ws;
    return ws;
  }

  void ensure_events() {
    if (!route_ready_event_initialized) {
      CUDA_CHECK(cudaEventCreateWithFlags(&route_ready_event, cudaEventDisableTiming));
      route_ready_event_initialized = true;
    }
  }
};

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void routing_pass1_kernel(
    const float* routing_logits,
    const __nv_bfloat16* routing_bias,
    int32_t* local_ids,
    float* local_weights,
    int32_t* counts,
    int32_t seq_len,
    int32_t local_expert_offset,
    float routed_scaling_factor) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= seq_len) {
    return;
  }

  const float* logits_ptr = routing_logits + static_cast<int64_t>(token_idx) * kNumExpertsGlobal;

  float s[kNumExpertsGlobal];
  float s_bias[kNumExpertsGlobal];
  float group_scores[kNGroup];
  bool keep_group[kNGroup];
  int top_experts[kTopK];
  float top_scores[kTopK];
  float top_s[kTopK];

  #pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    top_experts[i] = -1;
    top_scores[i] = kNegInf;
    top_s[i] = 0.0f;
  }

  for (int e = 0; e < kNumExpertsGlobal; ++e) {
    float sig = sigmoidf_fast(logits_ptr[e]);
    s[e] = sig;
    s_bias[e] = sig + __bfloat162float(routing_bias[e]);
  }

  for (int g = 0; g < kNGroup; ++g) {
    float best1 = kNegInf;
    float best2 = kNegInf;
    const int base = g * kExpertsPerGroup;
    for (int e = 0; e < kExpertsPerGroup; ++e) {
      float v = s_bias[base + e];
      if (v > best1) {
        best2 = best1;
        best1 = v;
      } else if (v > best2) {
        best2 = v;
      }
    }
    group_scores[g] = best1 + best2;
    keep_group[g] = false;
  }

  for (int iter = 0; iter < kTopKGroup; ++iter) {
    float best = kNegInf;
    int best_group = -1;
    for (int g = 0; g < kNGroup; ++g) {
      if (!keep_group[g] && group_scores[g] > best) {
        best = group_scores[g];
        best_group = g;
      }
    }
    keep_group[best_group] = true;
  }

  for (int g = 0; g < kNGroup; ++g) {
    if (!keep_group[g]) {
      continue;
    }
    const int base = g * kExpertsPerGroup;
    for (int e = 0; e < kExpertsPerGroup; ++e) {
      const int expert = base + e;
      const float score = s_bias[expert];
      int insert = -1;
      for (int i = 0; i < kTopK; ++i) {
        if (score > top_scores[i]) {
          insert = i;
          break;
        }
      }
      if (insert >= 0) {
        for (int i = kTopK - 1; i > insert; --i) {
          top_scores[i] = top_scores[i - 1];
          top_experts[i] = top_experts[i - 1];
          top_s[i] = top_s[i - 1];
        }
        top_scores[insert] = score;
        top_experts[insert] = expert;
        top_s[insert] = s[expert];
      }
    }
  }

  float sum_s = 0.0f;
  #pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    sum_s += top_s[i];
  }
  sum_s += kEps;

  int32_t* ids_ptr = local_ids + static_cast<int64_t>(token_idx) * kTopK;
  float* weights_ptr = local_weights + static_cast<int64_t>(token_idx) * kTopK;

  #pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    ids_ptr[i] = -1;
    weights_ptr[i] = 0.0f;
    const int expert = top_experts[i];
    const int local_expert = expert - local_expert_offset;
    if (local_expert >= 0 && local_expert < kNumLocalExperts) {
      ids_ptr[i] = local_expert;
      weights_ptr[i] = top_s[i] * routed_scaling_factor / sum_s;
      atomicAdd(counts + local_expert, 1);
    }
  }
}

__global__ void routing_pass2_kernel(
    const int32_t* local_ids,
    const float* local_weights,
    int32_t* write_offsets,
    int32_t* token_ids,
    float* pair_weights,
    int32_t seq_len) {
  int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= seq_len) {
    return;
  }

  const int32_t* ids_ptr = local_ids + static_cast<int64_t>(token_idx) * kTopK;
  const float* weights_ptr = local_weights + static_cast<int64_t>(token_idx) * kTopK;

  #pragma unroll
  for (int i = 0; i < kTopK; ++i) {
    const int local_expert = ids_ptr[i];
    if (local_expert >= 0) {
      const int pos = atomicAdd(write_offsets + local_expert, 1);
      token_ids[pos] = token_idx;
      pair_weights[pos] = weights_ptr[i];
    }
  }
}

__global__ void dequant_hidden_kernel(
    const __nv_fp8_e4m3* hidden_states,
    const float* hidden_states_scale,
    const int32_t* token_ids,
    __nv_bfloat16* output,
    int32_t count) {
  const int token_row = blockIdx.y;
  const int hidden_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_row >= count || hidden_col >= kHiddenSize) {
    return;
  }

  const int32_t token_idx = token_ids[token_row];
  const int scale_block = hidden_col / kBlock;
  const float scale = hidden_states_scale[static_cast<int64_t>(scale_block) * gridDim.y + token_idx];
  const float val = static_cast<float>(hidden_states[static_cast<int64_t>(token_idx) * kHiddenSize + hidden_col]);
  output[static_cast<int64_t>(token_row) * kHiddenSize + hidden_col] = __float2bfloat16(val * scale);
}

__global__ void dequant_hidden_kernel_strided(
    const __nv_fp8_e4m3* hidden_states,
    const float* hidden_states_scale,
    const int32_t* token_ids,
    __nv_bfloat16* output,
    int32_t count,
    int32_t scale_stride) {
  const int token_row = blockIdx.y;
  const int hidden_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_row >= count || hidden_col >= kHiddenSize) {
    return;
  }

  const int32_t token_idx = token_ids[token_row];
  const int scale_block = hidden_col / kBlock;
  const float scale = hidden_states_scale[static_cast<int64_t>(scale_block) * scale_stride + token_idx];
  const float val = static_cast<float>(hidden_states[static_cast<int64_t>(token_idx) * kHiddenSize + hidden_col]);
  output[static_cast<int64_t>(token_row) * kHiddenSize + hidden_col] = __float2bfloat16(val * scale);
}

__global__ void dequant_gemm1_weight_kernel(
    const __nv_fp8_e4m3* gemm1_weights,
    const float* gemm1_weights_scale,
    __nv_bfloat16* output,
    int32_t local_expert) {
  const int hidden_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (hidden_col >= kHiddenSize || out_col >= kGemm1OutSize) {
    return;
  }

  const int scale_n = out_col / kBlock;
  const int scale_k = hidden_col / kBlock;
  const int64_t weight_offset =
      (static_cast<int64_t>(local_expert) * kGemm1OutSize + out_col) * kHiddenSize + hidden_col;
  const int64_t scale_offset =
      (static_cast<int64_t>(local_expert) * kNumGemm1OutBlocks + scale_n) * kNumHiddenBlocks + scale_k;
  const float scale = gemm1_weights_scale[scale_offset];
  const float val = static_cast<float>(gemm1_weights[weight_offset]);
  output[static_cast<int64_t>(out_col) * kHiddenSize + hidden_col] = __float2bfloat16(val * scale);
}

__global__ void dequant_gemm2_weight_kernel(
    const __nv_fp8_e4m3* gemm2_weights,
    const float* gemm2_weights_scale,
    __nv_bfloat16* output,
    int32_t local_expert) {
  const int inter_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  if (inter_col >= kIntermediateSize || out_row >= kHiddenSize) {
    return;
  }

  const int scale_h = out_row / kBlock;
  const int scale_i = inter_col / kBlock;
  const int64_t weight_offset =
      (static_cast<int64_t>(local_expert) * kHiddenSize + out_row) * kIntermediateSize + inter_col;
  const int64_t scale_offset =
      (static_cast<int64_t>(local_expert) * kNumHiddenBlocks + scale_h) * kNumIntermediateBlocks + scale_i;
  const float scale = gemm2_weights_scale[scale_offset];
  const float val = static_cast<float>(gemm2_weights[weight_offset]);
  output[static_cast<int64_t>(out_row) * kIntermediateSize + inter_col] = __float2bfloat16(val * scale);
}

__global__ void swiglu_kernel(
    const __nv_bfloat16* g1,
    __nv_bfloat16* c,
    int32_t count) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count || col >= kIntermediateSize) {
    return;
  }

  const int64_t base = static_cast<int64_t>(row) * kGemm1OutSize;
  const float x1 = __bfloat162float(g1[base + col]);
  const float x2 = __bfloat162float(g1[base + kIntermediateSize + col]);
  const float silu = x2 / (1.0f + expf(-x2));
  c[static_cast<int64_t>(row) * kIntermediateSize + col] = __float2bfloat16(x1 * silu);
}

__global__ void accumulate_kernel(
    const int32_t* token_ids,
    const float* pair_weights,
    const __nv_bfloat16* expert_out,
    float* output_accum,
    int32_t count) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= count || col >= kHiddenSize) {
    return;
  }

  const int token_idx = token_ids[row];
  const float weight = pair_weights[row];
  const float val = __bfloat162float(expert_out[static_cast<int64_t>(row) * kHiddenSize + col]);
  atomicAdd(output_accum + static_cast<int64_t>(token_idx) * kHiddenSize + col, val * weight);
}

__global__ void cast_output_kernel(const float* input, __nv_bfloat16* output, int32_t total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = __float2bfloat16(input[idx]);
}

void gemm_bf16_rowmajor(
    cublasHandle_t handle,
    int m,
    int n,
    int k,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b,
      CUDA_R_16BF,
      k,
      a,
      CUDA_R_16BF,
      k,
      &beta,
      c,
      CUDA_R_16BF,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace

void launch_moe_cuda(
    tvm::ffi::TensorView routing_logits,
    tvm::ffi::TensorView routing_bias,
    tvm::ffi::TensorView hidden_states,
    tvm::ffi::TensorView hidden_states_scale,
    tvm::ffi::TensorView gemm1_weights,
    tvm::ffi::TensorView gemm1_weights_scale,
    tvm::ffi::TensorView gemm2_weights,
    tvm::ffi::TensorView gemm2_weights_scale,
    int32_t local_expert_offset,
    float routed_scaling_factor,
    tvm::ffi::TensorView output) {
  const int32_t seq_len = static_cast<int32_t>(routing_logits.size(0));
  auto& ws = Workspace::instance();

  const auto* routing_logits_ptr = static_cast<const float*>(routing_logits.data_ptr());
  const auto* routing_bias_ptr = static_cast<const __nv_bfloat16*>(routing_bias.data_ptr());
  const auto* hidden_states_ptr = static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr());
  const auto* hidden_states_scale_ptr = static_cast<const float*>(hidden_states_scale.data_ptr());
  const auto* gemm1_weights_ptr = static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr());
  const auto* gemm1_weights_scale_ptr = static_cast<const float*>(gemm1_weights_scale.data_ptr());
  const auto* gemm2_weights_ptr = static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr());
  const auto* gemm2_weights_scale_ptr = static_cast<const float*>(gemm2_weights_scale.data_ptr());
  auto* output_ptr = static_cast<__nv_bfloat16*>(output.data_ptr());

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(output.device().device_type, output.device().device_id));
  ws.ensure_events();
  for (auto& pipeline : ws.pipelines) {
    pipeline.ensure_initialized();
  }

  ws.local_ids.ensure(static_cast<size_t>(seq_len) * kTopK * sizeof(int32_t));
  ws.local_weights.ensure(static_cast<size_t>(seq_len) * kTopK * sizeof(float));
  ws.counts.ensure(kNumLocalExperts * sizeof(int32_t));
  ws.write_offsets.ensure(kNumLocalExperts * sizeof(int32_t));
  ws.output_accum.ensure(static_cast<size_t>(seq_len) * kHiddenSize * sizeof(float));

  CUDA_CHECK(cudaMemsetAsync(ws.counts.ptr, 0, kNumLocalExperts * sizeof(int32_t), stream));
  const int threads = 128;
  const int blocks = (seq_len + threads - 1) / threads;
  routing_pass1_kernel<<<blocks, threads, 0, stream>>>(
      routing_logits_ptr,
      routing_bias_ptr,
      static_cast<int32_t*>(ws.local_ids.ptr),
      static_cast<float*>(ws.local_weights.ptr),
      static_cast<int32_t*>(ws.counts.ptr),
      seq_len,
      local_expert_offset,
      routed_scaling_factor);
  CUDA_CHECK(cudaGetLastError());

  std::array<int32_t, kNumLocalExperts> host_counts{};
  CUDA_CHECK(cudaMemcpyAsync(
      host_counts.data(),
      ws.counts.ptr,
      kNumLocalExperts * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::array<int32_t, kNumLocalExperts> host_offsets{};
  std::vector<int32_t> active_experts;
  int32_t total_pairs = 0;
  for (int e = 0; e < kNumLocalExperts; ++e) {
    host_offsets[e] = total_pairs;
    const int32_t count = host_counts[e];
    if (count > 0) {
      active_experts.push_back(e);
      total_pairs += count;
    }
  }

  if (total_pairs == 0) {
    CUDA_CHECK(cudaMemsetAsync(output_ptr, 0, static_cast<size_t>(seq_len) * kHiddenSize * sizeof(__nv_bfloat16), stream));
    return;
  }

  ws.token_ids.ensure(static_cast<size_t>(total_pairs) * sizeof(int32_t));
  ws.pair_weights.ensure(static_cast<size_t>(total_pairs) * sizeof(float));
  ws.a_all.ensure(static_cast<size_t>(total_pairs) * kHiddenSize * sizeof(__nv_bfloat16));
  ws.g1_all.ensure(static_cast<size_t>(total_pairs) * kGemm1OutSize * sizeof(__nv_bfloat16));
  ws.c_all.ensure(static_cast<size_t>(total_pairs) * kIntermediateSize * sizeof(__nv_bfloat16));
  ws.o_all.ensure(static_cast<size_t>(total_pairs) * kHiddenSize * sizeof(__nv_bfloat16));

  CUDA_CHECK(cudaMemcpyAsync(
      ws.write_offsets.ptr,
      host_offsets.data(),
      kNumLocalExperts * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream));

  routing_pass2_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(ws.local_ids.ptr),
      static_cast<const float*>(ws.local_weights.ptr),
      static_cast<int32_t*>(ws.write_offsets.ptr),
      static_cast<int32_t*>(ws.token_ids.ptr),
      static_cast<float*>(ws.pair_weights.ptr),
      seq_len);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(
      ws.output_accum.ptr, 0, static_cast<size_t>(seq_len) * kHiddenSize * sizeof(float), stream));

  dim3 act_block(256);
  dim3 g1_weight_block(16, 16);
  dim3 g1_weight_grid(
      (kHiddenSize + g1_weight_block.x - 1) / g1_weight_block.x,
      (kGemm1OutSize + g1_weight_block.y - 1) / g1_weight_block.y);
  dim3 g2_weight_block(16, 16);
  dim3 g2_weight_grid(
      (kIntermediateSize + g2_weight_block.x - 1) / g2_weight_block.x,
      (kHiddenSize + g2_weight_block.y - 1) / g2_weight_block.y);
  dim3 swiglu_block(256);

  auto* token_ids_ptr = static_cast<const int32_t*>(ws.token_ids.ptr);
  auto* pair_weights_ptr = static_cast<const float*>(ws.pair_weights.ptr);
  auto* a_all_ptr = static_cast<__nv_bfloat16*>(ws.a_all.ptr);
  auto* g1_all_ptr = static_cast<__nv_bfloat16*>(ws.g1_all.ptr);
  auto* c_all_ptr = static_cast<__nv_bfloat16*>(ws.c_all.ptr);
  auto* o_all_ptr = static_cast<__nv_bfloat16*>(ws.o_all.ptr);
  auto* output_accum_ptr = static_cast<float*>(ws.output_accum.ptr);

  std::vector<int32_t> sorted_experts = active_experts;
  std::sort(sorted_experts.begin(), sorted_experts.end(), [&](int32_t lhs, int32_t rhs) {
    return host_counts[lhs] > host_counts[rhs];
  });

  std::array<std::vector<int32_t>, kNumPipelineStreams> stream_experts;
  std::array<int64_t, kNumPipelineStreams> stream_loads{};

  for (int32_t expert : sorted_experts) {
    int best_stream = 0;
    for (int s = 1; s < kNumPipelineStreams; ++s) {
      if (stream_loads[s] < stream_loads[best_stream]) {
        best_stream = s;
      }
    }
    stream_experts[best_stream].push_back(expert);
    stream_loads[best_stream] += host_counts[expert];
  }

  dim3 all_pairs_act_grid((kHiddenSize + act_block.x - 1) / act_block.x, total_pairs);
  dequant_hidden_kernel_strided<<<all_pairs_act_grid, act_block, 0, stream>>>(
      hidden_states_ptr,
      hidden_states_scale_ptr,
      token_ids_ptr,
      a_all_ptr,
      total_pairs,
      seq_len);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(ws.route_ready_event, stream));

  for (int s = 0; s < kNumPipelineStreams; ++s) {
    if (stream_experts[s].empty()) {
      continue;
    }

    auto& pipeline = ws.pipelines[s];
    pipeline.set_stream();
    CUDA_CHECK(cudaStreamWaitEvent(pipeline.stream, ws.route_ready_event, 0));

    pipeline.w13_buf.ensure(static_cast<size_t>(kGemm1OutSize) * kHiddenSize * sizeof(__nv_bfloat16));
    pipeline.w2_buf.ensure(static_cast<size_t>(kHiddenSize) * kIntermediateSize * sizeof(__nv_bfloat16));

    auto* w13_buf_ptr = static_cast<__nv_bfloat16*>(pipeline.w13_buf.ptr);
    auto* w2_buf_ptr = static_cast<__nv_bfloat16*>(pipeline.w2_buf.ptr);

    for (int32_t expert : stream_experts[s]) {
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];

      dequant_gemm1_weight_kernel<<<g1_weight_grid, g1_weight_block, 0, pipeline.stream>>>(
          gemm1_weights_ptr, gemm1_weights_scale_ptr, w13_buf_ptr, expert);
      CUDA_CHECK(cudaGetLastError());

      gemm_bf16_rowmajor(
          pipeline.cublas,
          count,
          kGemm1OutSize,
          kHiddenSize,
          a_all_ptr + static_cast<int64_t>(start) * kHiddenSize,
          w13_buf_ptr,
          g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize);
    }

    CUDA_CHECK(cudaEventRecord(pipeline.done_event, pipeline.stream));
  }

  for (int s = 0; s < kNumPipelineStreams; ++s) {
    if (!stream_experts[s].empty()) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ws.pipelines[s].done_event, 0));
    }
  }

  dim3 all_pairs_swiglu_grid((kIntermediateSize + swiglu_block.x - 1) / swiglu_block.x, total_pairs);
  swiglu_kernel<<<all_pairs_swiglu_grid, swiglu_block, 0, stream>>>(g1_all_ptr, c_all_ptr, total_pairs);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(ws.route_ready_event, stream));

  for (int s = 0; s < kNumPipelineStreams; ++s) {
    if (stream_experts[s].empty()) {
      continue;
    }

    auto& pipeline = ws.pipelines[s];
    pipeline.set_stream();
    CUDA_CHECK(cudaStreamWaitEvent(pipeline.stream, ws.route_ready_event, 0));

    auto* w2_buf_ptr = static_cast<__nv_bfloat16*>(pipeline.w2_buf.ptr);

    for (int32_t expert : stream_experts[s]) {
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];

      dequant_gemm2_weight_kernel<<<g2_weight_grid, g2_weight_block, 0, pipeline.stream>>>(
          gemm2_weights_ptr, gemm2_weights_scale_ptr, w2_buf_ptr, expert);
      CUDA_CHECK(cudaGetLastError());

      gemm_bf16_rowmajor(
          pipeline.cublas,
          count,
          kHiddenSize,
          kIntermediateSize,
          c_all_ptr + static_cast<int64_t>(start) * kIntermediateSize,
          w2_buf_ptr,
          o_all_ptr + static_cast<int64_t>(start) * kHiddenSize);
    }

    CUDA_CHECK(cudaEventRecord(pipeline.done_event, pipeline.stream));
  }

  for (int s = 0; s < kNumPipelineStreams; ++s) {
    if (!stream_experts[s].empty()) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ws.pipelines[s].done_event, 0));
    }
  }

  accumulate_kernel<<<all_pairs_act_grid, act_block, 0, stream>>>(
      token_ids_ptr,
      pair_weights_ptr,
      o_all_ptr,
      output_accum_ptr,
      total_pairs);
  CUDA_CHECK(cudaGetLastError());

  const int total = seq_len * kHiddenSize;
  const int cast_blocks = (total + 255) / 256;
  cast_output_kernel<<<cast_blocks, 256, 0, stream>>>(output_accum_ptr, output_ptr, total);
  CUDA_CHECK(cudaGetLastError());
}
