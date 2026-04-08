#include "kernel.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
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
constexpr int kNumPipelineStreams = 1;
constexpr int kExpertTokenAlign = 128;
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
  DeviceBuffer a_bf16_buf;
  DeviceBuffer a_fp8_buf;
  DeviceBuffer a_scale_buf;
  DeviceBuffer w13_buf;
  DeviceBuffer g1_tmp_buf;
  DeviceBuffer c_fp8_buf;
  DeviceBuffer c_scale_buf;
  DeviceBuffer w2_buf;
  DeviceBuffer o_tmp_buf;
  DeviceBuffer lt_workspace;
  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  cublasHandle_t cublas = nullptr;
  cublasLtHandle_t cublaslt = nullptr;
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
    CUBLAS_CHECK(cublasLtCreate(&cublaslt));
    lt_workspace.ensure(32u * 1024u * 1024u);
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

uint64_t make_lt_algo_key_with_batch(int batch_count, int m, int n, int k) {
  return (static_cast<uint64_t>(batch_count) << 56) ^ (static_cast<uint64_t>(m) << 38) ^
         (static_cast<uint64_t>(n) << 19) ^ static_cast<uint64_t>(k);
}

uint64_t make_lt_algo_key(int m, int n, int k) {
  return make_lt_algo_key_with_batch(1, m, n, k);
}

int align_up(int value, int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct ExpertRun {
  int32_t first_expert = 0;
  int32_t batch_size = 0;
  int32_t padded_count = 0;
};

std::vector<ExpertRun> build_expert_runs(
    const std::vector<int32_t>& experts, const std::array<int32_t, kNumLocalExperts>& host_counts) {
  std::vector<ExpertRun> runs;
  if (experts.empty()) {
    return runs;
  }

  ExpertRun run{};
  run.first_expert = experts[0];
  run.batch_size = 1;
  run.padded_count = align_up(host_counts[experts[0]], kExpertTokenAlign);

  for (size_t i = 1; i < experts.size(); ++i) {
    const int32_t expert = experts[i];
    const int32_t padded_count = align_up(host_counts[expert], kExpertTokenAlign);
    if (expert == experts[i - 1] + 1 && padded_count == run.padded_count) {
      ++run.batch_size;
      continue;
    }
    runs.push_back(run);
    run.first_expert = expert;
    run.batch_size = 1;
    run.padded_count = padded_count;
  }
  runs.push_back(run);
  return runs;
}

cublasLtMatmulAlgo_t get_fp8_blockscale_algo(
    cublasLtHandle_t handle,
    int batch_count,
    int m,
    int n,
    int k,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t d_batch_stride,
    size_t workspace_size,
    bool* supported) {
  struct CacheEntry {
    bool supported;
    cublasLtMatmulAlgo_t algo;
  };
  static std::unordered_map<uint64_t, CacheEntry> cache;

  const uint64_t key = make_lt_algo_key_with_batch(batch_count, m, n, k);
  auto it = cache.find(key);
  if (it != cache.end()) {
    *supported = it->second.supported;
    return it->second.algo;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
  int8_t fast_accum = 1;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, n, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n));
  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  if (batch_count > 1) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_batch_stride, sizeof(a_batch_stride)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_batch_stride, sizeof(b_batch_stride)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &d_batch_stride, sizeof(d_batch_stride)));
  }

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_results = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic_result, &returned_results);
  const bool ok = status == CUBLAS_STATUS_SUCCESS && returned_results > 0;
  CacheEntry entry{};
  entry.supported = ok;
  if (ok) {
    entry.algo = heuristic_result.algo;
  }
  cache.emplace(key, entry);
  *supported = ok;

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));

  return entry.algo;
}

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

__global__ void gather_hidden_fp8_kernel(
    const __nv_fp8_e4m3* hidden_states,
    const int32_t* token_ids,
    __nv_fp8_e4m3* output,
    int32_t count) {
  const int token_row = blockIdx.y;
  const int hidden_col = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_row >= count || hidden_col >= kHiddenSize) {
    return;
  }
  const int32_t token_idx = token_ids[token_row];
  output[static_cast<int64_t>(token_row) * kHiddenSize + hidden_col] =
      hidden_states[static_cast<int64_t>(token_idx) * kHiddenSize + hidden_col];
}

__global__ void gather_hidden_scale_kernel(
    const float* hidden_states_scale,
    const int32_t* token_ids,
    float* output,
    int32_t count,
    int32_t scale_stride,
    int32_t output_stride) {
  const int token_row = blockIdx.x * blockDim.x + threadIdx.x;
  const int scale_block = blockIdx.y;
  if (token_row >= count || scale_block >= kNumHiddenBlocks) {
    return;
  }
  const int32_t token_idx = token_ids[token_row];
  output[static_cast<int64_t>(scale_block) * output_stride + token_row] =
      hidden_states_scale[static_cast<int64_t>(scale_block) * scale_stride + token_idx];
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

__global__ void quantize_bf16_to_fp8_block128_kernel(
    const __nv_bfloat16* input,
    __nv_fp8_e4m3* output,
    float* scales,
    int32_t count,
    int32_t output_stride,
    int32_t width) {
  __shared__ float shared_abs_max;

  const int32_t row = blockIdx.y;
  const int32_t block_col = blockIdx.x;
  const int32_t lane = threadIdx.x;
  if (row >= count || block_col * kBlock + lane >= width) {
    return;
  }

  const int32_t col = block_col * kBlock + lane;
  const float val = __bfloat162float(input[static_cast<int64_t>(row) * width + col]);
  float abs_val = fabsf(val);

  if (lane == 0) {
    shared_abs_max = 0.0f;
  }
  __syncthreads();

  atomicMax(reinterpret_cast<int*>(&shared_abs_max), __float_as_int(abs_val));
  __syncthreads();

  const float abs_max = shared_abs_max;
  const float scale = abs_max > 0.0f ? abs_max / 448.0f : 1.0f;
  if (lane == 0) {
    scales[static_cast<int64_t>(block_col) * output_stride + row] = scale;
  }
  const float inv_scale = abs_max > 0.0f ? 1.0f / scale : 0.0f;
  output[static_cast<int64_t>(row) * width + col] = static_cast<__nv_fp8_e4m3>(val * inv_scale);
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

bool gemm_fp8_blockscale_rowmajor(
    cublasLtHandle_t handle,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_size,
    int m,
    int n,
    int k,
    const __nv_fp8_e4m3* a,
    const float* a_scale,
    const __nv_fp8_e4m3* b,
    const float* b_scale,
    __nv_bfloat16* d) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

  int8_t fast_accum = 1;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, n, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n));

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  bool supported = false;
  cublasLtMatmulAlgo_t algo = get_fp8_blockscale_algo(
      handle, 1, m, n, k, static_cast<int64_t>(m) * k, static_cast<int64_t>(n) * k,
      static_cast<int64_t>(m) * n, workspace_size, &supported);
  if (!supported) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
    return false;
  }

  cublasStatus_t status = cublasLtMatmul(
      handle,
      op_desc,
      &alpha,
      a,
      a_desc,
      b,
      b_desc,
      &beta,
      d,
      c_desc,
      d,
      c_desc,
      &algo,
      workspace,
      workspace_size,
      stream);

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  return status == CUBLAS_STATUS_SUCCESS;
}

bool gemm_fp8_blockscale_rowmajor_batched(
    cublasLtHandle_t handle,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_size,
    int batch_count,
    int m,
    int n,
    int k,
    const __nv_fp8_e4m3* a,
    const float* a_scale,
    const __nv_fp8_e4m3* b,
    const float* b_scale,
    __nv_bfloat16* d) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  const int64_t a_batch_stride = static_cast<int64_t>(m) * k;
  const int64_t b_batch_stride = static_cast<int64_t>(n) * k;
  const int64_t d_batch_stride = static_cast<int64_t>(m) * n;

  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_T;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatmulMatrixScale_t a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
  cublasLtMatmulMatrixScale_t b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

  int8_t fast_accum = 1;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, m, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, n, k, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, m, n, n));

  cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_batch_stride, sizeof(a_batch_stride)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_batch_stride, sizeof(b_batch_stride)));
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &d_batch_stride, sizeof(d_batch_stride)));

  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  bool supported = false;
  cublasLtMatmulAlgo_t algo = get_fp8_blockscale_algo(
      handle, batch_count, m, n, k, a_batch_stride, b_batch_stride, d_batch_stride, workspace_size,
      &supported);
  if (!supported) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
    return false;
  }

  cublasStatus_t status = cublasLtMatmul(
      handle,
      op_desc,
      &alpha,
      a,
      a_desc,
      b,
      b_desc,
      &beta,
      d,
      c_desc,
      d,
      c_desc,
      &algo,
      workspace,
      workspace_size,
      stream);

  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  return status == CUBLAS_STATUS_SUCCESS;
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
  auto* g1_all_ptr = static_cast<__nv_bfloat16*>(ws.g1_all.ptr);
  auto* c_all_ptr = static_cast<__nv_bfloat16*>(ws.c_all.ptr);
  auto* o_all_ptr = static_cast<__nv_bfloat16*>(ws.o_all.ptr);
  auto* output_accum_ptr = static_cast<float*>(ws.output_accum.ptr);

  std::array<std::vector<int32_t>, kNumPipelineStreams> stream_experts;
  stream_experts[0] = active_experts;
  const auto runs = build_expert_runs(stream_experts[0], host_counts);

  auto& pipeline = ws.pipelines[0];
  pipeline.set_stream();

  for (const ExpertRun& run : runs) {
    if (run.batch_size == 1) {
      const int32_t expert = run.first_expert;
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];
      const int32_t padded_count = run.padded_count;

      pipeline.a_fp8_buf.ensure(static_cast<size_t>(padded_count) * kHiddenSize * sizeof(__nv_fp8_e4m3));
      pipeline.a_scale_buf.ensure(static_cast<size_t>(kNumHiddenBlocks) * padded_count * sizeof(float));
      pipeline.g1_tmp_buf.ensure(static_cast<size_t>(padded_count) * kGemm1OutSize * sizeof(__nv_bfloat16));

      auto* a_fp8_ptr = static_cast<__nv_fp8_e4m3*>(pipeline.a_fp8_buf.ptr);
      auto* a_scale_ptr = static_cast<float*>(pipeline.a_scale_buf.ptr);
      auto* g1_tmp_ptr = static_cast<__nv_bfloat16*>(pipeline.g1_tmp_buf.ptr);

      CUDA_CHECK(cudaMemsetAsync(
          a_fp8_ptr, 0, static_cast<size_t>(padded_count) * kHiddenSize * sizeof(__nv_fp8_e4m3),
          pipeline.stream));
      CUDA_CHECK(cudaMemsetAsync(
          a_scale_ptr, 0, static_cast<size_t>(kNumHiddenBlocks) * padded_count * sizeof(float),
          pipeline.stream));

      dim3 gather_fp8_grid((kHiddenSize + act_block.x - 1) / act_block.x, count);
      gather_hidden_fp8_kernel<<<gather_fp8_grid, act_block, 0, pipeline.stream>>>(
          hidden_states_ptr, token_ids_ptr + start, a_fp8_ptr, count);
      CUDA_CHECK(cudaGetLastError());

      dim3 gather_scale_grid((count + 255) / 256, kNumHiddenBlocks);
      gather_hidden_scale_kernel<<<gather_scale_grid, 256, 0, pipeline.stream>>>(
          hidden_states_scale_ptr, token_ids_ptr + start, a_scale_ptr, count, seq_len, padded_count);
      CUDA_CHECK(cudaGetLastError());

      const auto* w13_expert_ptr =
          gemm1_weights_ptr + static_cast<int64_t>(expert) * kGemm1OutSize * kHiddenSize;
      const auto* w13_scale_expert_ptr =
          gemm1_weights_scale_ptr + static_cast<int64_t>(expert) * kNumGemm1OutBlocks * kNumHiddenBlocks;
      auto* g1_dst_ptr = padded_count == count ? (g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize)
                                               : g1_tmp_ptr;

      const bool fp8_ok = gemm_fp8_blockscale_rowmajor(
          pipeline.cublaslt, pipeline.stream, pipeline.lt_workspace.ptr, pipeline.lt_workspace.bytes,
          padded_count, kGemm1OutSize, kHiddenSize, a_fp8_ptr, a_scale_ptr, w13_expert_ptr,
          w13_scale_expert_ptr, g1_dst_ptr);

      if (fp8_ok) {
        if (padded_count != count) {
          CUDA_CHECK(cudaMemcpyAsync(
              g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize, g1_tmp_ptr,
              static_cast<size_t>(count) * kGemm1OutSize * sizeof(__nv_bfloat16),
              cudaMemcpyDeviceToDevice, pipeline.stream));
        }
      } else {
        pipeline.a_bf16_buf.ensure(static_cast<size_t>(count) * kHiddenSize * sizeof(__nv_bfloat16));
        pipeline.w13_buf.ensure(static_cast<size_t>(kGemm1OutSize) * kHiddenSize * sizeof(__nv_bfloat16));
        auto* a_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.a_bf16_buf.ptr);
        auto* w13_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.w13_buf.ptr);

        dim3 this_act_grid((kHiddenSize + act_block.x - 1) / act_block.x, count);
        dequant_hidden_kernel_strided<<<this_act_grid, act_block, 0, pipeline.stream>>>(
            hidden_states_ptr, hidden_states_scale_ptr, token_ids_ptr + start, a_bf16_ptr, count, seq_len);
        CUDA_CHECK(cudaGetLastError());

        dequant_gemm1_weight_kernel<<<g1_weight_grid, g1_weight_block, 0, pipeline.stream>>>(
            gemm1_weights_ptr, gemm1_weights_scale_ptr, w13_bf16_ptr, expert);
        CUDA_CHECK(cudaGetLastError());

        gemm_bf16_rowmajor(
            pipeline.cublas, count, kGemm1OutSize, kHiddenSize, a_bf16_ptr, w13_bf16_ptr,
            g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize);
      }
      continue;
    }

    const int32_t padded_count = run.padded_count;
    const int32_t batch_size = run.batch_size;
    const int32_t first_expert = run.first_expert;

    pipeline.a_fp8_buf.ensure(
        static_cast<size_t>(batch_size) * padded_count * kHiddenSize * sizeof(__nv_fp8_e4m3));
    pipeline.a_scale_buf.ensure(
        static_cast<size_t>(batch_size) * kNumHiddenBlocks * padded_count * sizeof(float));
    pipeline.g1_tmp_buf.ensure(
        static_cast<size_t>(batch_size) * padded_count * kGemm1OutSize * sizeof(__nv_bfloat16));

    auto* a_fp8_ptr = static_cast<__nv_fp8_e4m3*>(pipeline.a_fp8_buf.ptr);
    auto* a_scale_ptr = static_cast<float*>(pipeline.a_scale_buf.ptr);
    auto* g1_tmp_ptr = static_cast<__nv_bfloat16*>(pipeline.g1_tmp_buf.ptr);

    CUDA_CHECK(cudaMemsetAsync(
        a_fp8_ptr,
        0,
        static_cast<size_t>(batch_size) * padded_count * kHiddenSize * sizeof(__nv_fp8_e4m3),
        pipeline.stream));
    CUDA_CHECK(cudaMemsetAsync(
        a_scale_ptr,
        0,
        static_cast<size_t>(batch_size) * kNumHiddenBlocks * padded_count * sizeof(float),
        pipeline.stream));

    for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int32_t expert = first_expert + batch_idx;
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];

      dim3 gather_fp8_grid((kHiddenSize + act_block.x - 1) / act_block.x, count);
      gather_hidden_fp8_kernel<<<gather_fp8_grid, act_block, 0, pipeline.stream>>>(
          hidden_states_ptr,
          token_ids_ptr + start,
          a_fp8_ptr + static_cast<int64_t>(batch_idx) * padded_count * kHiddenSize,
          count);
      CUDA_CHECK(cudaGetLastError());

      dim3 gather_scale_grid((count + 255) / 256, kNumHiddenBlocks);
      gather_hidden_scale_kernel<<<gather_scale_grid, 256, 0, pipeline.stream>>>(
          hidden_states_scale_ptr,
          token_ids_ptr + start,
          a_scale_ptr + static_cast<int64_t>(batch_idx) * kNumHiddenBlocks * padded_count,
          count,
          seq_len,
          padded_count);
      CUDA_CHECK(cudaGetLastError());
    }

    const auto* w13_expert_ptr =
        gemm1_weights_ptr + static_cast<int64_t>(first_expert) * kGemm1OutSize * kHiddenSize;
    const auto* w13_scale_expert_ptr =
        gemm1_weights_scale_ptr +
        static_cast<int64_t>(first_expert) * kNumGemm1OutBlocks * kNumHiddenBlocks;

    const bool fp8_ok = gemm_fp8_blockscale_rowmajor_batched(
        pipeline.cublaslt,
        pipeline.stream,
        pipeline.lt_workspace.ptr,
        pipeline.lt_workspace.bytes,
        batch_size,
        padded_count,
        kGemm1OutSize,
        kHiddenSize,
        a_fp8_ptr,
        a_scale_ptr,
        w13_expert_ptr,
        w13_scale_expert_ptr,
        g1_tmp_ptr);

    if (fp8_ok) {
      for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t expert = first_expert + batch_idx;
        const int32_t start = host_offsets[expert];
        const int32_t count = host_counts[expert];
        CUDA_CHECK(cudaMemcpyAsync(
            g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize,
            g1_tmp_ptr + static_cast<int64_t>(batch_idx) * padded_count * kGemm1OutSize,
            static_cast<size_t>(count) * kGemm1OutSize * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            pipeline.stream));
      }
    } else {
      for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t expert = first_expert + batch_idx;
        const int32_t start = host_offsets[expert];
        const int32_t count = host_counts[expert];
        pipeline.a_bf16_buf.ensure(static_cast<size_t>(count) * kHiddenSize * sizeof(__nv_bfloat16));
        pipeline.w13_buf.ensure(static_cast<size_t>(kGemm1OutSize) * kHiddenSize * sizeof(__nv_bfloat16));
        auto* a_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.a_bf16_buf.ptr);
        auto* w13_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.w13_buf.ptr);

        dim3 this_act_grid((kHiddenSize + act_block.x - 1) / act_block.x, count);
        dequant_hidden_kernel_strided<<<this_act_grid, act_block, 0, pipeline.stream>>>(
            hidden_states_ptr, hidden_states_scale_ptr, token_ids_ptr + start, a_bf16_ptr, count, seq_len);
        CUDA_CHECK(cudaGetLastError());

        dequant_gemm1_weight_kernel<<<g1_weight_grid, g1_weight_block, 0, pipeline.stream>>>(
            gemm1_weights_ptr, gemm1_weights_scale_ptr, w13_bf16_ptr, expert);
        CUDA_CHECK(cudaGetLastError());

        gemm_bf16_rowmajor(
            pipeline.cublas, count, kGemm1OutSize, kHiddenSize, a_bf16_ptr, w13_bf16_ptr,
            g1_all_ptr + static_cast<int64_t>(start) * kGemm1OutSize);
      }
    }
  }

  CUDA_CHECK(cudaEventRecord(pipeline.done_event, pipeline.stream));
  CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline.done_event, 0));

  dim3 all_pairs_act_grid((kHiddenSize + act_block.x - 1) / act_block.x, total_pairs);
  dim3 all_pairs_swiglu_grid((kIntermediateSize + swiglu_block.x - 1) / swiglu_block.x, total_pairs);
  swiglu_kernel<<<all_pairs_swiglu_grid, swiglu_block, 0, stream>>>(g1_all_ptr, c_all_ptr, total_pairs);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(ws.route_ready_event, stream));
  pipeline.set_stream();
  CUDA_CHECK(cudaStreamWaitEvent(pipeline.stream, ws.route_ready_event, 0));

  for (const ExpertRun& run : runs) {
    if (run.batch_size == 1) {
      const int32_t expert = run.first_expert;
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];
      const int32_t padded_count = run.padded_count;

      pipeline.c_fp8_buf.ensure(
          static_cast<size_t>(padded_count) * kIntermediateSize * sizeof(__nv_fp8_e4m3));
      pipeline.c_scale_buf.ensure(
          static_cast<size_t>(kNumIntermediateBlocks) * padded_count * sizeof(float));
      pipeline.o_tmp_buf.ensure(
          static_cast<size_t>(padded_count) * kHiddenSize * sizeof(__nv_bfloat16));

      auto* c_fp8_ptr = static_cast<__nv_fp8_e4m3*>(pipeline.c_fp8_buf.ptr);
      auto* c_scale_ptr = static_cast<float*>(pipeline.c_scale_buf.ptr);
      auto* o_tmp_ptr = static_cast<__nv_bfloat16*>(pipeline.o_tmp_buf.ptr);

      CUDA_CHECK(cudaMemsetAsync(
          c_fp8_ptr, 0, static_cast<size_t>(padded_count) * kIntermediateSize * sizeof(__nv_fp8_e4m3),
          pipeline.stream));
      CUDA_CHECK(cudaMemsetAsync(
          c_scale_ptr, 0, static_cast<size_t>(kNumIntermediateBlocks) * padded_count * sizeof(float),
          pipeline.stream));

      dim3 quant_grid(kNumIntermediateBlocks, count);
      quantize_bf16_to_fp8_block128_kernel<<<quant_grid, kBlock, 0, pipeline.stream>>>(
          c_all_ptr + static_cast<int64_t>(start) * kIntermediateSize,
          c_fp8_ptr,
          c_scale_ptr,
          count,
          padded_count,
          kIntermediateSize);
      CUDA_CHECK(cudaGetLastError());

      const auto* w2_expert_ptr =
          gemm2_weights_ptr + static_cast<int64_t>(expert) * kHiddenSize * kIntermediateSize;
      const auto* w2_scale_expert_ptr =
          gemm2_weights_scale_ptr + static_cast<int64_t>(expert) * kNumHiddenBlocks * kNumIntermediateBlocks;
      auto* o_dst_ptr = padded_count == count ? (o_all_ptr + static_cast<int64_t>(start) * kHiddenSize)
                                              : o_tmp_ptr;

      const bool fp8_ok = gemm_fp8_blockscale_rowmajor(
          pipeline.cublaslt, pipeline.stream, pipeline.lt_workspace.ptr, pipeline.lt_workspace.bytes,
          padded_count, kHiddenSize, kIntermediateSize, c_fp8_ptr, c_scale_ptr, w2_expert_ptr,
          w2_scale_expert_ptr, o_dst_ptr);

      if (fp8_ok) {
        if (padded_count != count) {
          CUDA_CHECK(cudaMemcpyAsync(
              o_all_ptr + static_cast<int64_t>(start) * kHiddenSize, o_tmp_ptr,
              static_cast<size_t>(count) * kHiddenSize * sizeof(__nv_bfloat16),
              cudaMemcpyDeviceToDevice, pipeline.stream));
        }
      } else {
        pipeline.w2_buf.ensure(static_cast<size_t>(kHiddenSize) * kIntermediateSize * sizeof(__nv_bfloat16));
        auto* w2_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.w2_buf.ptr);
        dequant_gemm2_weight_kernel<<<g2_weight_grid, g2_weight_block, 0, pipeline.stream>>>(
            gemm2_weights_ptr, gemm2_weights_scale_ptr, w2_bf16_ptr, expert);
        CUDA_CHECK(cudaGetLastError());

        gemm_bf16_rowmajor(
            pipeline.cublas, count, kHiddenSize, kIntermediateSize,
            c_all_ptr + static_cast<int64_t>(start) * kIntermediateSize, w2_bf16_ptr,
            o_all_ptr + static_cast<int64_t>(start) * kHiddenSize);
      }
      continue;
    }

    const int32_t padded_count = run.padded_count;
    const int32_t batch_size = run.batch_size;
    const int32_t first_expert = run.first_expert;

    pipeline.c_fp8_buf.ensure(
        static_cast<size_t>(batch_size) * padded_count * kIntermediateSize * sizeof(__nv_fp8_e4m3));
    pipeline.c_scale_buf.ensure(
        static_cast<size_t>(batch_size) * kNumIntermediateBlocks * padded_count * sizeof(float));
    pipeline.o_tmp_buf.ensure(
        static_cast<size_t>(batch_size) * padded_count * kHiddenSize * sizeof(__nv_bfloat16));

    auto* c_fp8_ptr = static_cast<__nv_fp8_e4m3*>(pipeline.c_fp8_buf.ptr);
    auto* c_scale_ptr = static_cast<float*>(pipeline.c_scale_buf.ptr);
    auto* o_tmp_ptr = static_cast<__nv_bfloat16*>(pipeline.o_tmp_buf.ptr);

    CUDA_CHECK(cudaMemsetAsync(
        c_fp8_ptr,
        0,
        static_cast<size_t>(batch_size) * padded_count * kIntermediateSize * sizeof(__nv_fp8_e4m3),
        pipeline.stream));
    CUDA_CHECK(cudaMemsetAsync(
        c_scale_ptr,
        0,
        static_cast<size_t>(batch_size) * kNumIntermediateBlocks * padded_count * sizeof(float),
        pipeline.stream));

    for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const int32_t expert = first_expert + batch_idx;
      const int32_t start = host_offsets[expert];
      const int32_t count = host_counts[expert];
      dim3 quant_grid(kNumIntermediateBlocks, count);
      quantize_bf16_to_fp8_block128_kernel<<<quant_grid, kBlock, 0, pipeline.stream>>>(
          c_all_ptr + static_cast<int64_t>(start) * kIntermediateSize,
          c_fp8_ptr + static_cast<int64_t>(batch_idx) * padded_count * kIntermediateSize,
          c_scale_ptr + static_cast<int64_t>(batch_idx) * kNumIntermediateBlocks * padded_count,
          count,
          padded_count,
          kIntermediateSize);
      CUDA_CHECK(cudaGetLastError());
    }

    const auto* w2_expert_ptr =
        gemm2_weights_ptr + static_cast<int64_t>(first_expert) * kHiddenSize * kIntermediateSize;
    const auto* w2_scale_expert_ptr =
        gemm2_weights_scale_ptr +
        static_cast<int64_t>(first_expert) * kNumHiddenBlocks * kNumIntermediateBlocks;

    const bool fp8_ok = gemm_fp8_blockscale_rowmajor_batched(
        pipeline.cublaslt,
        pipeline.stream,
        pipeline.lt_workspace.ptr,
        pipeline.lt_workspace.bytes,
        batch_size,
        padded_count,
        kHiddenSize,
        kIntermediateSize,
        c_fp8_ptr,
        c_scale_ptr,
        w2_expert_ptr,
        w2_scale_expert_ptr,
        o_tmp_ptr);

    if (fp8_ok) {
      for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t expert = first_expert + batch_idx;
        const int32_t start = host_offsets[expert];
        const int32_t count = host_counts[expert];
        CUDA_CHECK(cudaMemcpyAsync(
            o_all_ptr + static_cast<int64_t>(start) * kHiddenSize,
            o_tmp_ptr + static_cast<int64_t>(batch_idx) * padded_count * kHiddenSize,
            static_cast<size_t>(count) * kHiddenSize * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            pipeline.stream));
      }
    } else {
      for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const int32_t expert = first_expert + batch_idx;
        const int32_t start = host_offsets[expert];
        const int32_t count = host_counts[expert];
        pipeline.w2_buf.ensure(static_cast<size_t>(kHiddenSize) * kIntermediateSize * sizeof(__nv_bfloat16));
        auto* w2_bf16_ptr = static_cast<__nv_bfloat16*>(pipeline.w2_buf.ptr);
        dequant_gemm2_weight_kernel<<<g2_weight_grid, g2_weight_block, 0, pipeline.stream>>>(
            gemm2_weights_ptr, gemm2_weights_scale_ptr, w2_bf16_ptr, expert);
        CUDA_CHECK(cudaGetLastError());

        gemm_bf16_rowmajor(
            pipeline.cublas, count, kHiddenSize, kIntermediateSize,
            c_all_ptr + static_cast<int64_t>(start) * kIntermediateSize, w2_bf16_ptr,
            o_all_ptr + static_cast<int64_t>(start) * kHiddenSize);
      }
    }
  }

  CUDA_CHECK(cudaEventRecord(pipeline.done_event, pipeline.stream));
  CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline.done_event, 0));

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
