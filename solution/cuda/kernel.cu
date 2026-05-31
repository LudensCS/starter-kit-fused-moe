#include "kernel.h"
#include "pure_moe.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cstdint>

namespace {

constexpr int kTopK = 8;
constexpr int kNumLocalExperts = 32;
constexpr int kHiddenSize = 7168;
constexpr int kIntermediateSize = 2048;

#define CUDA_CHECK(expr)                                                                     \
  do {                                                                                       \
    cudaError_t err__ = (expr);                                                              \
    TVM_FFI_ICHECK_EQ(err__, cudaSuccess) << "CUDA failure: " << cudaGetErrorString(err__); \
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

struct Workspace {
  DeviceBuffer topk_ids;
  DeviceBuffer topk_weights;
  DeviceBuffer expert_counts;
  DeviceBuffer expert_offsets;
  DeviceBuffer sorted_token_ids;
  DeviceBuffer sorted_weights;
  DeviceBuffer intermediate;

  static Workspace& instance() {
    static Workspace ws;
    return ws;
  }
};

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

  const size_t max_pairs = static_cast<size_t>(seq_len) * kTopK;
  ws.topk_ids.ensure(max_pairs * sizeof(int32_t));
  ws.topk_weights.ensure(max_pairs * sizeof(float));
  ws.expert_counts.ensure(kNumLocalExperts * sizeof(int32_t));
  ws.expert_offsets.ensure(kNumLocalExperts * sizeof(int32_t));
  ws.sorted_token_ids.ensure(max_pairs * sizeof(int32_t));
  ws.sorted_weights.ensure(max_pairs * sizeof(float));
  ws.intermediate.ensure(max_pairs * kIntermediateSize * sizeof(__nv_bfloat16));

  auto* stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(output.device().device_type, output.device().device_id));

  launch_moe_pipeline_cuda(
      static_cast<const float*>(routing_logits.data_ptr()),
      static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
      static_cast<const float*>(hidden_states_scale.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(gemm1_weights.data_ptr()),
      static_cast<const float*>(gemm1_weights_scale.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(gemm2_weights.data_ptr()),
      static_cast<const float*>(gemm2_weights_scale.data_ptr()),
      routed_scaling_factor,
      seq_len,
      local_expert_offset,
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int32_t*>(ws.topk_ids.ptr),
      static_cast<float*>(ws.topk_weights.ptr),
      static_cast<int32_t*>(ws.expert_counts.ptr),
      static_cast<int32_t*>(ws.expert_offsets.ptr),
      static_cast<int32_t*>(ws.sorted_token_ids.ptr),
      static_cast<float*>(ws.sorted_weights.ptr),
      static_cast<__nv_bfloat16*>(ws.intermediate.ptr),
      stream);
  CUDA_CHECK(cudaGetLastError());
}
