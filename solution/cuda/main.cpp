#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>

#include "kernel.h"

namespace {

constexpr int kNumExpertsGlobal = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kHiddenSize = 7168;
constexpr int kIntermediateSize = 2048;
constexpr int kGemm1OutSize = 4096;
constexpr int kBlock = 128;
constexpr int kNumHiddenBlocks = kHiddenSize / kBlock;
constexpr int kNumIntermediateBlocks = kIntermediateSize / kBlock;
constexpr int kNumGemm1OutBlocks = kGemm1OutSize / kBlock;

void check_cuda_tensor(const tvm::ffi::TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be on CUDA";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

void check_dtype(const tvm::ffi::TensorView& tensor, uint8_t code, uint8_t bits, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.dtype().code, code) << name << " has unexpected dtype code";
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, bits) << name << " has unexpected dtype bits";
  TVM_FFI_ICHECK_EQ(tensor.dtype().lanes, 1) << name << " must have one lane";
}

void check_fp8_tensor(const tvm::ffi::TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.dtype().bits, 8) << name << " must have 8-bit elements";
  TVM_FFI_ICHECK_EQ(tensor.dtype().lanes, 1) << name << " must have one lane";
}

}  // namespace

void run(
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
  const int64_t seq_len = routing_logits.size(0);

  check_cuda_tensor(routing_logits, "routing_logits");
  check_cuda_tensor(routing_bias, "routing_bias");
  check_cuda_tensor(hidden_states, "hidden_states");
  check_cuda_tensor(hidden_states_scale, "hidden_states_scale");
  check_cuda_tensor(gemm1_weights, "gemm1_weights");
  check_cuda_tensor(gemm1_weights_scale, "gemm1_weights_scale");
  check_cuda_tensor(gemm2_weights, "gemm2_weights");
  check_cuda_tensor(gemm2_weights_scale, "gemm2_weights_scale");
  check_cuda_tensor(output, "output");

  check_dtype(routing_logits, kDLFloat, 32, "routing_logits");
  check_dtype(routing_bias, kDLBfloat, 16, "routing_bias");
  check_fp8_tensor(hidden_states, "hidden_states");
  check_dtype(hidden_states_scale, kDLFloat, 32, "hidden_states_scale");
  check_fp8_tensor(gemm1_weights, "gemm1_weights");
  check_dtype(gemm1_weights_scale, kDLFloat, 32, "gemm1_weights_scale");
  check_fp8_tensor(gemm2_weights, "gemm2_weights");
  check_dtype(gemm2_weights_scale, kDLFloat, 32, "gemm2_weights_scale");
  check_dtype(output, kDLBfloat, 16, "output");

  TVM_FFI_ICHECK_EQ(routing_logits.ndim(), 2) << "routing_logits must be 2D";
  TVM_FFI_ICHECK_EQ(routing_bias.ndim(), 1) << "routing_bias must be 1D";
  TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D";
  TVM_FFI_ICHECK_EQ(hidden_states_scale.ndim(), 2) << "hidden_states_scale must be 2D";
  TVM_FFI_ICHECK_EQ(gemm1_weights.ndim(), 3) << "gemm1_weights must be 3D";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.ndim(), 3) << "gemm1_weights_scale must be 3D";
  TVM_FFI_ICHECK_EQ(gemm2_weights.ndim(), 3) << "gemm2_weights must be 3D";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.ndim(), 3) << "gemm2_weights_scale must be 3D";
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be 2D";

  TVM_FFI_ICHECK_EQ(routing_logits.size(1), kNumExpertsGlobal) << "routing_logits.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(routing_bias.size(0), kNumExpertsGlobal) << "routing_bias.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(hidden_states.size(0), seq_len) << "hidden_states.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(hidden_states.size(1), kHiddenSize) << "hidden_states.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), kNumHiddenBlocks) << "hidden_states_scale.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1), seq_len) << "hidden_states_scale.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(0), kNumLocalExperts) << "gemm1_weights.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), kGemm1OutSize) << "gemm1_weights.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(2), kHiddenSize) << "gemm1_weights.shape[2] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), kNumLocalExperts) << "gemm1_weights_scale.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1), kNumGemm1OutBlocks) << "gemm1_weights_scale.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2), kNumHiddenBlocks) << "gemm1_weights_scale.shape[2] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(0), kNumLocalExperts) << "gemm2_weights.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(1), kHiddenSize) << "gemm2_weights.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(2), kIntermediateSize) << "gemm2_weights.shape[2] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(0), kNumLocalExperts) << "gemm2_weights_scale.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(1), kNumHiddenBlocks) << "gemm2_weights_scale.shape[1] mismatch";
  TVM_FFI_ICHECK_EQ(gemm2_weights_scale.size(2), kNumIntermediateBlocks) << "gemm2_weights_scale.shape[2] mismatch";
  TVM_FFI_ICHECK_EQ(output.size(0), seq_len) << "output.shape[0] mismatch";
  TVM_FFI_ICHECK_EQ(output.size(1), kHiddenSize) << "output.shape[1] mismatch";

  launch_moe_cuda(
      routing_logits,
      routing_bias,
      hidden_states,
      hidden_states_scale,
      gemm1_weights,
      gemm1_weights_scale,
      gemm2_weights,
      gemm2_weights_scale,
      local_expert_offset,
      routed_scaling_factor,
      output);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
