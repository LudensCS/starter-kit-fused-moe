#ifndef MLSYS26_FUSED_MOE_KERNEL_H_
#define MLSYS26_FUSED_MOE_KERNEL_H_

#include <cstdint>

#include <tvm/ffi/container/tensor.h>

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
    tvm::ffi::TensorView output);

#endif
