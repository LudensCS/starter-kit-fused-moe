#ifndef MLSYS26_FUSED_MOE_PURE_MOE_H_
#define MLSYS26_FUSED_MOE_PURE_MOE_H_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

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
    cudaStream_t stream);

#endif
