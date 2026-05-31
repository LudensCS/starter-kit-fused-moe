"""CUTLASS-backed large-token path used by ``current_kernel``.

This module keeps the high-level MoE pipeline in our own code: routing is still
provided by ``current_kernel``, while this file owns the large-batch layout
conversion around the CUTLASS GEMM helpers.
"""

from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl


NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128
NUM_INTERMEDIATE_BLOCKS = INTERMEDIATE_SIZE // BLOCK_SIZE

_CUTLASS_READY = False
_CUTLASS_IMPORT_ATTEMPTED = False
_cutlass_gemm1_buffers = None
_cutlass_swiglu_kernel = None
_cutlass_run_gemm1 = None
_cutlass_run_gemm2_fp8 = None


def _ensure_cutlass_ready() -> bool:
    global _CUTLASS_READY
    global _CUTLASS_IMPORT_ATTEMPTED
    global _cutlass_gemm1_buffers
    global _cutlass_swiglu_kernel
    global _cutlass_run_gemm1
    global _cutlass_run_gemm2_fp8

    if _CUTLASS_READY:
        return True
    if _CUTLASS_IMPORT_ATTEMPTED:
        return False
    _CUTLASS_IMPORT_ATTEMPTED = True
    try:
        try:
            from .cutlass_gemm1_helper import _get_buffers as gemm1_buffers
            from .cutlass_gemm1_helper import fused_swiglu_fp8_padded_kernel
            from .cutlass_gemm1_helper import run_gemm1
            from .cutlass_gemm2_fp8_helper import run_gemm2_fp8
        except ImportError:
            from cutlass_gemm1_helper import _get_buffers as gemm1_buffers
            from cutlass_gemm1_helper import fused_swiglu_fp8_padded_kernel
            from cutlass_gemm1_helper import run_gemm1
            from cutlass_gemm2_fp8_helper import run_gemm2_fp8
    except Exception:
        return False

    _cutlass_gemm1_buffers = gemm1_buffers
    _cutlass_swiglu_kernel = fused_swiglu_fp8_padded_kernel
    _cutlass_run_gemm1 = run_gemm1
    _cutlass_run_gemm2_fp8 = run_gemm2_fp8
    _CUTLASS_READY = True
    return True


@triton.jit
def _prefix_sum32_dual_kernel(
    count_ptr,
    actual_offset_ptr,
    padded_offset_ptr,
    ALIGNMENT: tl.constexpr,
):
    actual = tl.zeros((), dtype=tl.int32)
    padded = tl.zeros((), dtype=tl.int32)
    for expert in range(32):
        count = tl.load(count_ptr + expert).to(tl.int32)
        tl.store(actual_offset_ptr + expert, actual)
        tl.store(padded_offset_ptr + expert, padded)
        actual += count
        padded += tl.cdiv(count, ALIGNMENT) * ALIGNMENT
    tl.store(actual_offset_ptr + 32, actual)
    tl.store(padded_offset_ptr + 32, padded)


@triton.jit
def _scatter_cutlass_rows_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    actual_offsets_ptr,
    padded_offsets_ptr,
    expert_cursor_ptr,
    token_cursor_ptr,
    sorted_token_ptr,
    gather_idx_ptr,
    gather_weight_ptr,
    total_slots,
    stride_tim,
    stride_tin,
    stride_twm,
    stride_twn,
    local_expert_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    active = offs < total_slots
    token_ids = (offs // 8).to(tl.int32)
    topk_slot = offs % 8

    experts = tl.load(
        topk_ids_ptr + token_ids * stride_tim + topk_slot * stride_tin,
        mask=active,
        other=0,
    ).to(tl.int32)
    weights = tl.load(
        topk_weights_ptr + token_ids * stride_twm + topk_slot * stride_twn,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    local_expert = experts - local_expert_offset
    local_hit = active & (local_expert >= 0) & (local_expert < 32)
    local_safe = tl.where(local_hit, local_expert, 0)

    local_pos = tl.atomic_add(expert_cursor_ptr + local_safe, 1, mask=local_hit)
    actual_base = tl.load(actual_offsets_ptr + local_safe, mask=local_hit, other=0).to(tl.int32)
    padded_base = tl.load(padded_offsets_ptr + local_safe, mask=local_hit, other=0).to(tl.int32)
    actual_row = actual_base + local_pos
    padded_row = padded_base + local_pos

    gather_slot = tl.atomic_add(token_cursor_ptr + token_ids, 1, mask=local_hit)
    gather_offset = token_ids * 8 + gather_slot

    tl.store(sorted_token_ptr + actual_row, token_ids, mask=local_hit)
    tl.store(gather_idx_ptr + gather_offset, padded_row, mask=local_hit)
    tl.store(gather_weight_ptr + gather_offset, weights, mask=local_hit)


@triton.jit
def _reduce_cutlass_rows_kernel(
    gemm2_out_ptr,
    gather_idx_ptr,
    gather_weight_ptr,
    gather_count_ptr,
    output_ptr,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    token = tl.program_id(1)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    valid_col = cols < 7168
    route_count = tl.load(gather_count_ptr + token).to(tl.int32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for slot in range(8):
        active = slot < route_count
        row = tl.load(gather_idx_ptr + token * 8 + slot, mask=active, other=0).to(tl.int32)
        weight = tl.load(
            gather_weight_ptr + token * 8 + slot,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        vals = tl.load(
            gemm2_out_ptr + row * stride_gm + cols * stride_gn,
            mask=valid_col & active,
            other=0.0,
        ).to(tl.float32)
        acc += vals * weight

    tl.store(
        output_ptr + token * stride_om + cols * stride_on,
        acc.to(tl.bfloat16),
        mask=valid_col,
    )


def run_cutlass_large_path(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
    *,
    cached_empty: Callable[[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor],
    cached_zero: Callable[[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor],
    launch_triton: Callable[..., None],
    routing_topk_kernel: Any,
    sync_free_seq_threshold: int,
    alignment: int,
) -> bool:
    if not _ensure_cutlass_ready():
        return False
    assert _cutlass_run_gemm1 is not None
    assert _cutlass_run_gemm2_fp8 is not None
    assert _cutlass_gemm1_buffers is not None
    assert _cutlass_swiglu_kernel is not None

    seq_len = routing_logits.shape[0]
    if seq_len <= 900:
        return False

    total_slots = seq_len * TOP_K
    device = routing_logits.device
    cutlass_alignment = 64 if seq_len <= sync_free_seq_threshold else alignment
    valid_m = seq_len + NUM_LOCAL_EXPERTS * (cutlass_alignment - 1)

    topk_ids = cached_empty("cutlass_topk_ids", (seq_len, TOP_K), torch.int32, device)
    topk_weights = cached_empty(
        "cutlass_topk_weights", (seq_len, TOP_K), torch.float32, device
    )
    counts = cached_zero("cutlass_counts", (NUM_LOCAL_EXPERTS,), torch.int32, device)
    token_counters = cached_empty("cutlass_token_counters", (seq_len,), torch.int32, device)
    bias = (
        cached_zero("zero_bias", (NUM_EXPERTS_GLOBAL,), torch.float32, device)
        if routing_bias is None
        else routing_bias.contiguous().view(NUM_EXPERTS_GLOBAL)
    )

    launch_triton(
        routing_topk_kernel,
        (seq_len,),
        routing_logits,
        bias,
        topk_ids,
        topk_weights,
        counts,
        token_counters,
        output,
        routing_logits.stride(0),
        routing_logits.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        output.stride(0),
        output.stride(1),
        local_expert_offset,
        routed_scaling_factor,
        False,
        _launch_key=("cutlass_route", seq_len, local_expert_offset, routed_scaling_factor),
        num_warps=8,
        num_stages=1,
    )

    actual_offsets = cached_empty(
        "cutlass_actual_offsets", (NUM_LOCAL_EXPERTS + 1,), torch.int32, device
    )
    padded_offsets = cached_empty(
        "cutlass_padded_offsets", (NUM_LOCAL_EXPERTS + 1,), torch.int32, device
    )

    sorted_token = cached_empty("cutlass_sorted_token", (valid_m,), torch.int32, device)
    gather_idx = cached_empty("cutlass_gather_idx", (seq_len, TOP_K), torch.int32, device)
    gather_weight = cached_empty(
        "cutlass_gather_weight", (seq_len, TOP_K), torch.float32, device
    )
    expert_cursor = cached_zero(
        "cutlass_expert_cursor", (NUM_LOCAL_EXPERTS,), torch.int32, device
    )
    gather_count = cached_zero("cutlass_token_counters", (seq_len,), torch.int32, device)

    launch_triton(
        _prefix_sum32_dual_kernel,
        (1,),
        counts,
        actual_offsets,
        padded_offsets,
        ALIGNMENT=cutlass_alignment,
        _launch_key=("cutlass_prefix", cutlass_alignment),
        num_warps=1,
        num_stages=1,
    )
    launch_triton(
        _scatter_cutlass_rows_kernel,
        (triton.cdiv(total_slots, 256),),
        topk_ids,
        topk_weights,
        actual_offsets,
        padded_offsets,
        expert_cursor,
        gather_count,
        sorted_token,
        gather_idx,
        gather_weight,
        total_slots,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        local_expert_offset,
        BLOCK=256,
        _launch_key=("cutlass_scatter", seq_len, local_expert_offset),
        num_warps=4,
        num_stages=1,
    )

    if cutlass_alignment == 64:
        cutlass_cfg = {"mma_m": 64, "mma_n": 128, "cl_m": 1, "cl_n": 1}
    else:
        cutlass_cfg = {}

    c_raw = _cutlass_run_gemm1(
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        actual_offsets,
        sorted_token,
        padded_offsets,
        valid_m,
        NUM_LOCAL_EXPERTS,
        **cutlass_cfg,
    )
    if c_raw is None:
        return True

    swiglu_fp8 = cached_empty(
        "cutlass_swiglu_fp8",
        (1, valid_m, INTERMEDIATE_SIZE),
        torch.float8_e4m3fn,
        device,
    )
    swiglu_scale = cached_empty(
        "cutlass_swiglu_scale",
        (1, NUM_INTERMEDIATE_BLOCKS, valid_m),
        torch.float32,
        device,
    )
    launch_triton(
        _cutlass_swiglu_kernel,
        (triton.cdiv(valid_m, 4),),
        c_raw,
        swiglu_fp8,
        swiglu_scale,
        padded_offsets,
        valid_m,
        I_dim=INTERMEDIATE_SIZE,
        N_dim=2 * INTERMEDIATE_SIZE,
        NUM_I_BLOCKS=NUM_INTERMEDIATE_BLOCKS,
        BLOCK_K=BLOCK_SIZE,
        BLOCK_M=4,
        E_LOCAL=NUM_LOCAL_EXPERTS,
        _launch_key=("cutlass_swiglu", seq_len, valid_m),
    )

    gemm2_out = cached_empty(
        "cutlass_gemm2_out", (1, valid_m, HIDDEN_SIZE), torch.bfloat16, device
    )
    gidx_mapping = _cutlass_gemm1_buffers(
        valid_m,
        HIDDEN_SIZE,
        2 * INTERMEDIATE_SIZE,
        NUM_LOCAL_EXPERTS,
        device,
    ).gidx_mapping
    _cutlass_run_gemm2_fp8(
        swiglu_fp8,
        swiglu_scale,
        gemm2_weights,
        gemm2_weights_scale,
        gidx_mapping,
        gemm2_out,
        valid_m,
        NUM_LOCAL_EXPERTS,
        **cutlass_cfg,
    )

    launch_triton(
        _reduce_cutlass_rows_kernel,
        (triton.cdiv(HIDDEN_SIZE, 2048), seq_len),
        gemm2_out,
        gather_idx,
        gather_weight,
        gather_count,
        output,
        gemm2_out.stride(1),
        gemm2_out.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_H=2048,
        _launch_key=("cutlass_reduce", seq_len),
        num_warps=8,
        num_stages=1,
    )
    return True
