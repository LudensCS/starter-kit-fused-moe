from typing import Any, cast

import torch
import triton
import triton.language as tl

NUM_EXPERTS_GLOBAL = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
BLOCK_SIZE = 128
NUM_HIDDEN_BLOCKS = HIDDEN_SIZE // BLOCK_SIZE
NUM_INTERMEDIATE_BLOCKS = INTERMEDIATE_SIZE // BLOCK_SIZE
EXPERTS_PER_GROUP = NUM_EXPERTS_GLOBAL // N_GROUP
EPS = 1e-20


@triton.jit
def _routing_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    count_ptr,
    out_ptr,
    stride_lm,
    stride_ln,
    stride_tim,
    stride_tin,
    stride_twm,
    stride_twn,
    stride_om,
    stride_on,
    local_expert_offset,
    routed_scaling_factor,
):
    pid = tl.program_id(0)
    expert_offs = tl.arange(0, 256)
    group_offs = tl.arange(0, 8)
    expert_tiebreak = expert_offs.to(tl.float32) * 1e-6
    group_tiebreak = group_offs.to(tl.float32) * 1e-4

    logits = tl.load(logits_ptr + pid * stride_lm + expert_offs * stride_ln).to(tl.float32)
    bias = tl.load(bias_ptr + expert_offs).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-logits))
    score = sig + bias
    score_tie = tl.reshape(score + expert_tiebreak, (8, 32))

    max1 = tl.max(score_tie, axis=1)
    score2 = tl.where(score_tie == max1[:, None], -1.0e30, score_tie)
    max2 = tl.max(score2, axis=1)
    group_scores = max1 + max2

    selected_groups = group_offs < 0
    tmp_group_scores = group_scores + group_tiebreak
    for _ in range(4):
        best_group = tl.max(tmp_group_scores, axis=0)
        chosen_group = tmp_group_scores == best_group
        selected_groups = selected_groups | chosen_group
        tmp_group_scores = tl.where(chosen_group, -1.0e30, tmp_group_scores)

    candidate_scores = tl.reshape(
        tl.where(selected_groups[:, None], score_tie, -1.0e30),
        (256,),
    )

    total_sig = 0.0
    for k in range(8):
        best_score = tl.max(candidate_scores, axis=0)
        chosen_expert = candidate_scores == best_score
        expert = tl.sum(tl.where(chosen_expert, expert_offs.to(tl.int32), 0), axis=0)
        expert_sig = tl.sum(tl.where(chosen_expert, sig, 0.0), axis=0)
        tl.store(topk_ids_ptr + pid * stride_tim + k * stride_tin, expert)
        tl.store(topk_weights_ptr + pid * stride_twm + k * stride_twn, expert_sig)
        local = expert - local_expert_offset
        local_mask = (local >= 0) & (local < 32)
        local_safe = tl.where(local_mask, local, 0)
        tl.atomic_add(count_ptr + local_safe, 1, mask=local_mask)
        total_sig += expert_sig
        candidate_scores = tl.where(chosen_expert, -1.0e30, candidate_scores)

    inv_total = 1.0 / (total_sig + 1.0e-20)
    for k in range(8):
        weight = tl.load(topk_weights_ptr + pid * stride_twm + k * stride_twn).to(tl.float32)
        tl.store(
            topk_weights_ptr + pid * stride_twm + k * stride_twn,
            weight * inv_total * routed_scaling_factor,
        )

    for base in range(0, 7168, 256):
        cols = base + expert_offs
        col_mask = cols < 7168
        tl.store(
            out_ptr + pid * stride_om + cols * stride_on,
            0.0,
            mask=col_mask,
        )


@triton.jit
def _prefix_sum32_kernel(
    count_ptr,
    offset_ptr,
):
    running = tl.zeros((), dtype=tl.int32)
    for i in range(32):
        count = tl.load(count_ptr + i).to(tl.int32)
        tl.store(offset_ptr + i, running)
        running += count


@triton.jit
def _scatter_local_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    offsets_ptr,
    cursor_ptr,
    sorted_token_ptr,
    sorted_weight_ptr,
    total_slots,
    stride_tim,
    stride_tin,
    stride_twm,
    stride_twn,
    stride_st,
    stride_sw,
    local_expert_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_slots
    token_ids = (offs // 8).to(tl.int32)
    k_slot = offs % 8

    experts = tl.load(
        topk_ids_ptr + token_ids * stride_tim + k_slot * stride_tin,
        mask=mask,
        other=0,
    ).to(tl.int32)
    weights = tl.load(
        topk_weights_ptr + token_ids * stride_twm + k_slot * stride_twn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    local = experts - local_expert_offset
    local_mask = mask & (local >= 0) & (local < 32)
    local_safe = tl.where(local_mask, local, 0)
    pos = tl.atomic_add(cursor_ptr + local_safe, 1, mask=local_mask)
    base = tl.load(offsets_ptr + local_safe, mask=local_mask, other=0).to(tl.int32)
    dest = base + pos

    tl.store(sorted_token_ptr + dest * stride_st, token_ids, mask=local_mask)
    tl.store(sorted_weight_ptr + dest * stride_sw, weights, mask=local_mask)


@triton.jit
def _gemm1_swiglu_kernel(
    hidden_ptr,
    hidden_scale_ptr,
    token_ptr,
    offset_ptr,
    count_ptr,
    weight_ptr,
    weight_scale_ptr,
    out_ptr,
    stride_hm,
    stride_hk,
    stride_hsm,
    stride_hsn,
    stride_t,
    stride_s,
    stride_c,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    count = tl.load(count_ptr + pid_e * stride_c).to(tl.int32)
    if pid_m * BLOCK_M >= count:
        return

    start = tl.load(offset_ptr + pid_e * stride_s).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_ids = start + offs_m
    mask_m = offs_m < count
    token_ids = tl.load(token_ptr + slot_ids * stride_t, mask=mask_m, other=0).to(tl.int32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < 2048

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(56):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_scale = tl.load(
            hidden_scale_ptr + kb * stride_hsm + token_ids * stride_hsn,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        hidden = (hidden * hidden_scale[:, None]).to(tl.bfloat16)

        gate = tl.load(
            weight_ptr
            + pid_e * stride_we
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk,
            mask=mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            weight_ptr
            + pid_e * stride_we
            + (offs_n + 2048)[None, :] * stride_wn
            + offs_k[:, None] * stride_wk,
            mask=mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        gate_scale = tl.load(
            weight_scale_ptr + pid_e * stride_wse + pid_n * stride_wsn + kb * stride_wsk
        ).to(tl.float32)
        up_scale = tl.load(
            weight_scale_ptr
            + pid_e * stride_wse
            + (pid_n + 16) * stride_wsn
            + kb * stride_wsk
        ).to(tl.float32)

        gate = (gate * gate_scale).to(tl.bfloat16)
        up = (up * up_scale).to(tl.bfloat16)

        acc_gate += tl.dot(hidden, gate)
        acc_up += tl.dot(hidden, up)

    out = acc_gate * (acc_up / (1.0 + tl.exp(-acc_up)))
    tl.store(
        out_ptr + slot_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _gemm2_scatter_kernel(
    act_ptr,
    token_ptr,
    sorted_weight_ptr,
    offset_ptr,
    count_ptr,
    weight_ptr,
    weight_scale_ptr,
    out_ptr,
    stride_am,
    stride_ak,
    stride_t,
    stride_sw,
    stride_s,
    stride_c,
    stride_we,
    stride_wn,
    stride_wk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_e = tl.program_id(2)

    count = tl.load(count_ptr + pid_e * stride_c).to(tl.int32)
    if pid_m * BLOCK_M >= count:
        return

    start = tl.load(offset_ptr + pid_e * stride_s).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_ids = start + offs_m
    mask_m = offs_m < count

    token_ids = tl.load(token_ptr + slot_ids * stride_t, mask=mask_m, other=0).to(tl.int32)
    row_weights = tl.load(
        sorted_weight_ptr + slot_ids * stride_sw,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < 7168
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(16):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        act = tl.load(
            act_ptr + slot_ids[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        weight = tl.load(
            weight_ptr
            + pid_e * stride_we
            + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk,
            mask=mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        weight_scale = tl.load(
            weight_scale_ptr + pid_e * stride_wse + pid_n * stride_wsn + kb * stride_wsk
        ).to(tl.float32)
        weight = (weight * weight_scale).to(tl.bfloat16)
        acc += tl.dot(act, weight)

    out_ptrs = out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(
        out_ptrs,
        (acc * row_weights[:, None]).to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _as_cuda_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is unavailable, cannot move {name} from {tensor.device} to CUDA."
            )
        tensor = tensor.cuda(non_blocking=True)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _routing_and_scatter(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    output: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    seq_len = routing_logits.shape[0]
    total_slots = seq_len * TOP_K

    topk_ids = torch.empty((seq_len, TOP_K), dtype=torch.int32, device=routing_logits.device)
    topk_weights = torch.empty(
        (seq_len, TOP_K), dtype=torch.float32, device=routing_logits.device
    )
    counts = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=routing_logits.device)
    offsets = torch.empty((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=routing_logits.device)
    sorted_token = torch.empty((total_slots,), dtype=torch.int32, device=routing_logits.device)
    sorted_weight = torch.empty(
        (total_slots,), dtype=torch.float32, device=routing_logits.device
    )

    bias = (
        torch.zeros((NUM_EXPERTS_GLOBAL,), dtype=torch.float32, device=routing_logits.device)
        if routing_bias is None
        else routing_bias.to(torch.float32).contiguous().view(NUM_EXPERTS_GLOBAL)
    )

    cast(Any, _routing_topk_kernel)[(seq_len,)](
        routing_logits,
        bias,
        topk_ids,
        topk_weights,
        counts,
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
        num_warps=8,
        num_stages=1,
    )

    cast(Any, _prefix_sum32_kernel)[(1,)](
        counts,
        offsets,
        num_warps=1,
        num_stages=1,
    )

    cursor = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=routing_logits.device)
    cast(Any, _scatter_local_kernel)[(triton.cdiv(total_slots, 256),)](
        topk_ids,
        topk_weights,
        offsets,
        cursor,
        sorted_token,
        sorted_weight,
        total_slots,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        sorted_token.stride(0),
        sorted_weight.stride(0),
        local_expert_offset,
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return sorted_token, sorted_weight, counts, offsets


def _gemm1_swiglu_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_sorted: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    max_count: int,
) -> torch.Tensor:
    total_rows = token_sorted.numel()
    out = torch.empty(
        (total_rows, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if total_rows == 0 or max_count == 0:
        return out

    grid = (
        triton.cdiv(INTERMEDIATE_SIZE, BLOCK_SIZE),
        triton.cdiv(max_count, 32),
        NUM_LOCAL_EXPERTS,
    )
    cast(Any, _gemm1_swiglu_kernel)[grid](
        hidden_states,
        hidden_states_scale,
        token_sorted,
        expert_offsets,
        counts,
        gemm1_weights,
        gemm1_weights_scale,
        out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        token_sorted.stride(0),
        expert_offsets.stride(0),
        counts.stride(0),
        gemm1_weights.stride(0),
        gemm1_weights.stride(1),
        gemm1_weights.stride(2),
        gemm1_weights_scale.stride(0),
        gemm1_weights_scale.stride(1),
        gemm1_weights_scale.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=BLOCK_SIZE,
        BLOCK_K=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


def _gemm2_scatter_triton(
    act: torch.Tensor,
    token_sorted: torch.Tensor,
    sorted_weight: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    max_count: int,
):
    if act.numel() == 0 or max_count == 0:
        return

    grid = (
        triton.cdiv(HIDDEN_SIZE, BLOCK_SIZE),
        triton.cdiv(max_count, 32),
        NUM_LOCAL_EXPERTS,
    )
    cast(Any, _gemm2_scatter_kernel)[grid](
        act,
        token_sorted,
        sorted_weight,
        expert_offsets,
        counts,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        act.stride(0),
        act.stride(1),
        token_sorted.stride(0),
        sorted_weight.stride(0),
        expert_offsets.stride(0),
        counts.stride(0),
        gemm2_weights.stride(0),
        gemm2_weights.stride(1),
        gemm2_weights.stride(2),
        gemm2_weights_scale.stride(0),
        gemm2_weights_scale.stride(1),
        gemm2_weights_scale.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M=32,
        BLOCK_N=BLOCK_SIZE,
        BLOCK_K=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    output_device = hidden_states.device
    return_to_origin = output_device.type != "cuda"

    if (
        routing_logits.device.type != "cuda"
        or hidden_states.device.type != "cuda"
        or hidden_states_scale.device.type != "cuda"
        or gemm1_weights.device.type != "cuda"
        or gemm1_weights_scale.device.type != "cuda"
        or gemm2_weights.device.type != "cuda"
        or gemm2_weights_scale.device.type != "cuda"
        or (routing_bias is not None and routing_bias.device.type != "cuda")
    ):
        routing_logits = _as_cuda_contiguous(routing_logits, "routing_logits")
        hidden_states = _as_cuda_contiguous(hidden_states, "hidden_states")
        hidden_states_scale = _as_cuda_contiguous(hidden_states_scale, "hidden_states_scale")
        gemm1_weights = _as_cuda_contiguous(gemm1_weights, "gemm1_weights")
        gemm1_weights_scale = _as_cuda_contiguous(gemm1_weights_scale, "gemm1_weights_scale")
        gemm2_weights = _as_cuda_contiguous(gemm2_weights, "gemm2_weights")
        gemm2_weights_scale = _as_cuda_contiguous(gemm2_weights_scale, "gemm2_weights_scale")
        if routing_bias is not None:
            routing_bias = _as_cuda_contiguous(routing_bias, "routing_bias")
    else:
        if not routing_logits.is_contiguous():
            routing_logits = routing_logits.contiguous()
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not hidden_states_scale.is_contiguous():
            hidden_states_scale = hidden_states_scale.contiguous()
        if not gemm1_weights.is_contiguous():
            gemm1_weights = gemm1_weights.contiguous()
        if not gemm1_weights_scale.is_contiguous():
            gemm1_weights_scale = gemm1_weights_scale.contiguous()
        if not gemm2_weights.is_contiguous():
            gemm2_weights = gemm2_weights.contiguous()
        if not gemm2_weights_scale.is_contiguous():
            gemm2_weights_scale = gemm2_weights_scale.contiguous()
        if routing_bias is not None and not routing_bias.is_contiguous():
            routing_bias = routing_bias.contiguous()

    local_expert_offset = (
        int(local_expert_offset.item())
        if isinstance(local_expert_offset, torch.Tensor)
        else int(local_expert_offset)
    )
    routed_scaling_factor = (
        float(routed_scaling_factor.item())
        if isinstance(routed_scaling_factor, torch.Tensor)
        else float(routed_scaling_factor)
    )

    seq_len = routing_logits.shape[0]
    if seq_len == 0:
        out = torch.empty((0, HIDDEN_SIZE), dtype=torch.bfloat16, device=routing_logits.device)
        return out if not return_to_origin else out.to(output_device)

    output = torch.empty((seq_len, HIDDEN_SIZE), dtype=torch.bfloat16, device=routing_logits.device)
    token_sorted, weight_sorted, counts, expert_offsets = _routing_and_scatter(
        routing_logits.to(torch.float32),
        routing_bias,
        output,
        local_expert_offset,
        routed_scaling_factor,
    )
    max_count = int(counts.max().item())
    if max_count == 0:
        return output if not return_to_origin else output.to(output_device)

    inter = _gemm1_swiglu_triton(
        hidden_states,
        hidden_states_scale,
        token_sorted,
        counts,
        expert_offsets,
        gemm1_weights,
        gemm1_weights_scale,
        max_count,
    )
    _gemm2_scatter_triton(
        inter,
        token_sorted,
        weight_sorted,
        counts,
        expert_offsets,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        max_count,
    )

    return output if not return_to_origin else output.to(output_device)
