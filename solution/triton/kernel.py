import torch
import triton
import triton.language as tl
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

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
NUM_GEMM1_OUT_BLOCKS = (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE
EXPERTS_PER_GROUP = NUM_EXPERTS_GLOBAL // N_GROUP
EPS = 1e-20
_GROUP_OFFSETS = None
HYBRID_FLASHINFER_MAX_SEQ = 4096
HYBRID_FLASHINFER_MIN_SEQ = 10000

_ACTIVE_WEIGHT_CACHE_KEY = None
_ACTIVE_WEIGHT_CACHE_VALUE = None


@triton.jit
def _swiglu_kernel(
    g_ptr,
    c_ptr,
    rows,
    stride_gm,
    stride_gn,
    stride_cm,
    stride_cn,
    I: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_m < rows) & (offs_n < I)

    x1 = tl.load(g_ptr + pid_m * stride_gm + offs_n * stride_gn, mask=mask, other=0.0)
    x2 = tl.load(
        g_ptr + pid_m * stride_gm + (offs_n + I) * stride_gn,
        mask=mask,
        other=0.0,
    )
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    y = x1 * (x2 / (1.0 + tl.exp(-x2)))
    tl.store(
        c_ptr + pid_m * stride_cm + offs_n * stride_cn,
        y.to(tl.bfloat16),
        mask=mask,
    )


def _as_cuda_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is unavailable, cannot move {name} from {tensor.device} to CUDA."
            )
        tensor = tensor.cuda(non_blocking=True)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _normalize_int_scalar(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _normalize_float_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    return float(x)


def _tensor_cache_key(t: torch.Tensor):
    return (
        int(t.data_ptr()),
        tuple(t.shape),
        tuple(t.stride()),
        t.dtype,
        t.device.type,
        t.device.index,
        int(t._version),
    )


def _swiglu_triton(g1: torch.Tensor) -> torch.Tensor:
    rows = g1.shape[0]
    out = torch.empty(
        (rows, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=g1.device,
    )
    if rows == 0:
        return out
    grid = (rows, triton.cdiv(INTERMEDIATE_SIZE, BLOCK_SIZE))
    _swiglu_kernel[grid](
        g1,
        out,
        rows,
        g1.stride(0),
        g1.stride(1),
        out.stride(0),
        out.stride(1),
        I=INTERMEDIATE_SIZE,
        BLOCK_N=BLOCK_SIZE,
    )
    return out


def _dequant_selected_hidden_states(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_idx: torch.Tensor,
) -> torch.Tensor:
    m = token_idx.numel()
    a = hidden_states.index_select(0, token_idx).to(torch.float32).view(
        m, NUM_HIDDEN_BLOCKS, BLOCK_SIZE
    )
    s = (
        hidden_states_scale.to(torch.float32)
        .transpose(0, 1)
        .contiguous()
        .index_select(0, token_idx)
        .unsqueeze(-1)
    )
    return (a * s).view(m, HIDDEN_SIZE).to(torch.bfloat16)


def _dequant_selected_w13_t(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    active_local_ids: torch.Tensor,
) -> torch.Tensor:
    n = active_local_ids.numel()
    w = gemm1_weights.index_select(0, active_local_ids).to(torch.float32).view(
        n,
        NUM_GEMM1_OUT_BLOCKS,
        BLOCK_SIZE,
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm1_weights_scale.index_select(0, active_local_ids).to(torch.float32).view(
        n,
        NUM_GEMM1_OUT_BLOCKS,
        NUM_HIDDEN_BLOCKS,
    )
    deq = (w * s[:, :, None, :, None]).view(n, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    return deq.transpose(1, 2).contiguous().to(torch.bfloat16)


def _dequant_selected_w2_t(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    active_local_ids: torch.Tensor,
) -> torch.Tensor:
    n = active_local_ids.numel()
    w = gemm2_weights.index_select(0, active_local_ids).to(torch.float32).view(
        n,
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
        NUM_INTERMEDIATE_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm2_weights_scale.index_select(0, active_local_ids).to(torch.float32).view(
        n,
        NUM_HIDDEN_BLOCKS,
        NUM_INTERMEDIATE_BLOCKS,
    )
    deq = (w * s[:, :, None, :, None]).view(n, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return deq.transpose(1, 2).contiguous().to(torch.bfloat16)


def _base_weight_cache_key(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
):
    return (
        _tensor_cache_key(gemm1_weights),
        _tensor_cache_key(gemm1_weights_scale),
        _tensor_cache_key(gemm2_weights),
        _tensor_cache_key(gemm2_weights_scale),
    )


def _get_active_dequantized_weight_pack(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    active_local_ids: torch.Tensor,
    cache_key,
):
    global _ACTIVE_WEIGHT_CACHE_KEY, _ACTIVE_WEIGHT_CACHE_VALUE

    if _ACTIVE_WEIGHT_CACHE_KEY == cache_key and _ACTIVE_WEIGHT_CACHE_VALUE is not None:
        return _ACTIVE_WEIGHT_CACHE_VALUE

    w13_t = _dequant_selected_w13_t(gemm1_weights, gemm1_weights_scale, active_local_ids)
    w2_t = _dequant_selected_w2_t(gemm2_weights, gemm2_weights_scale, active_local_ids)
    _ACTIVE_WEIGHT_CACHE_KEY = cache_key
    _ACTIVE_WEIGHT_CACHE_VALUE = (w13_t, w2_t)
    return w13_t, w2_t


def _build_local_plan(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    local_start: int,
    routed_scaling_factor: float,
):
    seq_len = routing_logits.shape[0]
    logits = routing_logits.to(torch.float32)
    bias = (
        torch.zeros((1, NUM_EXPERTS_GLOBAL), dtype=torch.float32, device=logits.device)
        if routing_bias is None
        else routing_bias.to(torch.float32).reshape(1, NUM_EXPERTS_GLOBAL)
    )

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    grouped_scores = s_with_bias.view(seq_len, N_GROUP, EXPERTS_PER_GROUP)
    top2_vals = torch.topk(grouped_scores, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2_vals.sum(dim=2)
    top_group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    ).indices
    global _GROUP_OFFSETS
    if (
        _GROUP_OFFSETS is None
        or _GROUP_OFFSETS.device != logits.device
        or _GROUP_OFFSETS.dtype != torch.int64
    ):
        _GROUP_OFFSETS = torch.arange(
            EXPERTS_PER_GROUP, device=logits.device, dtype=torch.int64
        ).view(1, 1, EXPERTS_PER_GROUP)

    candidate_experts = (
        top_group_idx.to(torch.int64).unsqueeze(-1) * EXPERTS_PER_GROUP + _GROUP_OFFSETS
    ).reshape(seq_len, TOPK_GROUP * EXPERTS_PER_GROUP)
    candidate_scores = torch.gather(s_with_bias, 1, candidate_experts)
    topk_pos = torch.topk(candidate_scores, k=TOP_K, dim=1, largest=True, sorted=False).indices
    topk_idx = torch.gather(candidate_experts, 1, topk_pos)

    topk_s = torch.gather(s, 1, topk_idx)
    topk_weights = (topk_s / (topk_s.sum(dim=1, keepdim=True) + EPS)) * routed_scaling_factor

    local_idx = topk_idx - local_start
    local_mask = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)

    nz = torch.nonzero(local_mask, as_tuple=False)
    token_flat = nz[:, 0].to(torch.int64)
    k_slot = nz[:, 1].to(torch.int64)
    expert_flat = local_idx[token_flat, k_slot].to(torch.int64)
    w_flat = topk_weights[token_flat, k_slot].to(torch.float32)

    order = torch.argsort(expert_flat, stable=True)
    token_sorted = token_flat[order]
    expert_sorted = expert_flat[order]
    w_sorted = w_flat[order]

    return (
        token_sorted.contiguous(),
        w_sorted.contiguous(),
        expert_sorted.contiguous(),
    )


def _run_flashinfer_path(
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
    seq_len = routing_logits.shape[0]
    tune_max_num_tokens = 16384 if seq_len >= 8192 else 8192

    routing_logits_f32 = (
        routing_logits if routing_logits.dtype == torch.float32 else routing_logits.float()
    )
    hidden_states_scale_f32 = (
        hidden_states_scale
        if hidden_states_scale.dtype == torch.float32
        else hidden_states_scale.float()
    )
    gemm1_weights_scale_f32 = (
        gemm1_weights_scale
        if gemm1_weights_scale.dtype == torch.float32
        else gemm1_weights_scale.float()
    )
    gemm2_weights_scale_f32 = (
        gemm2_weights_scale
        if gemm2_weights_scale.dtype == torch.float32
        else gemm2_weights_scale.float()
    )
    return trtllm_fp8_block_scale_moe(
        routing_logits_f32,
        routing_bias,
        hidden_states,
        hidden_states_scale_f32,
        gemm1_weights,
        gemm1_weights_scale_f32,
        gemm2_weights,
        gemm2_weights_scale_f32,
        NUM_EXPERTS_GLOBAL,
        TOP_K,
        N_GROUP,
        TOPK_GROUP,
        INTERMEDIATE_SIZE,
        local_expert_offset,
        NUM_LOCAL_EXPERTS,
        routed_scaling_factor,
        routing_method_type=2,
        use_shuffled_weight=False,
        enable_pdl=True,
        tune_max_num_tokens=tune_max_num_tokens,
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

    if seq_len <= HYBRID_FLASHINFER_MAX_SEQ or seq_len >= HYBRID_FLASHINFER_MIN_SEQ:
        out = _run_flashinfer_path(
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
        )
        return out if not return_to_origin else out.to(output_device)

    output = torch.zeros(
        (seq_len, HIDDEN_SIZE), dtype=torch.float32, device=routing_logits.device
    )

    plan = _build_local_plan(
        routing_logits,
        routing_bias,
        local_expert_offset,
        routed_scaling_factor,
    )
    token_sorted, w_sorted, expert_sorted = plan
    if token_sorted.numel() == 0:
        return output.to(torch.bfloat16).to(output_device)
    a_cat = _dequant_selected_hidden_states(hidden_states, hidden_states_scale, token_sorted)

    base_weight_key = _base_weight_cache_key(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    active_local_ids, counts = torch.unique_consecutive(expert_sorted, return_counts=True)
    offs = counts.cumsum(0).to(dtype=torch.int32, device=routing_logits.device).contiguous()
    active_weight_key = (
        base_weight_key,
        local_expert_offset,
        tuple(int(x) for x in active_local_ids.tolist()),
    )
    w13_t, w2_t = _get_active_dequantized_weight_pack(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        active_local_ids,
        active_weight_key,
    )

    g1 = torch.ops.aten._grouped_mm(a_cat, w13_t, offs)
    c = _swiglu_triton(g1)
    o = torch.ops.aten._grouped_mm(c, w2_t, offs)
    output.index_add_(0, token_sorted, o.to(torch.float32) * w_sorted.unsqueeze(1))

    out = output.to(torch.bfloat16)
    return out if not return_to_origin else out.to(output_device)
