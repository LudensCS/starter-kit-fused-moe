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
NUM_GEMM1_OUT_BLOCKS = (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE
EXPERTS_PER_GROUP = NUM_EXPERTS_GLOBAL // N_GROUP
EPS = 1e-20

_WEIGHT_CACHE_KEY = None
_WEIGHT_CACHE_VALUE = None


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


def _ensure_tensor(x, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return x


def _to_cuda_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.device.type == "cuda":
        return tensor
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable, cannot move {name} from {tensor.device} to CUDA."
        )
    return tensor.cuda()


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


def _dequant_hidden_states(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
) -> torch.Tensor:
    t = hidden_states.shape[0]
    a = hidden_states.to(torch.float32).view(t, NUM_HIDDEN_BLOCKS, BLOCK_SIZE)
    s = hidden_states_scale.to(torch.float32).transpose(0, 1).contiguous().unsqueeze(-1)
    return (a * s).view(t, HIDDEN_SIZE).to(torch.bfloat16)


def _dequant_all_w13_t(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
) -> torch.Tensor:
    w = gemm1_weights.to(torch.float32).view(
        NUM_LOCAL_EXPERTS,
        NUM_GEMM1_OUT_BLOCKS,
        BLOCK_SIZE,
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm1_weights_scale.to(torch.float32).view(
        NUM_LOCAL_EXPERTS,
        NUM_GEMM1_OUT_BLOCKS,
        NUM_HIDDEN_BLOCKS,
    )
    deq = (w * s[:, :, None, :, None]).view(NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    return deq.transpose(1, 2).contiguous().to(torch.bfloat16)


def _dequant_all_w2_t(
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> torch.Tensor:
    w = gemm2_weights.to(torch.float32).view(
        NUM_LOCAL_EXPERTS,
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
        NUM_INTERMEDIATE_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm2_weights_scale.to(torch.float32).view(
        NUM_LOCAL_EXPERTS,
        NUM_HIDDEN_BLOCKS,
        NUM_INTERMEDIATE_BLOCKS,
    )
    deq = (w * s[:, :, None, :, None]).view(NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return deq.transpose(1, 2).contiguous().to(torch.bfloat16)


def _get_dequantized_weight_pack(
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
):
    global _WEIGHT_CACHE_KEY, _WEIGHT_CACHE_VALUE

    key = (
        _tensor_cache_key(gemm1_weights),
        _tensor_cache_key(gemm1_weights_scale),
        _tensor_cache_key(gemm2_weights),
        _tensor_cache_key(gemm2_weights_scale),
    )
    if _WEIGHT_CACHE_KEY == key and _WEIGHT_CACHE_VALUE is not None:
        return _WEIGHT_CACHE_VALUE

    w13_t = _dequant_all_w13_t(gemm1_weights, gemm1_weights_scale)
    w2_t = _dequant_all_w2_t(gemm2_weights, gemm2_weights_scale)
    _WEIGHT_CACHE_KEY = key
    _WEIGHT_CACHE_VALUE = (w13_t, w2_t)
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

    group_mask = torch.zeros((seq_len, N_GROUP), dtype=torch.bool, device=logits.device)
    group_mask.scatter_(1, top_group_idx, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(seq_len, N_GROUP, EXPERTS_PER_GROUP)
        .reshape(seq_len, NUM_EXPERTS_GLOBAL)
    )

    min_float = torch.finfo(torch.float32).min
    pruned_scores = s_with_bias.masked_fill(~expert_mask, min_float)
    topk_idx = torch.topk(pruned_scores, k=TOP_K, dim=1, largest=True, sorted=False).indices

    topk_s = torch.gather(s, 1, topk_idx)
    topk_weights = (topk_s / (topk_s.sum(dim=1, keepdim=True) + EPS)) * routed_scaling_factor

    local_idx = topk_idx - local_start
    local_mask = (local_idx >= 0) & (local_idx < NUM_LOCAL_EXPERTS)
    if not bool(local_mask.any()):
        return None

    token_idx = (
        torch.arange(seq_len, device=logits.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(seq_len, TOP_K)
    )

    token_flat = token_idx[local_mask]
    expert_flat = local_idx[local_mask].to(torch.int64)
    w_flat = topk_weights[local_mask].to(torch.float32)

    order = torch.argsort(expert_flat, stable=True)
    token_sorted = token_flat[order]
    expert_sorted = expert_flat[order]
    w_sorted = w_flat[order]

    active_local_ids, counts = torch.unique_consecutive(expert_sorted, return_counts=True)
    offs = counts.cumsum(0).to(dtype=torch.int32, device=logits.device).contiguous()

    return token_sorted.contiguous(), w_sorted.contiguous(), active_local_ids.contiguous(), offs


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
    routing_logits = _ensure_tensor(routing_logits, "routing_logits")
    hidden_states = _ensure_tensor(hidden_states, "hidden_states")
    hidden_states_scale = _ensure_tensor(hidden_states_scale, "hidden_states_scale")
    gemm1_weights = _ensure_tensor(gemm1_weights, "gemm1_weights")
    gemm1_weights_scale = _ensure_tensor(gemm1_weights_scale, "gemm1_weights_scale")
    gemm2_weights = _ensure_tensor(gemm2_weights, "gemm2_weights")
    gemm2_weights_scale = _ensure_tensor(gemm2_weights_scale, "gemm2_weights_scale")
    if routing_bias is not None:
        routing_bias = _ensure_tensor(routing_bias, "routing_bias")

    output_device = hidden_states.device

    routing_logits = _to_cuda_tensor(routing_logits, "routing_logits").contiguous()
    hidden_states = _to_cuda_tensor(hidden_states, "hidden_states").contiguous()
    hidden_states_scale = _to_cuda_tensor(
        hidden_states_scale, "hidden_states_scale"
    ).contiguous()
    gemm1_weights = _to_cuda_tensor(gemm1_weights, "gemm1_weights").contiguous()
    gemm1_weights_scale = _to_cuda_tensor(
        gemm1_weights_scale, "gemm1_weights_scale"
    ).contiguous()
    gemm2_weights = _to_cuda_tensor(gemm2_weights, "gemm2_weights").contiguous()
    gemm2_weights_scale = _to_cuda_tensor(
        gemm2_weights_scale, "gemm2_weights_scale"
    ).contiguous()
    if routing_bias is not None:
        routing_bias = _to_cuda_tensor(routing_bias, "routing_bias").contiguous()

    local_expert_offset = _normalize_int_scalar(local_expert_offset)
    routed_scaling_factor = _normalize_float_scalar(routed_scaling_factor)

    seq_len, num_experts = routing_logits.shape
    local_num_experts = gemm1_weights.shape[0]

    assert num_experts == NUM_EXPERTS_GLOBAL
    assert local_num_experts == NUM_LOCAL_EXPERTS
    assert hidden_states.shape == (seq_len, HIDDEN_SIZE)
    assert hidden_states_scale.shape == (NUM_HIDDEN_BLOCKS, seq_len)
    assert gemm1_weights.shape == (NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert gemm1_weights_scale.shape == (
        NUM_LOCAL_EXPERTS,
        NUM_GEMM1_OUT_BLOCKS,
        NUM_HIDDEN_BLOCKS,
    )
    assert gemm2_weights.shape == (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    assert gemm2_weights_scale.shape == (
        NUM_LOCAL_EXPERTS,
        NUM_HIDDEN_BLOCKS,
        NUM_INTERMEDIATE_BLOCKS,
    )
    assert routing_bias is None or routing_bias.shape[-1] == NUM_EXPERTS_GLOBAL

    output = torch.zeros(
        (seq_len, HIDDEN_SIZE), dtype=torch.float32, device=routing_logits.device
    )

    plan = _build_local_plan(
        routing_logits,
        routing_bias,
        local_expert_offset,
        routed_scaling_factor,
    )
    if plan is None:
        return output.to(torch.bfloat16).to(output_device)

    token_sorted, w_sorted, active_local_ids, offs = plan
    a = _dequant_hidden_states(hidden_states, hidden_states_scale)
    a_cat = a.index_select(0, token_sorted).contiguous()

    w13_t_all, w2_t_all = _get_dequantized_weight_pack(
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    w13_t = w13_t_all.index_select(0, active_local_ids).contiguous()
    w2_t = w2_t_all.index_select(0, active_local_ids).contiguous()

    g1 = torch.ops.aten._grouped_mm(a_cat, w13_t, offs)
    c = _swiglu_triton(g1)
    o = torch.ops.aten._grouped_mm(c, w2_t, offs)

    output.index_add_(0, token_sorted, o.to(torch.float32) * w_sorted.unsqueeze(1))
    return output.to(torch.bfloat16).to(output_device)
