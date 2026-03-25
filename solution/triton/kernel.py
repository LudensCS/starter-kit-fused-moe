import torch
import torch.nn.functional as F
import triton
import triton.language as tl


BLOCK_Q = 128
H_CONST = 7168
I_CONST = 2048
GEMM1_OUT_CONST = 4096
E_GLOBAL_CONST = 256
E_LOCAL_CONST = 32
N_GROUP_CONST = 8
TOPK_GROUP_CONST = 4
TOP_K_CONST = 8

NUM_HIDDEN_BLOCKS = H_CONST // BLOCK_Q
NUM_INTERMEDIATE_BLOCKS = I_CONST // BLOCK_Q
NUM_GEMM1_OUT_BLOCKS = GEMM1_OUT_CONST // BLOCK_Q


@triton.jit
def _dequant_hidden_selected_kernel(
    x_ptr,
    scale_ptr,
    row_idx_ptr,
    out_ptr,
    stride_x_t,
    stride_x_h,
    stride_s_b,
    stride_s_t,
    stride_r,
    stride_o_t,
    stride_o_h,
    n_rows,
    H,
    BLOCK_H: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    if pid_row >= n_rows:
        return

    src_row = tl.load(row_idx_ptr + pid_row * stride_r)
    offs_h = pid_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    x = tl.load(
        x_ptr + src_row * stride_x_t + offs_h * stride_x_h,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)
    s = tl.load(scale_ptr + pid_blk * stride_s_b + src_row * stride_s_t).to(tl.float32)
    y = x * s
    tl.store(
        out_ptr + pid_row * stride_o_t + offs_h * stride_o_h,
        y,
        mask=mask_h,
    )


@triton.jit
def _dequant_single_expert_weight_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    expert_idx,
    stride_w_e,
    stride_w_m,
    stride_w_k,
    stride_s_e,
    stride_s_mb,
    stride_s_kb,
    stride_o_m,
    stride_o_k,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_mb = tl.program_id(0)
    pid_kb = tl.program_id(1)

    offs_m = pid_mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    w = tl.load(
        w_ptr
        + expert_idx * stride_w_e
        + offs_m[:, None] * stride_w_m
        + offs_k[None, :] * stride_w_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    s = tl.load(
        s_ptr + expert_idx * stride_s_e + pid_mb * stride_s_mb + pid_kb * stride_s_kb
    ).to(tl.float32)
    y = w * s
    tl.store(
        out_ptr + offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k,
        y,
        mask=mask,
    )


def _ensure_cuda_available():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this Triton implementation, but CUDA is not available."
        )


def _infer_primary_device(*args, **kwargs):
    for x in list(args) + list(kwargs.values()):
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")


def _move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x if x.device == device else x.to(device)
    return x


def _prepare_inputs(named_inputs, device):
    out = {}
    for k, v in named_inputs.items():
        out[k] = _move_to_device(v, device) if isinstance(v, torch.Tensor) else v
    return out


def _validate_shapes(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
):
    if routing_logits.ndim != 2:
        raise ValueError(f"routing_logits must be 2D, got ndim={routing_logits.ndim}")
    T = routing_logits.shape[0]

    if tuple(routing_logits.shape) != (T, E_GLOBAL_CONST):
        raise ValueError(
            f"routing_logits must have shape ({T}, {E_GLOBAL_CONST}), got {tuple(routing_logits.shape)}"
        )
    if routing_bias.shape[-1] != E_GLOBAL_CONST:
        raise ValueError(
            f"routing_bias last dim must be {E_GLOBAL_CONST}, got {routing_bias.shape[-1]}"
        )
    if tuple(hidden_states.shape) != (T, H_CONST):
        raise ValueError(
            f"hidden_states must have shape ({T}, {H_CONST}), got {tuple(hidden_states.shape)}"
        )
    if tuple(hidden_states_scale.shape) != (NUM_HIDDEN_BLOCKS, T):
        raise ValueError(
            f"hidden_states_scale must have shape ({NUM_HIDDEN_BLOCKS}, {T}), got {tuple(hidden_states_scale.shape)}"
        )
    if tuple(gemm1_weights.shape) != (E_LOCAL_CONST, GEMM1_OUT_CONST, H_CONST):
        raise ValueError(
            f"gemm1_weights must have shape ({E_LOCAL_CONST}, {GEMM1_OUT_CONST}, {H_CONST}), got {tuple(gemm1_weights.shape)}"
        )
    if tuple(gemm1_weights_scale.shape) != (
        E_LOCAL_CONST,
        NUM_GEMM1_OUT_BLOCKS,
        NUM_HIDDEN_BLOCKS,
    ):
        raise ValueError(
            f"gemm1_weights_scale must have shape ({E_LOCAL_CONST}, {NUM_GEMM1_OUT_BLOCKS}, {NUM_HIDDEN_BLOCKS}), got {tuple(gemm1_weights_scale.shape)}"
        )
    if tuple(gemm2_weights.shape) != (E_LOCAL_CONST, H_CONST, I_CONST):
        raise ValueError(
            f"gemm2_weights must have shape ({E_LOCAL_CONST}, {H_CONST}, {I_CONST}), got {tuple(gemm2_weights.shape)}"
        )
    if tuple(gemm2_weights_scale.shape) != (
        E_LOCAL_CONST,
        NUM_HIDDEN_BLOCKS,
        NUM_INTERMEDIATE_BLOCKS,
    ):
        raise ValueError(
            f"gemm2_weights_scale must have shape ({E_LOCAL_CONST}, {NUM_HIDDEN_BLOCKS}, {NUM_INTERMEDIATE_BLOCKS}), got {tuple(gemm2_weights_scale.shape)}"
        )


def _dequant_hidden_selected_triton(hidden_states, hidden_states_scale, row_idx):
    n_rows = int(row_idx.numel())
    out = torch.empty(
        (n_rows, H_CONST), device=hidden_states.device, dtype=torch.float32
    )
    if n_rows == 0:
        return out

    x = hidden_states.contiguous()
    s = hidden_states_scale.contiguous()
    row_idx = row_idx.contiguous()

    grid = (n_rows, NUM_HIDDEN_BLOCKS)
    _dequant_hidden_selected_kernel[grid](
        x,
        s,
        row_idx,
        out,
        x.stride(0),
        x.stride(1),
        s.stride(0),
        s.stride(1),
        row_idx.stride(0),
        out.stride(0),
        out.stride(1),
        n_rows,
        H_CONST,
        BLOCK_H=BLOCK_Q,
        num_warps=8,
        num_stages=4,
    )
    return out


def _dequant_single_w13_triton(gemm1_weights, gemm1_weights_scale, local_expert_idx):
    out = torch.empty(
        (GEMM1_OUT_CONST, H_CONST), device=gemm1_weights.device, dtype=torch.float32
    )
    w = gemm1_weights.contiguous()
    s = gemm1_weights_scale.contiguous()

    grid = (NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS)
    _dequant_single_expert_weight_kernel[grid](
        w,
        s,
        out,
        int(local_expert_idx),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        s.stride(0),
        s.stride(1),
        s.stride(2),
        out.stride(0),
        out.stride(1),
        GEMM1_OUT_CONST,
        H_CONST,
        BLOCK_M=BLOCK_Q,
        BLOCK_K=BLOCK_Q,
        num_warps=8,
        num_stages=4,
    )
    return out


def _dequant_single_w2_triton(gemm2_weights, gemm2_weights_scale, local_expert_idx):
    out = torch.empty(
        (H_CONST, I_CONST), device=gemm2_weights.device, dtype=torch.float32
    )
    w = gemm2_weights.contiguous()
    s = gemm2_weights_scale.contiguous()

    grid = (NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS)
    _dequant_single_expert_weight_kernel[grid](
        w,
        s,
        out,
        int(local_expert_idx),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        s.stride(0),
        s.stride(1),
        s.stride(2),
        out.stride(0),
        out.stride(1),
        H_CONST,
        I_CONST,
        BLOCK_M=BLOCK_Q,
        BLOCK_K=BLOCK_Q,
        num_warps=8,
        num_stages=4,
    )
    return out


def _route_local_compact(
    routing_logits,
    routing_bias,
    local_expert_offset,
    routed_scaling_factor,
):
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    group_size = E_GLOBAL_CONST // N_GROUP_CONST
    grouped = s_with_bias.view(-1, N_GROUP_CONST, group_size)

    top2_vals, _ = torch.topk(grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    _, group_idx = torch.topk(
        group_scores, k=TOPK_GROUP_CONST, dim=1, largest=True, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    score_mask = (
        group_mask.unsqueeze(2)
        .expand(-1, N_GROUP_CONST, group_size)
        .reshape(-1, E_GLOBAL_CONST)
    )
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)

    _, topk_idx = torch.topk(
        scores_pruned, k=TOP_K_CONST, dim=1, largest=True, sorted=False
    )

    topk_s = s.gather(1, topk_idx)
    topk_s_sum = topk_s.sum(dim=1, keepdim=True) + 1.0e-20
    topk_weights = (topk_s / topk_s_sum) * float(routed_scaling_factor)

    device = routing_logits.device
    topk_local = topk_idx - int(local_expert_offset)
    valid_slot = (topk_local >= 0) & (topk_local < E_LOCAL_CONST)

    if not bool(valid_slot.any().item()):
        empty = torch.empty((0,), device=device, dtype=torch.int64)
        return empty, {}, []

    token_grid = (
        torch.arange(topk_idx.shape[0], device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, TOP_K_CONST)
    )
    pair_token_idx = token_grid[valid_slot]
    pair_local_idx = topk_local[valid_slot].to(torch.int64)
    pair_weight = topk_weights[valid_slot]

    order = torch.argsort(pair_local_idx)
    pair_token_idx = pair_token_idx[order]
    pair_local_idx = pair_local_idx[order]
    pair_weight = pair_weight[order]

    counts = torch.bincount(pair_local_idx, minlength=E_LOCAL_CONST)
    active_local = torch.nonzero(counts > 0, as_tuple=False).squeeze(1)
    selected_rows = torch.unique(pair_token_idx, sorted=True)

    expert_data = {}
    start = 0
    for le in active_local.tolist():
        cnt = int(counts[le].item())
        end = start + cnt
        expert_data[int(le)] = (pair_token_idx[start:end], pair_weight[start:end])
        start = end

    return selected_rows, expert_data, active_local.tolist()


def _accumulate_expert_outputs_batched(
    output,
    A_sel,
    selected_rows,
    pos_map,
    expert_data,
    active_local,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
):
    if len(active_local) == 0:
        return

    n_active = len(active_local)
    max_tokens = max((expert_data[le][0].numel() for le in active_local), default=0)
    if max_tokens == 0:
        return

    # pad to uniform shape for batch GEMM
    tokens_padded = torch.full(
        (n_active, max_tokens), selected_rows[0], device=A_sel.device, dtype=torch.int64
    )
    weights_padded = torch.zeros(
        (n_active, max_tokens), device=A_sel.device, dtype=torch.float32
    )
    for i, le in enumerate(active_local):
        tok_idx, w_tok = expert_data[le]
        if tok_idx.numel() == 0:
            continue
        tokens_padded[i, : tok_idx.numel()] = tok_idx
        weights_padded[i, : tok_idx.numel()] = w_tok

    # Map to selected row indices and gather A
    flat_tokens = tokens_padded.view(-1)
    selected_pos = pos_map[flat_tokens]
    A_block = A_sel.index_select(0, selected_pos)
    A_block = A_block.view(n_active, max_tokens, H_CONST)

    # Dequantize expert weights and stack
    W13_list = []
    W2_list = []
    for le in active_local:
        W13_e = _dequant_single_w13_triton(gemm1_weights, gemm1_weights_scale, le)
        W2_e = _dequant_single_w2_triton(gemm2_weights, gemm2_weights_scale, le)
        W13_list.append(W13_e)
        W2_list.append(W2_e)

    W13_all = torch.stack(W13_list, dim=0)
    W2_all = torch.stack(W2_list, dim=0)

    # Batched GEMM1 -> SwiGLU -> GEMM2
    G1 = torch.bmm(A_block, W13_all.transpose(1, 2))
    X1 = G1[:, :, :I_CONST]
    X2 = G1[:, :, I_CONST:]
    C = X1 * F.silu(X2)
    O = torch.bmm(C, W2_all.transpose(1, 2))

    # Accumulate
    scaled_O = O * weights_padded.unsqueeze(-1)
    output.index_add_(0, flat_tokens, scaled_O.view(-1, H_CONST))


@torch.no_grad()
def run(*args, **kwargs):
    _ensure_cuda_available()

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    arg_names = [
        "routing_logits",
        "routing_bias",
        "hidden_states",
        "hidden_states_scale",
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
        "local_expert_offset",
        "routed_scaling_factor",
    ]

    values = {}
    for i, name in enumerate(arg_names):
        if i < len(args):
            values[name] = args[i]
        elif name in kwargs:
            values[name] = kwargs[name]
        else:
            raise TypeError(f"Missing required argument: {name}")

    original_device = _infer_primary_device(*args, **kwargs)
    cuda_device = torch.device("cuda")
    prepared = _prepare_inputs(values, cuda_device)

    routing_logits = prepared["routing_logits"]
    routing_bias = prepared["routing_bias"]
    hidden_states = prepared["hidden_states"]
    hidden_states_scale = prepared["hidden_states_scale"]
    gemm1_weights = prepared["gemm1_weights"]
    gemm1_weights_scale = prepared["gemm1_weights_scale"]
    gemm2_weights = prepared["gemm2_weights"]
    gemm2_weights_scale = prepared["gemm2_weights_scale"]
    local_expert_offset = int(prepared["local_expert_offset"])
    routed_scaling_factor = float(prepared["routed_scaling_factor"])

    if routing_logits.device.type != "cuda":
        raise RuntimeError("routing_logits must be on CUDA after device preparation.")
    if hidden_states.device.type != "cuda":
        raise RuntimeError("hidden_states must be on CUDA after device preparation.")

    _validate_shapes(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )

    T = routing_logits.shape[0]
    device = hidden_states.device
    output = torch.zeros((T, H_CONST), dtype=torch.float32, device=device)

    selected_rows, expert_data, active_local = _route_local_compact(
        routing_logits,
        routing_bias,
        local_expert_offset,
        routed_scaling_factor,
    )

    if selected_rows.numel() == 0 or len(active_local) == 0:
        result = output.to(torch.bfloat16)
        if original_device.type != "cuda":
            result = result.to(original_device)
        return result

    A_sel = _dequant_hidden_selected_triton(
        hidden_states, hidden_states_scale, selected_rows
    )

    pos_map = torch.full((T,), -1, device=device, dtype=torch.int64)
    pos_map[selected_rows] = torch.arange(
        selected_rows.numel(), device=device, dtype=torch.int64
    )

    _accumulate_expert_outputs_batched(
        output,
        A_sel,
        selected_rows,
        pos_map,
        expert_data,
        active_local,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )

    result = output.to(torch.bfloat16)
    if original_device.type != "cuda":
        result = result.to(original_device)
    return result
