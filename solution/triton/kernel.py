import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dtype == torch.float32 and B.dtype == torch.float32
    assert A.device == B.device
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, 'Incompatible matmul dimensions'

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def _to_cuda_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == 'cuda':
        return tensor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run the Triton MoE kernel.')
    return tensor.cuda()


def _ensure_tensor(x, name: str):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    return x


def _expand_scale(scale: torch.Tensor, repeat_dim1: int, repeat_dim2: int) -> torch.Tensor:
    return scale.to(torch.float32).repeat_interleave(repeat_dim1, dim=1).repeat_interleave(repeat_dim2, dim=2)


def _expand_hidden_scale(hidden_states_scale: torch.Tensor, block_size: int) -> torch.Tensor:
    return hidden_states_scale.to(torch.float32).permute(1, 0).contiguous().repeat_interleave(block_size, dim=1)


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
    H = 7168
    I = 2048
    E_global = 256
    E_local = gemm1_weights.shape[0]
    BLOCK = 128

    assert H == 7168, 'hidden_size must be 7168'
    assert I == 2048, 'intermediate_size must be 2048'
    assert E_global == 256, 'num_experts must be 256'
    assert E_local == 32, 'num_local_experts must be 32'

    T = routing_logits.shape[0]
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (H // BLOCK, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, (2 * I) // BLOCK, H // BLOCK)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, H // BLOCK, I // BLOCK)
    assert routing_bias is None or routing_bias.shape[-1] == E_global

    if isinstance(local_expert_offset, torch.Tensor):
        local_expert_offset = int(local_expert_offset.item())
    else:
        local_expert_offset = int(local_expert_offset)
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())
    else:
        routed_scaling_factor = float(routed_scaling_factor)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run the Triton MoE kernel.')

    original_device = hidden_states.device
    device = torch.device('cuda')

    routing_logits = _to_cuda_tensor(_ensure_tensor(routing_logits, 'routing_logits'))
    hidden_states = _to_cuda_tensor(_ensure_tensor(hidden_states, 'hidden_states'))
    hidden_states_scale = _to_cuda_tensor(
        _ensure_tensor(hidden_states_scale, 'hidden_states_scale')
    )
    gemm1_weights = _to_cuda_tensor(_ensure_tensor(gemm1_weights, 'gemm1_weights'))
    gemm1_weights_scale = _to_cuda_tensor(
        _ensure_tensor(gemm1_weights_scale, 'gemm1_weights_scale')
    )
    gemm2_weights = _to_cuda_tensor(_ensure_tensor(gemm2_weights, 'gemm2_weights'))
    gemm2_weights_scale = _to_cuda_tensor(
        _ensure_tensor(gemm2_weights_scale, 'gemm2_weights_scale')
    )
    if routing_bias is not None:
        routing_bias = _to_cuda_tensor(_ensure_tensor(routing_bias, 'routing_bias'))

    hidden_states = hidden_states.contiguous()
    routing_logits = routing_logits.contiguous()
    hidden_states_scale = hidden_states_scale.contiguous()
    gemm1_weights = gemm1_weights.contiguous()
    gemm1_weights_scale = gemm1_weights_scale.contiguous()
    gemm2_weights = gemm2_weights.contiguous()
    gemm2_weights_scale = gemm2_weights_scale.contiguous()

    A_scale = _expand_hidden_scale(hidden_states_scale, BLOCK)
    A = hidden_states.to(torch.float32) * A_scale

    W13 = gemm1_weights.to(torch.float32) * _expand_scale(gemm1_weights_scale, BLOCK, BLOCK)
    W2 = gemm2_weights.to(torch.float32) * _expand_scale(gemm2_weights_scale, BLOCK, BLOCK)

    logits = routing_logits.to(torch.float32)
    bias = None if routing_bias is None else routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits + bias if bias is not None else logits)
    s_with_bias = s

    group_size = E_global // 8
    s_wb_grouped = s_with_bias.view(T, 8, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=4, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, 8, group_size).reshape(T, E_global)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=8, dim=1, largest=True, sorted=False)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-20)
    weights = (weights / weights_sum) * routed_scaling_factor

    local_start = local_expert_offset
    local_end = local_start + E_local
    expert_weights = weights[:, local_start:local_end]

    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    for expert_idx in range(E_local):
        gate = expert_weights[:, expert_idx : expert_idx + 1]
        if gate.abs().max() == 0.0:
            continue

        W13_e = W13[expert_idx]
        W2_e = W2[expert_idx]
        hidden_proj = torch.matmul(A, W13_e.t())
        x1 = hidden_proj[:, :I]
        x2 = hidden_proj[:, I:]
        activated = x1 * torch.sigmoid(x2)
        expert_output = torch.matmul(activated, W2_e.t())
        output += expert_output * gate

    return output.to(torch.bfloat16).to(original_device)
