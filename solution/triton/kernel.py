"""DeepSeek-V3/R1 FP8 block-scale MoE 的 Triton 实现。

整体执行流程如下：
1. `_routing_topk_kernel`
   对每个 token 计算 no-aux routing，选出 top-8 expert，并统计本 rank 上
   32 个 local expert 各自接收到多少条 token 路由。
2. `_prefix_sum32_kernel` + `_scatter_local_kernel`
   把 “token 选中了哪些 local expert” 重新整理成按 expert 连续存放的布局，
   方便后续 grouped GEMM。
3. `_gemm1_swiglu_kernel`
   对每个 local expert 执行第一层 GEMM（W13），并在 kernel 内直接做 SwiGLU。
4. `_gemm2_scatter_kernel`
   对中间激活执行第二层 GEMM（W2），再按 routing weight 乘权写回最终输出。

这里的核心点是：
- 输入 hidden states / 权重都是 FP8，配合 block-wise scale 在 kernel 内边读边反量化。
- 只计算本 rank 的 local expert，避免无意义的全量专家计算。
- 最终输出是 bfloat16，与题目接口保持一致。
"""

from typing import Any, cast

import torch
import triton
import triton.language as tl

# 题目固定的几何参数。这里全都写死，是为了让 Triton 编译器更激进地做常量传播。
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

# GEMM1 是 W13（gate/up projection）的 tile 参数。
# 这一组参数目前保持手工指定，不走 autotune。
GEMM1_BLOCK_M = 128
GEMM1_BLOCK_N = 128
GEMM1_BLOCK_K = 128
GEMM1_NUM_WARPS = 8
GEMM1_NUM_STAGES = 3

# GEMM2 是 down projection + scatter accumulate。这里剪掉运行时 autotune 候选，
# 只保留小/大 M 两条手工分支，避免每个 max_count key 都试跑多组配置。
GEMM2_SMALL_BLOCK_M = 32
GEMM2_LARGE_BLOCK_M = 64
GEMM2_LARGE_M_THRESHOLD = 64
GEMM2_BLOCK_N = 128
GEMM2_BLOCK_K = 64
GEMM2_NUM_WARPS = 4
GEMM2_SMALL_NUM_STAGES = 3
GEMM2_LARGE_NUM_STAGES = 2

# OPTIMIZATION: small-batch path threshold.
# Workloads with 2..128 tokens are dominated by CPU sync and tiny-kernel launch
# overhead. They use a GPU-only routing/scatter path and dense GEMM grids below.
# seq_len==1 stays on the compact path because dense 32-expert grids waste work.
SYNC_FREE_SEQ_THRESHOLD = 128


@triton.jit
def _routing_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    count_ptr,
    token_local_count_ptr,
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
    """对单个 token 完成 routing，并顺手初始化输出。

    这个 kernel 一次处理一个 token，完成以下事情：
    - 基于 sigmoid(logits) + bias 计算 no-aux routing score
    - 先按 group 选 topk_group=4 个 group，再在保留组内选全局 top_k=8 个 expert
    - 记录 topk expert id 与归一化后的 routing weight
    - 统计落在本 rank 的 local expert 的样本数
    - 把该 token 的最终输出行先清零，便于后续 GEMM2 用 atomic_add 累加
    """
    pid = tl.program_id(0)
    expert_offs = tl.arange(0, 256)
    group_offs = tl.arange(0, 8)

    # 为了让相同分数时结果稳定，给 expert 和 group 一个极小的 tie-break 偏置。
    expert_tiebreak = expert_offs.to(tl.float32) * -1e-6
    group_tiebreak = group_offs.to(tl.float32) * -1e-4

    # score = sigmoid(logits) + bias
    logits = tl.load(logits_ptr + pid * stride_lm + expert_offs * stride_ln).to(tl.float32)
    bias = tl.load(bias_ptr + expert_offs).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-logits))
    score = sig + bias
    score_tie = tl.reshape(score + expert_tiebreak, (8, 32))

    # 每个 group 取 top-2 分数求和，作为该 group 的 group score。
    max1 = tl.max(score_tie, axis=1)
    score2 = tl.where(score_tie == max1[:, None], -1.0e30, score_tie)
    max2 = tl.max(score2, axis=1)
    group_scores = max1 + max2

    # 在 8 个 group 中再选 top-4 个 group。
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

    # 在保留的 group 内做全局 top-8 expert 选择。
    # 注意：最终乘权用的是 sigmoid(logits)（不含 bias），与题目 reference 保持一致。
    total_sig = 0.0
    local_hits = tl.zeros((), dtype=tl.int32)
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
        local_hits += tl.where(local_mask, 1, 0)
        total_sig += expert_sig
        candidate_scores = tl.where(chosen_expert, -1.0e30, candidate_scores)

    # 对 top-k 权重做归一化，再乘 routed_scaling_factor。
    for k in range(8):
        weight = tl.load(topk_weights_ptr + pid * stride_twm + k * stride_twn).to(tl.float32)
        tl.store(
            topk_weights_ptr + pid * stride_twm + k * stride_twn,
            tl.where(total_sig > 1.0e-20, weight / total_sig, 1.0) * routed_scaling_factor,
        )

    tl.store(token_local_count_ptr + pid, local_hits)

    # 后续 GEMM2 可能用 atomic_add 累加，所以先把这条 token 的输出整行清零。
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
    """对 32 个 local expert 的计数做前缀和。

    输出的 `offset_ptr[i]` 表示 expert i 在排序后缓冲区中的起始位置。
    由于 local expert 数固定为 32，这里直接串行展开即可。
    """
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
    """把 token 的 top-k 路由结果按 local expert 重排。

    输入还是 token-major 布局：每个 token 有 8 个 expert。
    输出改成 expert-major 的连续布局：
    - `sorted_token_ptr`：某个 local expert 命中的 token id 列表
    - `sorted_weight_ptr`：对应的 routing weight

    这样 GEMM1/GEMM2 就能按 expert 分组顺序处理，避免在 GEMM 内做复杂分支。
    """
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

    # `cursor_ptr` 是每个 local expert 的写入游标。
    # 配合 offsets，可以把不同 token 的命中项稳定地写到 expert 连续区间内。
    pos = tl.atomic_add(cursor_ptr + local_safe, 1, mask=local_mask)
    base = tl.load(offsets_ptr + local_safe, mask=local_mask, other=0).to(tl.int32)
    dest = base + pos

    tl.store(sorted_token_ptr + dest * stride_st, token_ids, mask=local_mask)
    tl.store(sorted_weight_ptr + dest * stride_sw, weights, mask=local_mask)


@triton.jit
def _small_prefix_scatter_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    count_ptr,
    offsets_ptr,
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
    """OPTIMIZATION: small-batch fused prefix sum + expert-major scatter.

    One launch replaces `_prefix_sum32_kernel`, cursor zeroing and
    `_scatter_local_kernel` for 2..128 token workloads. Each program owns one
    local expert, computes its prefix offset from `counts`, then compacts hits
    using vector cumsum. This keeps small batches off the host sync path.
    """
    pid_e = tl.program_id(0)

    expert_offs = tl.arange(0, 32)
    counts = tl.load(count_ptr + expert_offs).to(tl.int32)
    start = tl.sum(tl.where(expert_offs < pid_e, counts, 0), axis=0)
    tl.store(offsets_ptr + pid_e, start)

    offs = tl.arange(0, BLOCK)
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
    hit = mask & (local == pid_e)
    rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
    dest = start + rank

    tl.store(sorted_token_ptr + dest * stride_st, token_ids, mask=hit)
    tl.store(sorted_weight_ptr + dest * stride_sw, weights, mask=hit)


@triton.jit
def _gemm1_swiglu_kernel(
    hidden_ptr,
    hidden_scale_ptr,
    token_ptr,
    offset_ptr,
    count_ptr,
    job_expert_ptr,
    job_m_tile_ptr,
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
    K_TILES: tl.constexpr,
):
    """对每个 local expert 执行 GEMM1，并在 kernel 内直接做 SwiGLU。

    输入：
    - hidden states: FP8 + block scale
    - gemm1 weights(W13): FP8 + block scale

    输出：
    - 每个 routed slot 的中间激活，形状为 [total_local_rows, INTERMEDIATE_SIZE]

    这里把 gate/up 两个投影一起做，是为了复用同一份输入读取和 K 维遍历。
    """
    pid_n = tl.program_id(0)
    pid_job = tl.program_id(1)
    pid_e = tl.load(job_expert_ptr + pid_job).to(tl.int32)
    pid_m = tl.load(job_m_tile_ptr + pid_job).to(tl.int32)

    # host 端只生成有效 row-block job；这里保留防御性检查。
    count = tl.load(count_ptr + pid_e * stride_c).to(tl.int32)
    if pid_m * BLOCK_M >= count:
        return

    # 根据 prefix-sum 算出的起始位置，定位这个 expert 对应的 token 列表。
    start = tl.load(offset_ptr + pid_e * stride_s).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_ids = start + offs_m
    mask_m = offs_m < count
    token_ids = tl.load(
        token_ptr + slot_ids * stride_t,
        mask=mask_m,
        other=0,
    ).to(tl.int32)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < 2048
    gate_scale_n = (pid_n * BLOCK_N) // 128
    up_scale_n = gate_scale_n + 16

    # 分别累加 gate 和 up 两路 matmul 结果，最后再做 SwiGLU。
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K_TILES):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        scale_k = (kb * BLOCK_K) // 128

        # hidden states 采用 [T, H] 的 FP8 存储；scale 的布局是 [H/128, T]。
        hidden = tl.load(
            hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_scale = tl.load(
            hidden_scale_ptr + scale_k * stride_hsm + token_ids * stride_hsn,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        hidden = (hidden * hidden_scale[:, None]).to(tl.bfloat16)

        # W13 的前半段是 gate，后半段是 up。
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
            weight_scale_ptr
            + pid_e * stride_wse
            + gate_scale_n * stride_wsn
            + scale_k * stride_wsk
        ).to(tl.float32)
        up_scale = tl.load(
            weight_scale_ptr
            + pid_e * stride_wse
            + up_scale_n * stride_wsn
            + scale_k * stride_wsk
        ).to(tl.float32)

        gate = (gate * gate_scale).to(tl.bfloat16)
        up = (up * up_scale).to(tl.bfloat16)

        # tl.dot 这里走的是 Triton 为当前数据类型生成的矩阵乘路径。
        acc_gate += tl.dot(hidden, gate)
        acc_up += tl.dot(hidden, up)

    # SwiGLU: gate * silu(up)
    out = acc_gate * (acc_up / (1.0 + tl.exp(-acc_up)))
    tl.store(
        out_ptr + slot_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _gemm1_swiglu_dense_kernel(
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
    K_TILES: tl.constexpr,
):
    # OPTIMIZATION: dense expert x row-block grid for small batches.
    # The compact job-list path needs tiny torch tensor creation and host-side
    # planning. Dense grids avoid that overhead; empty expert tiles return after
    # the count check below.
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
    gate_scale_n = (pid_n * BLOCK_N) // 128
    up_scale_n = gate_scale_n + 16

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(K_TILES):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        scale_k = (kb * BLOCK_K) // 128
        hidden = tl.load(
            hidden_ptr + token_ids[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        hidden_scale = tl.load(
            hidden_scale_ptr + scale_k * stride_hsm + token_ids * stride_hsn,
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
            weight_scale_ptr
            + pid_e * stride_wse
            + gate_scale_n * stride_wsn
            + scale_k * stride_wsk
        ).to(tl.float32)
        up_scale = tl.load(
            weight_scale_ptr
            + pid_e * stride_wse
            + up_scale_n * stride_wsn
            + scale_k * stride_wsk
        ).to(tl.float32)

        acc_gate += tl.dot(hidden, (gate * gate_scale).to(tl.bfloat16))
        acc_up += tl.dot(hidden, (up * up_scale).to(tl.bfloat16))

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
    token_local_count_ptr,
    offset_ptr,
    count_ptr,
    job_expert_ptr,
    job_m_tile_ptr,
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
    """对中间激活执行 GEMM2，并直接把结果 scatter 回最终输出。

    这个 kernel 与普通 GEMM 的最大区别是最后一步：
    - 每一行结果并不写到连续输出，而是根据 `token_ids` 写回原 token 位置
    - 同一个 token 会被多个 expert 命中，所以这里必须使用 atomic_add 做累加

    也因此，GEMM2 的性能既受 matmul 本身影响，也会受到 scatter/atomic 行为影响。
    """
    pid_n = tl.program_id(0)
    pid_job = tl.program_id(1)
    pid_e = tl.load(job_expert_ptr + pid_job).to(tl.int32)
    pid_m = tl.load(job_m_tile_ptr + pid_job).to(tl.int32)

    # host 端只生成有效 row-block job；这里保留防御性检查。
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

    # 与 GEMM1 不同，这里输出维是 H=7168，因此 scale 是 [H/128, I/128]。
    # `scale_n` 需要按输出列逐列计算，才能兼容 BLOCK_N=256 这种跨两个 scale block 的配置。
    scale_n = offs_n // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 这里显式写 2048，是因为题目把 intermediate size 固定死了。
    for kb in range(0, 2048, BLOCK_K):
        offs_k = kb + tl.arange(0, BLOCK_K)
        scale_k = kb // 128
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
            weight_scale_ptr
            + pid_e * stride_wse
            + scale_n * stride_wsn
            + scale_k * stride_wsk,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        weight = (weight * weight_scale[None, :]).to(tl.bfloat16)
        acc += tl.dot(act, weight)

    # 只有同一个 token 在本 rank 命中多个 local expert 时才需要 atomic。
    # 单 local contribution 的 token 直接 store，避免不必要的 BF16 atomic 写回开销。
    out_ptrs = out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    token_local_counts = tl.load(
        token_local_count_ptr + token_ids,
        mask=mask_m,
        other=0,
    ).to(tl.int32)
    out_vals = (acc * row_weights[:, None]).to(tl.bfloat16)
    write_mask = mask_m[:, None] & mask_n[None, :]
    single_local = token_local_counts == 1
    tl.store(
        out_ptrs,
        out_vals,
        mask=write_mask & single_local[:, None],
    )
    tl.atomic_add(
        out_ptrs,
        out_vals,
        mask=write_mask & (token_local_counts > 1)[:, None],
        sem="relaxed",
    )


@triton.jit
def _gemm2_scatter_dense_kernel(
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
    # OPTIMIZATION: dense-grid GEMM2 for the small-batch path.
    # It intentionally uses relaxed atomic_add for every contribution. That is
    # cheaper here than loading token_local_counts and branching between store
    # and atomic in a launch dominated by small scattered output writes.
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
    scale_n = offs_n // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in range(0, 2048, BLOCK_K):
        offs_k = kb + tl.arange(0, BLOCK_K)
        scale_k = kb // 128
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
            weight_scale_ptr
            + pid_e * stride_wse
            + scale_n * stride_wsn
            + scale_k * stride_wsk,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(act, (weight * weight_scale[None, :]).to(tl.bfloat16))

    out_ptrs = out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(
        out_ptrs,
        (acc * row_weights[:, None]).to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
        sem="relaxed",
    )


def _as_cuda_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """确保输入张量位于 CUDA 上且内存连续。

    benchmark 框架不保证所有输入都已经是 CUDA contiguous tensor，这里统一做兜底。
    """
    if tensor.device.type != "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is unavailable, cannot move {name} from {tensor.device} to CUDA."
            )
        tensor = tensor.cuda(non_blocking=True)
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _build_tile_plan(
    counts_host: list[int],
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a compact list of valid (expert, row-block) jobs on host."""
    job_experts: list[int] = []
    job_m_tiles: list[int] = []
    for expert, count in enumerate(counts_host):
        for tile_m in range(triton.cdiv(count, block_m)):
            job_experts.append(expert)
            job_m_tiles.append(tile_m)

    return (
        torch.tensor(job_experts, dtype=torch.int32, device=device),
        torch.tensor(job_m_tiles, dtype=torch.int32, device=device),
    )


def _select_gemm2_config(max_count: int) -> tuple[int, int]:
    if max_count >= GEMM2_LARGE_M_THRESHOLD:
        return GEMM2_LARGE_BLOCK_M, GEMM2_LARGE_NUM_STAGES
    return GEMM2_SMALL_BLOCK_M, GEMM2_SMALL_NUM_STAGES


def _routing_and_scatter(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor | None,
    output: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    sync_counts: bool = True,
):
    """host 端路由整理函数。

    负责分配 routing 相关的中间缓冲区，并串起三个小 kernel：
    1. routing top-k
    2. 32 个 local expert 的 prefix sum
    3. 按 expert 重排 token / weight
    """
    seq_len = routing_logits.shape[0]
    total_slots = seq_len * TOP_K

    # topk_ids / topk_weights 仍然是 token-major。
    topk_ids = torch.empty((seq_len, TOP_K), dtype=torch.int32, device=routing_logits.device)
    topk_weights = torch.empty(
        (seq_len, TOP_K), dtype=torch.float32, device=routing_logits.device
    )

    # counts / offsets 是后续 expert-major 视图所需的辅助缓冲区。
    counts = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=routing_logits.device)
    offsets = torch.empty((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=routing_logits.device)
    token_local_counts = torch.empty(
        (seq_len,), dtype=torch.int32, device=routing_logits.device
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
        token_local_counts,
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

    if sync_counts:
        # Compact path for seq_len==1 and large batches:
        # synchronize counts to host once, then build a minimal GEMM job list.
        cast(Any, _prefix_sum32_kernel)[(1,)](
            counts,
            offsets,
            num_warps=1,
            num_stages=1,
        )
        counts_host = [int(v) for v in counts.detach().cpu().tolist()]
        total_local_slots = sum(counts_host)
    else:
        counts_host = None
        total_local_slots = total_slots

    sorted_token = torch.empty(
        (total_local_slots,), dtype=torch.int32, device=routing_logits.device
    )
    sorted_weight = torch.empty(
        (total_local_slots,), dtype=torch.float32, device=routing_logits.device
    )
    if total_local_slots == 0:
        return sorted_token, sorted_weight, counts, offsets, token_local_counts, counts_host

    if not sync_counts:
        # OPTIMIZATION: no counts.cpu() and no cursor zeroing for small batches.
        # The fused kernel computes offsets and writes expert-major queues.
        cast(Any, _small_prefix_scatter_kernel)[(NUM_LOCAL_EXPERTS,)](
            topk_ids,
            topk_weights,
            counts,
            offsets,
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
            BLOCK=1024,
            num_warps=8,
            num_stages=1,
        )
        return sorted_token, sorted_weight, counts, offsets, token_local_counts, counts_host

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
    return sorted_token, sorted_weight, counts, offsets, token_local_counts, counts_host


def _gemm1_swiglu_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_sorted: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    job_experts: torch.Tensor,
    job_m_tiles: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
) -> torch.Tensor:
    """host 端封装 GEMM1 + SwiGLU。"""
    total_rows = token_sorted.numel()
    out = torch.empty(
        (total_rows, INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    if total_rows == 0 or job_experts.numel() == 0:
        return out

    # grid.y 是压缩后的有效 (expert, m_tile) job，避免空 expert / 空 row-block launch。
    grid = (
        triton.cdiv(INTERMEDIATE_SIZE, GEMM1_BLOCK_N),
        job_experts.numel(),
    )
    cast(Any, _gemm1_swiglu_kernel)[grid](
        hidden_states,
        hidden_states_scale,
        token_sorted,
        expert_offsets,
        counts,
        job_experts,
        job_m_tiles,
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
        BLOCK_M=GEMM1_BLOCK_M,
        BLOCK_N=GEMM1_BLOCK_N,
        BLOCK_K=GEMM1_BLOCK_K,
        K_TILES=HIDDEN_SIZE // GEMM1_BLOCK_K,
        num_warps=GEMM1_NUM_WARPS,
        num_stages=GEMM1_NUM_STAGES,
    )
    return out


def _gemm1_swiglu_dense_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    token_sorted: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """OPTIMIZATION: small-batch GEMM1 wrapper using dense expert x row-block grid."""
    out = torch.empty(
        (token_sorted.numel(), INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    grid = (
        triton.cdiv(INTERMEDIATE_SIZE, GEMM1_BLOCK_N),
        triton.cdiv(seq_len, GEMM1_BLOCK_M),
        NUM_LOCAL_EXPERTS,
    )
    cast(Any, _gemm1_swiglu_dense_kernel)[grid](
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
        BLOCK_M=GEMM1_BLOCK_M,
        BLOCK_N=GEMM1_BLOCK_N,
        BLOCK_K=GEMM1_BLOCK_K,
        K_TILES=HIDDEN_SIZE // GEMM1_BLOCK_K,
        num_warps=GEMM1_NUM_WARPS,
        num_stages=GEMM1_NUM_STAGES,
    )
    return out


def _gemm2_scatter_triton(
    act: torch.Tensor,
    token_sorted: torch.Tensor,
    sorted_weight: torch.Tensor,
    token_local_counts: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    job_experts: torch.Tensor,
    job_m_tiles: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    block_m: int,
    num_stages: int,
):
    """host 端封装 GEMM2 + scatter accumulate。"""
    if act.numel() == 0 or job_experts.numel() == 0:
        return

    # grid.y 是压缩后的有效 (expert, m_tile) job，避免空 expert / 空 row-block launch。
    grid = (
        triton.cdiv(HIDDEN_SIZE, GEMM2_BLOCK_N),
        job_experts.numel(),
    )
    cast(Any, _gemm2_scatter_kernel)[grid](
        act,
        token_sorted,
        sorted_weight,
        token_local_counts,
        expert_offsets,
        counts,
        job_experts,
        job_m_tiles,
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
        BLOCK_M=block_m,
        BLOCK_N=GEMM2_BLOCK_N,
        BLOCK_K=GEMM2_BLOCK_K,
        num_warps=GEMM2_NUM_WARPS,
        num_stages=num_stages,
    )


def _gemm2_scatter_dense_triton(
    act: torch.Tensor,
    token_sorted: torch.Tensor,
    sorted_weight: torch.Tensor,
    counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    output: torch.Tensor,
    seq_len: int,
):
    """OPTIMIZATION: small-batch GEMM2 wrapper using dense expert x row-block grid."""
    if act.numel() == 0:
        return

    grid = (
        triton.cdiv(HIDDEN_SIZE, GEMM2_BLOCK_N),
        triton.cdiv(seq_len, GEMM2_SMALL_BLOCK_M),
        NUM_LOCAL_EXPERTS,
    )
    cast(Any, _gemm2_scatter_dense_kernel)[grid](
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
        BLOCK_M=GEMM2_SMALL_BLOCK_M,
        BLOCK_N=GEMM2_BLOCK_N,
        BLOCK_K=GEMM2_BLOCK_K,
        num_warps=GEMM2_NUM_WARPS,
        num_stages=GEMM2_SMALL_NUM_STAGES,
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
    output: torch.Tensor | None = None,
):
    """题目要求的统一入口函数。

    这个函数负责：
    - 处理输入设备 / contiguous / 标量参数格式
    - 准备最终输出缓冲区
    - 串起 routing -> GEMM1/SwiGLU -> GEMM2/scatter 三段主流程
    - 把结果按 benchmark 期望的方式返回或写入 `output`
    """
    output_tensor = output
    output_device = output_tensor.device if output_tensor is not None else hidden_states.device

    # 如果输入不在 CUDA，统一搬运到 CUDA；否则至少保证 contiguous。
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

    # benchmark 会传 destination-passing-style 的 output，这里先检查形状是否正确。
    expected_output_shape = (seq_len, HIDDEN_SIZE)
    if output_tensor is not None and tuple(output_tensor.shape) != expected_output_shape:
        raise ValueError(
            f"output shape mismatch: expected {expected_output_shape}, got {tuple(output_tensor.shape)}"
        )

    # 如果外部给的 output 不能直接作为内部工作缓冲区，就单独申请一个 CUDA bf16 buffer。
    if (
        output_tensor is not None
        and output_tensor.device.type == "cuda"
        and output_tensor.dtype == torch.bfloat16
        and output_tensor.is_contiguous()
    ):
        output_buffer = output_tensor
    else:
        output_buffer = torch.empty(
            expected_output_shape,
            dtype=torch.bfloat16,
            device=routing_logits.device,
        )

    def _finalize_output(result: torch.Tensor) -> torch.Tensor:
        # 统一处理 destination-passing-style 与非 DPS 两种返回方式。
        if output_tensor is None:
            return result if output_device.type == "cuda" else result.to(output_device)
        if result.data_ptr() != output_tensor.data_ptr():
            output_tensor.copy_(result.to(device=output_tensor.device, dtype=output_tensor.dtype))
        return output_tensor

    # 空输入直接返回空输出，避免后续 kernel launch。
    if seq_len == 0:
        return _finalize_output(output_buffer)

    # 第一段：routing，并把 token 路由整理成 expert-major 布局。
    # OPTIMIZATION: 小 batch 避免 counts.cpu() 硬同步；大 batch 保留压缩 job，
    # 减少空 GEMM tile。seq_len==1 也保留压缩路径，因为 dense 32-expert grid
    # 在单 token 下会引入过多空 tile。
    sync_counts = seq_len == 1 or seq_len > SYNC_FREE_SEQ_THRESHOLD
    token_sorted, weight_sorted, counts, expert_offsets, token_local_counts, counts_host = (
        _routing_and_scatter(
            routing_logits.to(torch.float32),
            routing_bias,
            output_buffer,
            local_expert_offset,
            routed_scaling_factor,
            sync_counts=sync_counts,
        )
    )

    if counts_host is None:
        # OPTIMIZATION: small-batch GPU-only path. Routing/scatter/GEMM planning
        # stays on device; GEMM kernels use dense grids and skip empty experts
        # inside the kernels with count checks.
        inter = _gemm1_swiglu_dense_triton(
            hidden_states,
            hidden_states_scale,
            token_sorted,
            counts,
            expert_offsets,
            gemm1_weights,
            gemm1_weights_scale,
            seq_len,
        )
        _gemm2_scatter_dense_triton(
            inter,
            token_sorted,
            weight_sorted,
            counts,
            expert_offsets,
            gemm2_weights,
            gemm2_weights_scale,
            output_buffer,
            seq_len,
        )
        return _finalize_output(output_buffer)

    # `counts_host` 来自 32 个 local expert 计数，用于剪枝出有效 GEMM tile job。
    max_count = max(counts_host)
    if max_count == 0:
        return _finalize_output(output_buffer)
    gemm2_block_m, gemm2_num_stages = _select_gemm2_config(max_count)
    gemm1_job_experts, gemm1_job_m_tiles = _build_tile_plan(
        counts_host,
        GEMM1_BLOCK_M,
        routing_logits.device,
    )
    gemm2_job_experts, gemm2_job_m_tiles = _build_tile_plan(
        counts_host,
        gemm2_block_m,
        routing_logits.device,
    )

    # 第二段：local expert 上的 GEMM1 + SwiGLU。
    inter = _gemm1_swiglu_triton(
        hidden_states,
        hidden_states_scale,
        token_sorted,
        counts,
        expert_offsets,
        gemm1_job_experts,
        gemm1_job_m_tiles,
        gemm1_weights,
        gemm1_weights_scale,
    )

    # 第三段：GEMM2，并带着 routing weight scatter 回原 token 输出。
    _gemm2_scatter_triton(
        inter,
        token_sorted,
        weight_sorted,
        token_local_counts,
        counts,
        expert_offsets,
        gemm2_job_experts,
        gemm2_job_m_tiles,
        gemm2_weights,
        gemm2_weights_scale,
        output_buffer,
        gemm2_block_m,
        gemm2_num_stages,
    )

    return _finalize_output(output_buffer)
