import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
NUM_LOCAL_EXPERTS = 32
TOP_K = 8
NUM_GROUPS = 8
TOPK_GROUP = 4
BLOCK_SIZE = 128
NUM_HIDDEN_BLOCKS = HIDDEN_SIZE // BLOCK_SIZE
NUM_INTERMEDIATE_BLOCKS = INTERMEDIATE_SIZE // BLOCK_SIZE
NUM_GEMM1_BLOCKS = (2 * INTERMEDIATE_SIZE) // BLOCK_SIZE

_FP8_MAX = 448.0

_routing_compiled_fn = None
_routing_compile_tried = False
_routing_result_cache = {}
_assignment_cache = {}
_permute_cache = {}
_CACHE_MAX_ENTRIES = 24


def _cache_put(cache: dict, key, value):
    cache[key] = value
    if len(cache) > _CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))

if triton is not None:

    @triton.jit
    def _dequant_hidden_states_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        num_tokens,
        num_hidden_blocks,
        stride_x_t,
        stride_x_h,
        stride_scale_hb,
        stride_scale_t,
        stride_out_t,
        stride_out_h,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        SWIZZLE_H: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        pid_t = pid // num_hidden_blocks
        pid_h_linear = pid % num_hidden_blocks

        swizzle_lane = pid_t % SWIZZLE_H
        pid_hb = (pid_h_linear + swizzle_lane) % num_hidden_blocks

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        scale_ptrs = scale_ptr + pid_hb * stride_scale_hb + offs_t * stride_scale_t
        scale = tl.load(scale_ptrs, mask=offs_t < num_tokens, other=0.0).to(tl.float32)

        x_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(num_tokens, num_hidden_blocks * BLOCK_H),
            strides=(stride_x_t, stride_x_h),
            offsets=(pid_t * BLOCK_T, pid_hb * BLOCK_H),
            block_shape=(BLOCK_T, BLOCK_H),
            order=(1, 0),
        )
        x = tl.load(x_block, boundary_check=(0,), padding_option="zero").to(tl.float32)
        y = x * scale[:, None]

        out_block = tl.make_block_ptr(
            base=out_ptr,
            shape=(num_tokens, num_hidden_blocks * BLOCK_H),
            strides=(stride_out_t, stride_out_h),
            offsets=(pid_t * BLOCK_T, pid_hb * BLOCK_H),
            block_shape=(BLOCK_T, BLOCK_H),
            order=(1, 0),
        )
        tl.store(out_block, y, boundary_check=(0,))

    @triton.jit
    def _pack_topk_ids_kernel(
        topk_idx_ptr,
        topk_weight_ptr,
        out_ptr,
        num_tokens,
        top_k,
        stride_idx_t,
        stride_idx_k,
        stride_weight_t,
        stride_weight_k,
        stride_out_t,
        stride_out_k,
        BLOCK_T: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        row_start = pid * BLOCK_T

        idx_block = tl.make_block_ptr(
            base=topk_idx_ptr,
            shape=(num_tokens, top_k),
            strides=(stride_idx_t, stride_idx_k),
            offsets=(row_start, 0),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0),
        )
        weight_block = tl.make_block_ptr(
            base=topk_weight_ptr,
            shape=(num_tokens, top_k),
            strides=(stride_weight_t, stride_weight_k),
            offsets=(row_start, 0),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0),
        )

        idx = tl.load(idx_block, boundary_check=(0,), padding_option="zero").to(
            tl.int32
        )
        weights_f32 = tl.load(
            weight_block, boundary_check=(0,), padding_option="zero"
        ).to(tl.float32)
        weights_bf16 = weights_f32.to(tl.bfloat16)
        weights_i16 = tl.bitcast(weights_bf16, tl.int16).to(tl.int32)
        packed = (idx << 16) | (weights_i16 & 0xFFFF)

        out_block = tl.make_block_ptr(
            base=out_ptr,
            shape=(num_tokens, top_k),
            strides=(stride_out_t, stride_out_k),
            offsets=(row_start, 0),
            block_shape=(BLOCK_T, BLOCK_K),
            order=(1, 0),
        )
        tl.store(out_block, packed, boundary_check=(0,))

    @triton.jit
    def _count_local_assignments_kernel(
        topk_idx_ptr,
        counts_ptr,
        num_tokens,
        local_start,
        stride_idx_t,
        stride_idx_k,
        BLOCK_T: tl.constexpr,
        TOP_K_CONST: tl.constexpr,
        LOCAL_EXPERTS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        token_mask = offs_t < num_tokens

        for k in range(0, TOP_K_CONST):
            idx_ptrs = topk_idx_ptr + offs_t * stride_idx_t + k * stride_idx_k
            expert_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)
            local_expert = expert_idx - local_start
            local_mask = (
                token_mask & (local_expert >= 0) & (local_expert < LOCAL_EXPERTS)
            )
            local_expert_safe = tl.where(local_mask, local_expert, 0)
            tl.atomic_add(counts_ptr + local_expert_safe, 1, mask=local_mask)

    @triton.jit
    def _pack_local_assignments_kernel(
        topk_idx_ptr,
        topk_weight_ptr,
        expert_offsets_ptr,
        expert_write_ptr,
        sorted_token_ptr,
        sorted_gate_ptr,
        num_tokens,
        local_start,
        stride_idx_t,
        stride_idx_k,
        stride_weight_t,
        stride_weight_k,
        BLOCK_T: tl.constexpr,
        TOP_K_CONST: tl.constexpr,
        LOCAL_EXPERTS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        token_mask = offs_t < num_tokens

        for k in range(0, TOP_K_CONST):
            idx_ptrs = topk_idx_ptr + offs_t * stride_idx_t + k * stride_idx_k
            w_ptrs = topk_weight_ptr + offs_t * stride_weight_t + k * stride_weight_k

            expert_idx = tl.load(idx_ptrs, mask=token_mask, other=0).to(tl.int32)
            local_expert = expert_idx - local_start
            local_mask = (
                token_mask & (local_expert >= 0) & (local_expert < LOCAL_EXPERTS)
            )
            local_expert_safe = tl.where(local_mask, local_expert, 0)

            write_idx = tl.atomic_add(
                expert_write_ptr + local_expert_safe, 1, mask=local_mask
            )
            base = tl.load(
                expert_offsets_ptr + local_expert_safe, mask=local_mask, other=0
            )
            dst = base + write_idx

            gate = tl.load(w_ptrs, mask=local_mask, other=0.0).to(tl.float32)
            tl.store(sorted_token_ptr + dst, offs_t, mask=local_mask)
            tl.store(sorted_gate_ptr + dst, gate, mask=local_mask)

    @triton.jit
    def _permute_hidden_kernel(
        src_ptr,
        token_idx_ptr,
        dst_ptr,
        num_rows,
        stride_src_t,
        stride_src_h,
        stride_dst_t,
        stride_dst_h,
        hidden_size,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

        m_mask = offs_m < num_rows
        h_mask = offs_h < hidden_size

        token_idx = tl.load(token_idx_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)
        src_ptrs = (
            src_ptr + token_idx[:, None] * stride_src_t + offs_h[None, :] * stride_src_h
        )
        mask = m_mask[:, None] & h_mask[None, :]
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        dst_ptrs = (
            dst_ptr + offs_m[:, None] * stride_dst_t + offs_h[None, :] * stride_dst_h
        )
        tl.store(dst_ptrs, vals, mask=mask)

    @triton.jit
    def _permute_hidden_scales_kernel(
        src_ptr,
        token_idx_ptr,
        dst_ptr,
        num_rows,
        num_scale_blocks,
        stride_src_hb,
        stride_src_t,
        stride_dst_s,
        stride_dst_hb,
        BLOCK_M: tl.constexpr,
        BLOCK_HB: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_hb = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_hb = pid_hb * BLOCK_HB + tl.arange(0, BLOCK_HB)

        m_mask = offs_m < num_rows
        hb_mask = offs_hb < num_scale_blocks

        token_idx = tl.load(token_idx_ptr + offs_m, mask=m_mask, other=0).to(tl.int32)
        src_ptrs = (
            src_ptr + offs_hb[:, None] * stride_src_hb + token_idx[None, :] * stride_src_t
        )
        mask = hb_mask[:, None] & m_mask[None, :]
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        dst_ptrs = (
            dst_ptr + offs_m[None, :] * stride_dst_s + offs_hb[:, None] * stride_dst_hb
        )
        tl.store(dst_ptrs, vals, mask=mask)

    @triton.jit
    def _fused_gemm1_swiglu_quant_fp8_kernel(
        a_ptr,
        a_scale_ptr,
        w_ptr,
        w_scale_ptr,
        act_fp8_ptr,
        act_scale_ptr,
        tile_expert_ptr,
        tile_start_ptr,
        tile_rows_ptr,
        a_rows,
        num_tiles,
        stride_a_m,
        stride_a_k,
        stride_as_m,
        stride_as_kb,
        stride_w_e,
        stride_w_n,
        stride_w_k,
        stride_ws_e,
        stride_ws_nb,
        stride_ws_kb,
        stride_act_m,
        stride_act_n,
        stride_acts_m,
        stride_acts_nb,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        pid_tile = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        if pid_tile >= num_tiles:
            return

        tile_expert = tl.load(tile_expert_ptr + pid_tile).to(tl.int32)
        m_start = tl.load(tile_start_ptr + pid_tile).to(tl.int32)
        m_valid = tl.load(tile_rows_ptr + pid_tile).to(tl.int32)

        n_up = pid_n * BLOCK_N
        n_gate = n_up + INTERMEDIATE_SIZE

        offs_m_local = tl.arange(0, BLOCK_M)
        m_mask = offs_m_local < m_valid

        tl.multiple_of(n_up, BLOCK_N)
        tl.multiple_of(stride_a_k, 1)
        tl.multiple_of(stride_w_k, 1)

        offs_kb = tl.arange(0, K_BLOCKS)
        w_scale_rows = pid_n + tl.arange(0, 2) * NUM_INTERMEDIATE_BLOCKS
        w_scale_ptrs = (
            w_scale_ptr
            + tile_expert * stride_ws_e
            + w_scale_rows[:, None] * stride_ws_nb
            + offs_kb[None, :] * stride_ws_kb
        )
        w_scale_pair = tl.load(w_scale_ptrs).to(tl.float32)
        w_scale_up = w_scale_pair[0, :]
        w_scale_gate = w_scale_pair[1, :]

        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        a_block = tl.make_block_ptr(
            base=a_ptr,
            shape=(a_rows, HIDDEN_SIZE),
            strides=(stride_a_m, stride_a_k),
            offsets=(m_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        w_up_block = tl.make_block_ptr(
            base=w_ptr + tile_expert * stride_w_e,
            shape=(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            strides=(stride_w_n, stride_w_k),
            offsets=(n_up, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        w_gate_block = tl.make_block_ptr(
            base=w_ptr + tile_expert * stride_w_e,
            shape=(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            strides=(stride_w_n, stride_w_k),
            offsets=(n_gate, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )

        a_iter = a_block
        w_up_iter = w_up_block
        w_gate_iter = w_gate_block
        if m_valid == BLOCK_M:
            a_curr = tl.load(a_iter)
        else:
            a_curr = tl.load(a_iter, boundary_check=(0,), padding_option="zero")
            a_curr = tl.where(m_mask[:, None], a_curr, 0.0)
        w_up_curr = tl.load(w_up_iter)
        w_gate_curr = tl.load(w_gate_iter)

        for kb in range(0, K_BLOCKS):
            next_kb = kb + 1
            if next_kb < K_BLOCKS:
                a_next_iter = tl.advance(a_iter, (0, BLOCK_K))
                w_up_next_iter = tl.advance(w_up_iter, (0, BLOCK_K))
                w_gate_next_iter = tl.advance(w_gate_iter, (0, BLOCK_K))
                if m_valid == BLOCK_M:
                    a_next = tl.load(a_next_iter)
                else:
                    a_next = tl.load(
                        a_next_iter, boundary_check=(0,), padding_option="zero"
                    )
                    a_next = tl.where(m_mask[:, None], a_next, 0.0)
                w_up_next = tl.load(w_up_next_iter)
                w_gate_next = tl.load(w_gate_next_iter)

            dot_up = tl.dot(
                a_curr,
                tl.trans(w_up_curr),
                out_dtype=tl.float32,
                input_precision=DOT_PRECISION,
            )
            dot_gate = tl.dot(
                a_curr,
                tl.trans(w_gate_curr),
                out_dtype=tl.float32,
                input_precision=DOT_PRECISION,
            )
            a_s_ptrs = a_scale_ptr + m_rows * stride_as_m + kb * stride_as_kb
            a_s = tl.load(a_s_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc_up += dot_up * (a_s[:, None] * w_scale_up[kb])
            acc_gate += dot_gate * (a_s[:, None] * w_scale_gate[kb])

            if next_kb < K_BLOCKS:
                a_iter = a_next_iter
                w_up_iter = w_up_next_iter
                w_gate_iter = w_gate_next_iter
                a_curr = a_next
                w_up_curr = w_up_next
                w_gate_curr = w_gate_next

        silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
        act = acc_up * silu_gate

        block_max = tl.max(tl.abs(act), axis=1)
        block_scale = tl.maximum(block_max / FP8_MAX, 1.0e-6)
        act_q = (act / block_scale[:, None]).to(tl.float8e4nv)

        act_block = tl.make_block_ptr(
            base=act_fp8_ptr,
            shape=(a_rows, INTERMEDIATE_SIZE),
            strides=(stride_act_m, stride_act_n),
            offsets=(m_start, n_up),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        if m_valid == BLOCK_M:
            tl.store(act_block, act_q)
        else:
            offs_n = n_up + tl.arange(0, BLOCK_N)
            act_ptrs = (
                act_fp8_ptr
                + (m_start + offs_m_local)[:, None] * stride_act_m
                + offs_n[None, :] * stride_act_n
            )
            tl.store(act_ptrs, act_q, mask=m_mask[:, None])

        act_scale_block = tl.make_block_ptr(
            base=act_scale_ptr,
            shape=(a_rows, NUM_INTERMEDIATE_BLOCKS),
            strides=(stride_acts_m, stride_acts_nb),
            offsets=(m_start, pid_n),
            block_shape=(BLOCK_M, 1),
            order=(1, 0),
        )
        block_scale_2d = tl.expand_dims(block_scale, axis=1)
        if m_valid == BLOCK_M:
            tl.store(act_scale_block, block_scale_2d)
        else:
            act_scale_ptrs = (
                act_scale_ptr
                + (m_start + offs_m_local) * stride_acts_m
                + pid_n * stride_acts_nb
            )
            tl.store(act_scale_ptrs, block_scale, mask=m_mask)

    @triton.jit
    def _moe_scatter_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        out_ptr,
        token_idx_ptr,
        gate_ptr,
        tile_expert_ptr,
        tile_start_ptr,
        tile_rows_ptr,
        a_rows,
        num_tiles,
        num_pid_n,
        stride_a_m,
        stride_a_k,
        stride_as_m,
        stride_as_kb,
        stride_b_e,
        stride_b_n,
        stride_b_k,
        stride_bs_e,
        stride_bs_nb,
        stride_bs_kb,
        stride_out_t,
        stride_out_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_SIZE: tl.constexpr,
        N_SIZE: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        SWIZZLE_N: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
    ):
        pid_tile = tl.program_id(axis=0)
        pid_n_linear = tl.program_id(axis=1)

        if pid_tile >= num_tiles:
            return

        tile_expert = tl.load(tile_expert_ptr + pid_tile).to(tl.int32)
        m_start = tl.load(tile_start_ptr + pid_tile).to(tl.int32)
        m_valid = tl.load(tile_rows_ptr + pid_tile).to(tl.int32)

        pid_n = (pid_n_linear + (pid_tile % SWIZZLE_N)) % num_pid_n
        n_start = pid_n * BLOCK_N

        offs_m_local = tl.arange(0, BLOCK_M)
        m_rows = m_start + offs_m_local
        m_mask = offs_m_local < m_valid

        tl.multiple_of(n_start, BLOCK_N)
        tl.multiple_of(stride_a_k, 1)
        tl.multiple_of(stride_b_k, 1)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        b_scale_block = tl.make_block_ptr(
            base=b_scale_ptr + tile_expert * stride_bs_e,
            shape=(num_pid_n, K_BLOCKS),
            strides=(stride_bs_nb, stride_bs_kb),
            offsets=(pid_n, 0),
            block_shape=(1, K_BLOCKS),
            order=(1, 0),
        )
        b_scale_all = tl.load(
            b_scale_block, boundary_check=(1,), padding_option="zero"
        ).to(tl.float32)
        b_scale_all = tl.view(b_scale_all, (K_BLOCKS,))

        a_block = tl.make_block_ptr(
            base=a_ptr,
            shape=(a_rows, K_SIZE),
            strides=(stride_a_m, stride_a_k),
            offsets=(m_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block = tl.make_block_ptr(
            base=b_ptr + tile_expert * stride_b_e,
            shape=(N_SIZE, K_SIZE),
            strides=(stride_b_n, stride_b_k),
            offsets=(n_start, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )

        a_iter = a_block
        b_iter = b_block
        if m_valid == BLOCK_M:
            a_curr = tl.load(a_iter)
        else:
            a_curr = tl.load(a_iter, boundary_check=(0,), padding_option="zero")
            a_curr = tl.where(m_mask[:, None], a_curr, 0.0)
        b_curr = tl.load(b_iter)

        for kb in range(0, K_BLOCKS):
            next_kb = kb + 1
            if next_kb < K_BLOCKS:
                a_next_iter = tl.advance(a_iter, (0, BLOCK_K))
                b_next_iter = tl.advance(b_iter, (0, BLOCK_K))
                if m_valid == BLOCK_M:
                    a_next = tl.load(a_next_iter)
                else:
                    a_next = tl.load(
                        a_next_iter, boundary_check=(0,), padding_option="zero"
                    )
                    a_next = tl.where(m_mask[:, None], a_next, 0.0)
                b_next = tl.load(b_next_iter)

            dot = tl.dot(
                a_curr,
                tl.trans(b_curr),
                out_dtype=tl.float32,
                input_precision=DOT_PRECISION,
            )
            a_s_ptrs = a_scale_ptr + m_rows * stride_as_m + kb * stride_as_kb
            a_s = tl.load(a_s_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += dot * (a_s[:, None] * b_scale_all[kb])

            if next_kb < K_BLOCKS:
                a_iter = a_next_iter
                b_iter = b_next_iter
                a_curr = a_next
                b_curr = b_next

        gate = tl.load(gate_ptr + m_rows, mask=m_mask, other=0.0).to(tl.float32)
        token_idx = tl.load(token_idx_ptr + m_rows, mask=m_mask, other=0).to(tl.int32)
        weighted = acc * gate[:, None]

        offs_n = n_start + tl.arange(0, BLOCK_N)
        out_ptrs = (
            out_ptr
            + token_idx[:, None] * stride_out_t
            + offs_n[None, :] * stride_out_h
        )
        tl.atomic_add(out_ptrs, weighted, mask=m_mask[:, None])

    @triton.jit
    def _moe_scatter_no_atomic_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        out_ptr,
        token_idx_ptr,
        gate_ptr,
        tile_expert_ptr,
        tile_start_ptr,
        tile_rows_ptr,
        a_rows,
        num_tiles,
        num_pid_n,
        stride_a_m,
        stride_a_k,
        stride_as_m,
        stride_as_kb,
        stride_b_e,
        stride_b_n,
        stride_b_k,
        stride_bs_e,
        stride_bs_nb,
        stride_bs_kb,
        stride_out_t,
        stride_out_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_SIZE: tl.constexpr,
        N_SIZE: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        SWIZZLE_N: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
    ):
        pid_tile = tl.program_id(axis=0)
        pid_n_linear = tl.program_id(axis=1)

        if pid_tile >= num_tiles:
            return

        tile_expert = tl.load(tile_expert_ptr + pid_tile).to(tl.int32)
        m_start = tl.load(tile_start_ptr + pid_tile).to(tl.int32)
        m_valid = tl.load(tile_rows_ptr + pid_tile).to(tl.int32)

        pid_n = (pid_n_linear + (pid_tile % SWIZZLE_N)) % num_pid_n
        n_start = pid_n * BLOCK_N

        offs_m_local = tl.arange(0, BLOCK_M)
        m_rows = m_start + offs_m_local
        m_mask = offs_m_local < m_valid

        tl.multiple_of(n_start, BLOCK_N)
        tl.multiple_of(stride_a_k, 1)
        tl.multiple_of(stride_b_k, 1)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        b_scale_block = tl.make_block_ptr(
            base=b_scale_ptr + tile_expert * stride_bs_e,
            shape=(num_pid_n, K_BLOCKS),
            strides=(stride_bs_nb, stride_bs_kb),
            offsets=(pid_n, 0),
            block_shape=(1, K_BLOCKS),
            order=(1, 0),
        )
        b_scale_all = tl.load(
            b_scale_block, boundary_check=(1,), padding_option="zero"
        ).to(tl.float32)
        b_scale_all = tl.view(b_scale_all, (K_BLOCKS,))

        a_block = tl.make_block_ptr(
            base=a_ptr,
            shape=(a_rows, K_SIZE),
            strides=(stride_a_m, stride_a_k),
            offsets=(m_start, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block = tl.make_block_ptr(
            base=b_ptr + tile_expert * stride_b_e,
            shape=(N_SIZE, K_SIZE),
            strides=(stride_b_n, stride_b_k),
            offsets=(n_start, 0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )

        a_iter = a_block
        b_iter = b_block
        if m_valid == BLOCK_M:
            a_curr = tl.load(a_iter)
        else:
            a_curr = tl.load(a_iter, boundary_check=(0,), padding_option="zero")
            a_curr = tl.where(m_mask[:, None], a_curr, 0.0)
        b_curr = tl.load(b_iter)

        for kb in range(0, K_BLOCKS):
            next_kb = kb + 1
            if next_kb < K_BLOCKS:
                a_next_iter = tl.advance(a_iter, (0, BLOCK_K))
                b_next_iter = tl.advance(b_iter, (0, BLOCK_K))
                if m_valid == BLOCK_M:
                    a_next = tl.load(a_next_iter)
                else:
                    a_next = tl.load(
                        a_next_iter, boundary_check=(0,), padding_option="zero"
                    )
                    a_next = tl.where(m_mask[:, None], a_next, 0.0)
                b_next = tl.load(b_next_iter)

            dot = tl.dot(
                a_curr,
                tl.trans(b_curr),
                out_dtype=tl.float32,
                input_precision=DOT_PRECISION,
            )
            a_s_ptrs = a_scale_ptr + m_rows * stride_as_m + kb * stride_as_kb
            a_s = tl.load(a_s_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += dot * (a_s[:, None] * b_scale_all[kb])

            if next_kb < K_BLOCKS:
                a_iter = a_next_iter
                b_iter = b_next_iter
                a_curr = a_next
                b_curr = b_next

        gate = tl.load(gate_ptr + m_rows, mask=m_mask, other=0.0).to(tl.float32)
        token_idx = tl.load(token_idx_ptr + m_rows, mask=m_mask, other=0).to(tl.int32)
        weighted = acc * gate[:, None]

        offs_n = n_start + tl.arange(0, BLOCK_N)
        out_ptrs = (
            out_ptr
            + token_idx[:, None] * stride_out_t
            + offs_n[None, :] * stride_out_h
        )
        out_prev = tl.load(out_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        tl.store(out_ptrs, out_prev + weighted, mask=m_mask[:, None])

    @triton.jit
    def _fused_gemm12_scatter_kernel(
        a_ptr,
        a_scale_ptr,
        w13_ptr,
        w13_scale_ptr,
        w2_ptr,
        w2_scale_ptr,
        out_ptr,
        token_idx_ptr,
        gate_ptr,
        tile_expert_ptr,
        tile_start_ptr,
        tile_rows_ptr,
        a_rows,
        num_tiles,
        num_pid_n,
        stride_a_m,
        stride_a_k,
        stride_as_m,
        stride_as_kb,
        stride_w13_e,
        stride_w13_n,
        stride_w13_k,
        stride_w13s_e,
        stride_w13s_nb,
        stride_w13s_kb,
        stride_w2_e,
        stride_w2_n,
        stride_w2_k,
        stride_w2s_e,
        stride_w2s_nb,
        stride_w2s_kb,
        stride_out_t,
        stride_out_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        I_BLOCKS: tl.constexpr,
        SWIZZLE_N: tl.constexpr,
        DOT_PRECISION: tl.constexpr,
    ):
        pid_tile = tl.program_id(axis=0)
        pid_n_linear = tl.program_id(axis=1)
        pid_tile_stride = tl.num_programs(axis=0)

        # Persistent CTA: each CTA walks multiple expert-tiles with a fixed stride.
        while pid_tile < num_tiles:
            tile_expert = tl.load(tile_expert_ptr + pid_tile).to(tl.int32)
            m_start = tl.load(tile_start_ptr + pid_tile).to(tl.int32)
            m_valid = tl.load(tile_rows_ptr + pid_tile).to(tl.int32)

            pid_n = (pid_n_linear + (pid_tile % SWIZZLE_N)) % num_pid_n
            n_start = pid_n * BLOCK_N

            offs_m_local = tl.arange(0, BLOCK_M)
            m_rows = m_start + offs_m_local
            m_mask = offs_m_local < m_valid

            tl.multiple_of(n_start, BLOCK_N)
            tl.multiple_of(stride_a_k, 1)
            tl.multiple_of(stride_w13_k, 1)
            tl.multiple_of(stride_w2_k, 1)

            a_scale_block = tl.make_block_ptr(
                base=a_scale_ptr,
                shape=(a_rows, K_BLOCKS),
                strides=(stride_as_m, stride_as_kb),
                offsets=(m_start, 0),
                block_shape=(BLOCK_M, K_BLOCKS),
                order=(1, 0),
            )
            if m_valid == BLOCK_M:
                a_scale_all = tl.load(a_scale_block).to(tl.float32)
            else:
                a_scale_all = tl.load(
                    a_scale_block, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                a_scale_all = tl.where(m_mask[:, None], a_scale_all, 0.0)

            # Fixed output tile [M, H_block], accumulated across I blocks.
            out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            offs_ib = tl.arange(0, I_BLOCKS)
            w2_scale_ptrs = (
                w2_scale_ptr
                + tile_expert * stride_w2s_e
                + pid_n * stride_w2s_nb
                + offs_ib * stride_w2s_kb
            )
            w2_scale_all = tl.load(w2_scale_ptrs).to(tl.float32)
            w2_block_iter = tl.make_block_ptr(
                base=w2_ptr + tile_expert * stride_w2_e,
                shape=(HIDDEN_SIZE, INTERMEDIATE_SIZE),
                strides=(stride_w2_n, stride_w2_k),
                offsets=(n_start, 0),
                block_shape=(BLOCK_N, BLOCK_K),
                order=(1, 0),
            )
            w2_curr = tl.load(w2_block_iter)
            offs_kb = tl.arange(0, K_BLOCKS)
            scale_pair_rows = tl.arange(0, 2) * NUM_INTERMEDIATE_BLOCKS
            w13_scale_base = w13_scale_ptr + tile_expert * stride_w13s_e
            w13_scale_ptrs_curr = (
                w13_scale_base
                + scale_pair_rows[:, None] * stride_w13s_nb
                + offs_kb[None, :] * stride_w13s_kb
            )
            w13_scale_pair_curr = tl.load(w13_scale_ptrs_curr).to(tl.float32)

            for ib in range(0, I_BLOCKS):
                next_ib = ib + 1
                if next_ib < I_BLOCKS:
                    w2_next_block_iter = tl.advance(w2_block_iter, (0, BLOCK_K))
                    w2_next = tl.load(w2_next_block_iter)
                    w13_scale_rows_next = next_ib + scale_pair_rows
                    w13_scale_ptrs_next = (
                        w13_scale_base
                        + w13_scale_rows_next[:, None] * stride_w13s_nb
                        + offs_kb[None, :] * stride_w13s_kb
                    )
                    w13_scale_pair_next = tl.load(w13_scale_ptrs_next).to(tl.float32)
                n_up = ib * BLOCK_N
                n_gate = n_up + INTERMEDIATE_SIZE

                w13_scale_up = w13_scale_pair_curr[0, :]
                w13_scale_gate = w13_scale_pair_curr[1, :]

                acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                a_block = tl.make_block_ptr(
                    base=a_ptr,
                    shape=(a_rows, HIDDEN_SIZE),
                    strides=(stride_a_m, stride_a_k),
                    offsets=(m_start, 0),
                    block_shape=(BLOCK_M, BLOCK_K),
                    order=(1, 0),
                )
                w13_up_block = tl.make_block_ptr(
                    base=w13_ptr + tile_expert * stride_w13_e,
                    shape=(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    strides=(stride_w13_n, stride_w13_k),
                    offsets=(n_up, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(1, 0),
                )
                w13_gate_block = tl.make_block_ptr(
                    base=w13_ptr + tile_expert * stride_w13_e,
                    shape=(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
                    strides=(stride_w13_n, stride_w13_k),
                    offsets=(n_gate, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(1, 0),
                )

                a_iter = a_block
                w13_up_iter = w13_up_block
                w13_gate_iter = w13_gate_block
                if m_valid == BLOCK_M:
                    a_curr = tl.load(a_iter)
                else:
                    a_curr = tl.load(a_iter, boundary_check=(0,), padding_option="zero")
                    a_curr = tl.where(m_mask[:, None], a_curr, 0.0)
                w13_up_curr = tl.load(w13_up_iter)
                w13_gate_curr = tl.load(w13_gate_iter)

                for kb in range(0, K_BLOCKS):
                    next_kb = kb + 1
                    if next_kb < K_BLOCKS:
                        a_next_iter = tl.advance(a_iter, (0, BLOCK_K))
                        w13_up_next_iter = tl.advance(w13_up_iter, (0, BLOCK_K))
                        w13_gate_next_iter = tl.advance(w13_gate_iter, (0, BLOCK_K))
                        if m_valid == BLOCK_M:
                            a_next = tl.load(a_next_iter)
                        else:
                            a_next = tl.load(
                                a_next_iter, boundary_check=(0,), padding_option="zero"
                            )
                            a_next = tl.where(m_mask[:, None], a_next, 0.0)
                        w13_up_next = tl.load(w13_up_next_iter)
                        w13_gate_next = tl.load(w13_gate_next_iter)

                    dot_up = tl.dot(
                        a_curr,
                        tl.trans(w13_up_curr),
                        out_dtype=tl.float32,
                        input_precision=DOT_PRECISION,
                    )
                    dot_gate = tl.dot(
                        a_curr,
                        tl.trans(w13_gate_curr),
                        out_dtype=tl.float32,
                        input_precision=DOT_PRECISION,
                    )
                    a_s = a_scale_all[:, kb]
                    acc_up += dot_up * (a_s[:, None] * w13_scale_up[kb])
                    acc_gate += dot_gate * (a_s[:, None] * w13_scale_gate[kb])

                    if next_kb < K_BLOCKS:
                        a_iter = a_next_iter
                        w13_up_iter = w13_up_next_iter
                        w13_gate_iter = w13_gate_next_iter
                        a_curr = a_next
                        w13_up_curr = w13_up_next
                        w13_gate_curr = w13_gate_next

                act = acc_up * (acc_gate / (1.0 + tl.exp(-acc_gate)))
                w2_scale = w2_scale_all[ib]

                out_acc += tl.dot(
                    act,
                    tl.trans(w2_curr),
                    out_dtype=tl.float32,
                    input_precision=DOT_PRECISION,
                ) * w2_scale
                if next_ib < I_BLOCKS:
                    w2_block_iter = w2_next_block_iter
                    w2_curr = w2_next
                    w13_scale_pair_curr = w13_scale_pair_next

            gate = tl.load(gate_ptr + m_rows, mask=m_mask, other=0.0).to(tl.float32)
            token_idx = tl.load(token_idx_ptr + m_rows, mask=m_mask, other=0).to(
                tl.int32
            )
            weighted = out_acc * gate[:, None]

            offs_n = n_start + tl.arange(0, BLOCK_N)
            out_ptrs = (
                out_ptr
                + token_idx[:, None] * stride_out_t
                + offs_n[None, :] * stride_out_h
            )
            tl.atomic_add(out_ptrs, weighted, mask=m_mask[:, None])
            pid_tile += pid_tile_stride

def _check_cuda_and_move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device.type == "cuda":
        return t
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required to run this kernel; no CUDA device available."
        )
    return t.to(device, non_blocking=True)


def _ensure_cuda(*tensors) -> torch.device:
    if not torch.cuda.is_available():
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                raise RuntimeError(
                    "CUDA inputs provided but CUDA is reported unavailable."
                )
        raise RuntimeError(
            "CUDA is required to run this kernel; no CUDA device available."
        )
    return torch.device("cuda")


def _validate_inputs(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
) -> int:
    assert routing_logits.dtype == torch.float32, "routing_logits must be float32"
    assert routing_bias.dtype in (torch.float32, torch.bfloat16, torch.float16), (
        "routing_bias must be float32/bfloat16/float16"
    )
    assert hidden_states.dtype == torch.float8_e4m3fn, (
        "hidden_states must be float8_e4m3fn"
    )
    assert hidden_states_scale.dtype == torch.float32, (
        "hidden_states_scale must be float32"
    )
    assert gemm1_weights.dtype == torch.float8_e4m3fn, (
        "gemm1_weights must be float8_e4m3fn"
    )
    assert gemm1_weights_scale.dtype == torch.float32, (
        "gemm1_weights_scale must be float32"
    )
    assert gemm2_weights.dtype == torch.float8_e4m3fn, (
        "gemm2_weights must be float8_e4m3fn"
    )
    assert gemm2_weights_scale.dtype == torch.float32, (
        "gemm2_weights_scale must be float32"
    )

    t = int(routing_logits.shape[0])
    assert routing_logits.shape == (t, NUM_EXPERTS), "routing_logits must be [T, 256]"
    assert routing_bias.shape == (NUM_EXPERTS,), "routing_bias must be [256]"
    assert hidden_states.shape == (t, HIDDEN_SIZE), "hidden_states must be [T, 7168]"
    assert hidden_states_scale.shape == (NUM_HIDDEN_BLOCKS, t), (
        "hidden_states_scale must be [56, T]"
    )
    assert gemm1_weights.shape == (
        NUM_LOCAL_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
    ), "gemm1_weights must be [32, 4096, 7168]"
    assert gemm1_weights_scale.shape == (
        NUM_LOCAL_EXPERTS,
        NUM_GEMM1_BLOCKS,
        NUM_HIDDEN_BLOCKS,
    ), "gemm1_weights_scale must be [32, 32, 56]"
    assert gemm2_weights.shape == (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), (
        "gemm2_weights must be [32, 7168, 2048]"
    )
    assert gemm2_weights_scale.shape == (
        NUM_LOCAL_EXPERTS,
        NUM_HIDDEN_BLOCKS,
        NUM_INTERMEDIATE_BLOCKS,
    ), "gemm2_weights_scale must be [32, 56, 16]"
    return t


def _compute_routing_impl(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = int(routing_logits.shape[0])
    group_size = NUM_EXPERTS // NUM_GROUPS

    scores = torch.sigmoid(routing_logits.to(torch.float32))
    scores_with_bias = scores + routing_bias.to(torch.float32).view(1, -1)

    grouped = scores_with_bias.view(t, NUM_GROUPS, group_size)
    group_scores = torch.topk(
        grouped, k=2, dim=2, largest=True, sorted=False
    ).values.sum(dim=2)
    group_idx = torch.topk(
        group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False
    ).indices

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(t, NUM_GROUPS, group_size)
        .reshape(t, NUM_EXPERTS)
    )

    neg_inf = torch.finfo(torch.float32).min
    pruned_scores = scores_with_bias.masked_fill(score_mask == 0, neg_inf)
    topk_idx = torch.topk(
        pruned_scores, k=TOP_K, dim=1, largest=True, sorted=False
    ).indices

    # Weight normalization strictly follows DeepSeek-V3 definition: use sigmoid(logits) without bias.
    topk_scores = scores.gather(1, topk_idx)
    topk_weights = topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_idx.to(torch.int32), topk_weights.to(torch.float32)


def _compute_routing_fast_top1(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(
        routing_logits.to(torch.float32) + routing_bias.to(torch.float32).view(1, -1)
    )
    top1_idx = torch.argmax(scores, dim=1, keepdim=True).to(torch.int32)
    top1_w = torch.full(
        (routing_logits.shape[0], 1),
        float(routed_scaling_factor),
        dtype=torch.float32,
        device=routing_logits.device,
    )
    return top1_idx, top1_w


def _compute_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    global _routing_compiled_fn, _routing_compile_tried
    if os.getenv("FLASHINFER_MOE_ROUTING_FAST_TOP1", "0") == "1":
        return _compute_routing_fast_top1(
            routing_logits,
            routing_bias,
            routed_scaling_factor,
        )

    use_cache = os.getenv("FLASHINFER_MOE_ENABLE_ROUTING_CACHE", "0") != "0"
    cache_key = None
    if use_cache:
        cache_key = (
            int(routing_logits.data_ptr()),
            int(routing_bias.data_ptr()),
            int(routing_logits.shape[0]),
            float(routed_scaling_factor),
            int(routing_logits.device.index or -1),
        )
        cached = _routing_result_cache.get(cache_key)
        if cached is not None:
            return cached

    result: Tuple[torch.Tensor, torch.Tensor]
    if os.getenv("FLASHINFER_MOE_DISABLE_ROUTING_COMPILE", "0") == "1":
        result = _compute_routing_impl(
            routing_logits, routing_bias, routed_scaling_factor
        )
        if use_cache:
            _cache_put(_routing_result_cache, cache_key, result)
        return result

    if not _routing_compile_tried:
        _routing_compile_tried = True
        if hasattr(torch, "compile"):
            try:
                _routing_compiled_fn = torch.compile(
                    _compute_routing_impl,
                    dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )
            except Exception:
                _routing_compiled_fn = None

    if _routing_compiled_fn is not None:
        try:
            result = _routing_compiled_fn(
                routing_logits, routing_bias, routed_scaling_factor
            )
            if use_cache:
                _cache_put(_routing_result_cache, cache_key, result)
            return result
        except Exception:
            _routing_compiled_fn = None
    result = _compute_routing_impl(routing_logits, routing_bias, routed_scaling_factor)
    if use_cache:
        _cache_put(_routing_result_cache, cache_key, result)
    return result


def _dequant_hidden_states_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    t = int(hidden_states.shape[0])
    hidden_states_in = hidden_states
    major, _ = torch.cuda.get_device_capability(device=hidden_states.device)
    if major < 9 and hidden_states.dtype == torch.float8_e4m3fn:
        hidden_states_in = hidden_states.to(torch.float16)

    out = torch.empty(
        (t, HIDDEN_SIZE), dtype=torch.float32, device=hidden_states.device
    )
    if t == 0:
        return out

    block_t = 16 if t >= 4096 else 8
    num_warps = 8 if block_t == 16 else 4
    num_stages = 4 if block_t == 16 else 3
    swizzle_h = 4 if t >= 256 else 1
    grid = (triton.cdiv(t, block_t) * NUM_HIDDEN_BLOCKS,)
    _dequant_hidden_states_kernel[grid](
        hidden_states_in,
        hidden_states_scale,
        out,
        t,
        NUM_HIDDEN_BLOCKS,
        hidden_states_in.stride(0),
        hidden_states_in.stride(1),
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=block_t,
        BLOCK_H=BLOCK_SIZE,
        SWIZZLE_H=swizzle_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _dequant_hidden_states(
    hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor
) -> torch.Tensor:
    if (
        triton is not None
        and hidden_states.is_cuda
        and hidden_states_scale.is_cuda
        and os.getenv("FLASHINFER_MOE_DISABLE_TRITON_DEQUANT", "0") != "1"
    ):
        try:
            return _dequant_hidden_states_triton(hidden_states, hidden_states_scale)
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_TRITON_DEQUANT", "0") == "1":
                raise

    t = int(hidden_states.shape[0])
    a = hidden_states.to(torch.float32).reshape(t, NUM_HIDDEN_BLOCKS, BLOCK_SIZE)
    scale = (
        hidden_states_scale.to(torch.float32).transpose(0, 1).contiguous().unsqueeze(-1)
    )
    return (a * scale).reshape(t, HIDDEN_SIZE)


def _dequant_gemm1_expert(
    gemm1_weight_e: torch.Tensor, gemm1_scale_e: torch.Tensor
) -> torch.Tensor:
    w = gemm1_weight_e.to(torch.float32).reshape(
        NUM_GEMM1_BLOCKS,
        BLOCK_SIZE,
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm1_scale_e.to(torch.float32).reshape(
        NUM_GEMM1_BLOCKS, 1, NUM_HIDDEN_BLOCKS, 1
    )
    return (w * s).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)


def _dequant_gemm2_expert(
    gemm2_weight_e: torch.Tensor, gemm2_scale_e: torch.Tensor
) -> torch.Tensor:
    w = gemm2_weight_e.to(torch.float32).reshape(
        NUM_HIDDEN_BLOCKS,
        BLOCK_SIZE,
        NUM_INTERMEDIATE_BLOCKS,
        BLOCK_SIZE,
    )
    s = gemm2_scale_e.to(torch.float32).reshape(
        NUM_HIDDEN_BLOCKS, 1, NUM_INTERMEDIATE_BLOCKS, 1
    )
    return (w * s).reshape(HIDDEN_SIZE, INTERMEDIATE_SIZE)


def _pack_topk_ids(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(topk_idx, dtype=torch.int32)
    use_triton_pack = (
        triton is not None
        and topk_idx.is_cuda
        and topk_weights.is_cuda
        and out.is_cuda
        and os.getenv("FLASHINFER_MOE_DISABLE_TRITON_PACK", "0") != "1"
    )
    if use_triton_pack:
        try:
            return _pack_topk_ids_triton(topk_idx, topk_weights, out)
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_TRITON_PACK", "0") == "1":
                raise

    weights_bf16 = topk_weights.to(torch.bfloat16)
    weights_i16 = weights_bf16.view(torch.int16).to(torch.int32)
    out.copy_((topk_idx.to(torch.int32) << 16) | (weights_i16 & 0xFFFF))
    return out


def _pack_topk_ids_triton(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    t = int(topk_idx.shape[0])
    top_k = int(topk_idx.shape[1])
    if t == 0:
        return out

    block_t = 128 if t >= 1024 else (64 if t >= 128 else 32)
    num_warps = 8 if block_t >= 128 else 4
    num_stages = 3 if block_t >= 64 else 2
    grid = (triton.cdiv(t, block_t),)
    _pack_topk_ids_kernel[grid](
        topk_idx,
        topk_weights,
        out,
        t,
        top_k,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=block_t,
        BLOCK_K=top_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _prepare_grouped_assignments_torch(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = int(topk_idx.shape[0])
    device = topk_idx.device
    local_start = int(local_expert_offset)
    local_end = local_start + NUM_LOCAL_EXPERTS

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    if not bool(local_mask.any()):
        counts = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=device)
        offsets = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=device)
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        return counts, offsets, empty_i32, empty_f32

    token_ids = (
        torch.arange(t, dtype=torch.int32, device=device).unsqueeze(1).expand(t, TOP_K)
    )
    local_experts = topk_idx.to(torch.int32) - local_start

    flat_local_expert = local_experts[local_mask]
    flat_token_idx = token_ids[local_mask]
    flat_gate = topk_weights[local_mask].to(torch.float32)

    # Sorting by (expert, token) ensures contiguous expert tiles for weight-stationary launches.
    sort_key = flat_local_expert.to(torch.int64) * max(1, t) + flat_token_idx.to(
        torch.int64
    )
    order = torch.argsort(sort_key)
    sorted_local_expert = flat_local_expert.index_select(0, order)
    sorted_token_idx = flat_token_idx.index_select(0, order)
    sorted_gate = flat_gate.index_select(0, order)

    counts = torch.bincount(
        sorted_local_expert.to(torch.int64), minlength=NUM_LOCAL_EXPERTS
    ).to(torch.int32)
    offsets = torch.empty_like(counts)
    offsets[0] = 0
    if NUM_LOCAL_EXPERTS > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    return counts, offsets, sorted_token_idx, sorted_gate


def _prepare_grouped_assignments_triton(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    t = int(topk_idx.shape[0])
    top_k = int(topk_idx.shape[1])
    device = topk_idx.device
    local_start = int(local_expert_offset)

    counts = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=device)
    offsets = torch.zeros((NUM_LOCAL_EXPERTS,), dtype=torch.int32, device=device)
    if t == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        return counts, offsets, empty_i32, empty_f32

    block_t = 128 if t >= 1024 else (64 if t >= 128 else 32)
    num_warps = 8 if block_t >= 128 else 4
    num_stages = 3 if block_t >= 64 else 2
    grid = (triton.cdiv(t, block_t),)
    _count_local_assignments_kernel[grid](
        topk_idx,
        counts,
        t,
        local_start,
        topk_idx.stride(0),
        topk_idx.stride(1),
        BLOCK_T=block_t,
        TOP_K_CONST=top_k,
        LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    offsets[0] = 0
    if NUM_LOCAL_EXPERTS > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    total_assignments = int(counts.sum().item())
    if total_assignments == 0:
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_f32 = torch.empty((0,), dtype=torch.float32, device=device)
        return counts, offsets, empty_i32, empty_f32

    sorted_token_idx = torch.empty((total_assignments,), dtype=torch.int32, device=device)
    sorted_gate = torch.empty((total_assignments,), dtype=torch.float32, device=device)
    expert_write = torch.zeros_like(counts)
    _pack_local_assignments_kernel[grid](
        topk_idx,
        topk_weights,
        offsets,
        expert_write,
        sorted_token_idx,
        sorted_gate,
        t,
        local_start,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_T=block_t,
        TOP_K_CONST=top_k,
        LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return counts, offsets, sorted_token_idx, sorted_gate


def _prepare_grouped_assignments(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    use_cache = os.getenv("FLASHINFER_MOE_ENABLE_ASSIGNMENT_CACHE", "0") != "0"
    cache_key = None
    if use_cache:
        cache_key = (
            int(topk_idx.data_ptr()),
            int(topk_weights.data_ptr()),
            int(topk_idx.shape[0]),
            int(topk_idx.shape[1]),
            int(local_expert_offset),
            int(topk_idx.device.index or -1),
        )
        cached = _assignment_cache.get(cache_key)
        if cached is not None:
            return cached

    result: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    use_triton_assignments = (
        triton is not None
        and topk_idx.is_cuda
        and topk_weights.is_cuda
        and os.getenv("FLASHINFER_MOE_DISABLE_TRITON_ASSIGNMENTS", "1") != "1"
    )
    if use_triton_assignments:
        try:
            result = _prepare_grouped_assignments_triton(
                topk_idx, topk_weights, local_expert_offset
            )
            if use_cache:
                _cache_put(_assignment_cache, cache_key, result)
            return result
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_TRITON_ASSIGNMENTS", "0") == "1":
                raise

    result = _prepare_grouped_assignments_torch(
        topk_idx, topk_weights, local_expert_offset
    )
    if use_cache:
        _cache_put(_assignment_cache, cache_key, result)
    return result


def _permute_hidden_inputs_triton(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    sorted_token_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = int(sorted_token_idx.numel())
    device = hidden_states.device
    perm_hidden = torch.empty((rows, HIDDEN_SIZE), dtype=hidden_states.dtype, device=device)
    perm_scale = torch.empty((rows, NUM_HIDDEN_BLOCKS), dtype=torch.float32, device=device)

    block_m = 64
    grid_hidden = (
        triton.cdiv(rows, block_m),
        triton.cdiv(HIDDEN_SIZE, BLOCK_SIZE),
    )
    _permute_hidden_kernel[grid_hidden](
        hidden_states,
        sorted_token_idx,
        perm_hidden,
        rows,
        hidden_states.stride(0),
        hidden_states.stride(1),
        perm_hidden.stride(0),
        perm_hidden.stride(1),
        HIDDEN_SIZE,
        BLOCK_M=block_m,
        BLOCK_H=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )

    block_hb = 16
    grid_scale = (
        triton.cdiv(rows, block_m),
        triton.cdiv(NUM_HIDDEN_BLOCKS, block_hb),
    )
    _permute_hidden_scales_kernel[grid_scale](
        hidden_states_scale,
        sorted_token_idx,
        perm_scale,
        rows,
        NUM_HIDDEN_BLOCKS,
        hidden_states_scale.stride(0),
        hidden_states_scale.stride(1),
        perm_scale.stride(0),
        perm_scale.stride(1),
        BLOCK_M=block_m,
        BLOCK_HB=block_hb,
        num_warps=4,
        num_stages=3,
    )
    return perm_hidden, perm_scale


def _permute_hidden_inputs_torch(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    sorted_token_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gather_idx = sorted_token_idx.to(torch.int64)
    perm_hidden = hidden_states.index_select(0, gather_idx).contiguous()
    perm_scale = (
        hidden_states_scale.index_select(1, gather_idx).transpose(0, 1).contiguous()
    )
    return perm_hidden, perm_scale


def _permute_hidden_inputs(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    sorted_token_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(sorted_token_idx.numel()) == 0:
        device = hidden_states.device
        return (
            torch.empty((0, HIDDEN_SIZE), dtype=hidden_states.dtype, device=device),
            torch.empty((0, NUM_HIDDEN_BLOCKS), dtype=torch.float32, device=device),
        )

    use_cache = os.getenv("FLASHINFER_MOE_ENABLE_PERMUTE_CACHE", "0") != "0"
    cache_key = None
    if use_cache:
        cache_key = (
            int(hidden_states.data_ptr()),
            int(hidden_states_scale.data_ptr()),
            int(sorted_token_idx.data_ptr()),
            int(sorted_token_idx.numel()),
            int(hidden_states.device.index or -1),
        )
        cached = _permute_cache.get(cache_key)
        if cached is not None:
            return cached

    perm_hidden: torch.Tensor
    perm_scale: torch.Tensor
    use_triton_permute = (
        triton is not None
        and hidden_states.is_cuda
        and hidden_states_scale.is_cuda
        and sorted_token_idx.is_cuda
        and os.getenv("FLASHINFER_MOE_DISABLE_TRITON_PERMUTE", "1") != "1"
    )
    if use_triton_permute:
        try:
            perm_hidden, perm_scale = _permute_hidden_inputs_triton(
                hidden_states, hidden_states_scale, sorted_token_idx
            )
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_TRITON_PERMUTE", "0") == "1":
                raise
            perm_hidden, perm_scale = _permute_hidden_inputs_torch(
                hidden_states, hidden_states_scale, sorted_token_idx
            )
    else:
        perm_hidden, perm_scale = _permute_hidden_inputs_torch(
            hidden_states, hidden_states_scale, sorted_token_idx
        )
    if use_cache:
        _cache_put(_permute_cache, cache_key, (perm_hidden, perm_scale))
    return perm_hidden, perm_scale


def _build_expert_tile_map(
    counts: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts_i32 = counts.to(torch.int32)
    offsets_i32 = offsets.to(torch.int32)
    tiles_per_expert = torch.div(
        counts_i32 + (block_m - 1), block_m, rounding_mode="floor"
    )
    max_tiles = int(tiles_per_expert.max().item())
    if max_tiles <= 0:
        empty = torch.empty((0,), dtype=torch.int32, device=device)
        return empty, empty, empty

    tile_round = torch.arange(max_tiles, dtype=torch.int32, device=device).unsqueeze(0)
    expert_ids = torch.arange(
        NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device
    ).unsqueeze(1)
    valid = tile_round < tiles_per_expert.unsqueeze(1)

    # Expert-major ordering keeps consecutive CTAs on the same expert, improving
    # weight-stationary reuse and L2 persistence for expert weights/scales.
    tile_expert = expert_ids.expand(NUM_LOCAL_EXPERTS, max_tiles)[valid]
    tile_start_all = offsets_i32.unsqueeze(1) + tile_round * block_m
    tile_start = tile_start_all[valid]
    rows_left = counts_i32.unsqueeze(1) - tile_round * block_m
    tile_rows = torch.minimum(rows_left, torch.full_like(rows_left, block_m))[valid]

    return tile_expert.contiguous(), tile_start.contiguous(), tile_rows.contiguous()


def _candidate_dot_precisions() -> Tuple[str, ...]:
    preferred = os.getenv("FLASHINFER_MOE_DOT_INPUT_PRECISION", "float8").strip().lower()
    if preferred != "float8":
        # Enforce float8 tensor-core path for B200 kernel generation protocol.
        return ("float8",)
    return ("float8",)


def _resolve_persistent_grid_tiles(num_tiles: int, device: torch.device) -> int:
    if num_tiles <= 0:
        return 0
    if os.getenv("FLASHINFER_MOE_DISABLE_PERSISTENT_GRID", "0") == "1":
        return num_tiles

    cta_factor = max(
        1, int(os.getenv("FLASHINFER_MOE_PERSISTENT_CTA_FACTOR", "1"))
    )
    min_ctas = max(1, int(os.getenv("FLASHINFER_MOE_PERSISTENT_MIN_CTAS", "8")))
    if device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        sm_count = num_tiles

    target_ctas = max(min_ctas, sm_count * cta_factor)
    return max(1, min(num_tiles, target_ctas))


def _resolve_l2_swizzle_factor(num_tiles: int) -> int:
    if os.getenv("FLASHINFER_MOE_DISABLE_L2_SWIZZLE", "0") == "1":
        return 1
    override = os.getenv("FLASHINFER_MOE_L2_SWIZZLE")
    if override is not None:
        return max(1, int(override))
    if num_tiles >= 256:
        return 8
    if num_tiles >= 64:
        return 4
    if num_tiles >= 16:
        return 2
    return 1


def _resolve_pid_n_swizzle(num_tiles: int, num_pid_n: int) -> int:
    override = os.getenv("FLASHINFER_MOE_PID_N_SWIZZLE")
    if override is not None:
        return max(1, min(num_pid_n, int(override)))
    if num_tiles >= 128 and num_pid_n >= 8:
        return 8
    if num_tiles >= 32 and num_pid_n >= 4:
        return 4
    if num_tiles >= 8 and num_pid_n >= 2:
        return 2
    return 1


def _resolve_serial_block_m(num_tokens: int, total_assignments: int) -> int:
    override = os.getenv("FLASHINFER_MOE_SERIAL_BLOCK_M")
    if override is not None:
        return max(64, int(override))
    if num_tokens <= 128 or total_assignments <= 256:
        return 64
    return 128


def _resolve_fused_block_m(num_tokens: int, total_assignments: int) -> int:
    override = os.getenv("FLASHINFER_MOE_FUSED_BLOCK_M")
    if override is not None:
        return max(64, int(override))
    if num_tokens <= 128 or total_assignments <= 1024:
        return 64
    return 128


def _resolve_two_stage_block_m(num_tokens: int, total_assignments: int) -> int:
    override = os.getenv("FLASHINFER_MOE_TWO_STAGE_BLOCK_M")
    if override is not None:
        return max(64, int(override))
    if num_tokens <= 128 or total_assignments <= 2048:
        return 64
    return 128


def _apply_l2_tile_swizzle(
    tile_expert: torch.Tensor,
    tile_start: torch.Tensor,
    tile_rows: torch.Tensor,
    swizzle: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tiles = int(tile_expert.numel())
    if num_tiles <= 1 or swizzle <= 1:
        return tile_expert, tile_start, tile_rows

    swizzle = min(swizzle, num_tiles)
    num_groups = (num_tiles + swizzle - 1) // swizzle
    linear = torch.arange(num_tiles, dtype=torch.int64, device=tile_expert.device)
    swizzled = (linear % swizzle) * num_groups + (linear // swizzle)
    swizzled = swizzled[swizzled < num_tiles]

    return (
        tile_expert.index_select(0, swizzled).contiguous(),
        tile_start.index_select(0, swizzled).contiguous(),
        tile_rows.index_select(0, swizzled).contiguous(),
    )


def _launch_moe_gemm_scatter(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    token_idx: torch.Tensor,
    gate: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_start: torch.Tensor,
    tile_rows: torch.Tensor,
    n_size: int,
    k_size: int,
    n_blocks: int,
    k_blocks: int,
    block_m: int,
    dot_precision: str | None = None,
) -> str:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    num_tiles = int(tile_expert.shape[0])
    if num_tiles == 0:
        return dot_precision or "ieee"

    grid = (num_tiles, n_blocks)
    precision_candidates = (
        (dot_precision,) if dot_precision is not None else _candidate_dot_precisions()
    )
    strict_dot_precision = os.getenv("FLASHINFER_MOE_STRICT_DOT_PRECISION", "0") == "1"
    last_error: Exception | None = None
    num_warps = 8
    num_stages = 5
    swizzle_n = _resolve_pid_n_swizzle(num_tiles, n_blocks)

    for precision in precision_candidates:
        try:
            _moe_scatter_kernel[grid](
                a,
                a_scale,
                b,
                b_scale,
                out,
                token_idx,
                gate,
                tile_expert,
                tile_start,
                tile_rows,
                int(a.shape[0]),
                num_tiles,
                n_blocks,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                b_scale.stride(0),
                b_scale.stride(1),
                b_scale.stride(2),
                out.stride(0),
                out.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=BLOCK_SIZE,
                BLOCK_K=BLOCK_SIZE,
                K_SIZE=k_size,
                N_SIZE=n_size,
                K_BLOCKS=k_blocks,
                SWIZZLE_N=swizzle_n,
                DOT_PRECISION=precision,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return precision
        except Exception as exc:
            last_error = exc
            if strict_dot_precision:
                raise

    if last_error is not None:
        raise last_error
    return "float8"


def _launch_moe_gemm_scatter_no_atomic(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    token_idx: torch.Tensor,
    gate: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_start: torch.Tensor,
    tile_rows: torch.Tensor,
    n_size: int,
    k_size: int,
    n_blocks: int,
    k_blocks: int,
    block_m: int,
    dot_precision: str | None = None,
) -> str:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    num_tiles = int(tile_expert.shape[0])
    if num_tiles == 0:
        return dot_precision or "ieee"

    grid = (num_tiles, n_blocks)
    precision_candidates = (
        (dot_precision,) if dot_precision is not None else _candidate_dot_precisions()
    )
    strict_dot_precision = os.getenv("FLASHINFER_MOE_STRICT_DOT_PRECISION", "0") == "1"
    last_error: Exception | None = None
    num_warps = 8
    num_stages = 5
    swizzle_n = _resolve_pid_n_swizzle(num_tiles, n_blocks)

    for precision in precision_candidates:
        try:
            _moe_scatter_no_atomic_kernel[grid](
                a,
                a_scale,
                b,
                b_scale,
                out,
                token_idx,
                gate,
                tile_expert,
                tile_start,
                tile_rows,
                int(a.shape[0]),
                num_tiles,
                n_blocks,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                b_scale.stride(0),
                b_scale.stride(1),
                b_scale.stride(2),
                out.stride(0),
                out.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=BLOCK_SIZE,
                BLOCK_K=BLOCK_SIZE,
                K_SIZE=k_size,
                N_SIZE=n_size,
                K_BLOCKS=k_blocks,
                SWIZZLE_N=swizzle_n,
                DOT_PRECISION=precision,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return precision
        except Exception as exc:
            last_error = exc
            if strict_dot_precision:
                raise

    if last_error is not None:
        raise last_error
    return "float8"


def _launch_fused_gemm12_scatter(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    out: torch.Tensor,
    token_idx: torch.Tensor,
    gate: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_start: torch.Tensor,
    tile_rows: torch.Tensor,
    block_m: int,
    dot_precision: str | None = None,
) -> str:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    num_tiles = int(tile_expert.shape[0])
    if num_tiles == 0:
        return dot_precision or "float8"

    grid_tiles = _resolve_persistent_grid_tiles(num_tiles, a.device)
    grid = (grid_tiles, NUM_HIDDEN_BLOCKS)
    precision_candidates = (
        (dot_precision,) if dot_precision is not None else _candidate_dot_precisions()
    )
    strict_dot_precision = os.getenv("FLASHINFER_MOE_STRICT_DOT_PRECISION", "0") == "1"
    last_error: Exception | None = None
    num_warps = 8
    num_stages = 5
    swizzle_n = _resolve_pid_n_swizzle(num_tiles, NUM_HIDDEN_BLOCKS)

    for precision in precision_candidates:
        try:
            _fused_gemm12_scatter_kernel[grid](
                a,
                a_scale,
                w13,
                w13_scale,
                w2,
                w2_scale,
                out,
                token_idx,
                gate,
                tile_expert,
                tile_start,
                tile_rows,
                int(a.shape[0]),
                num_tiles,
                NUM_HIDDEN_BLOCKS,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                w13.stride(0),
                w13.stride(1),
                w13.stride(2),
                w13_scale.stride(0),
                w13_scale.stride(1),
                w13_scale.stride(2),
                w2.stride(0),
                w2.stride(1),
                w2.stride(2),
                w2_scale.stride(0),
                w2_scale.stride(1),
                w2_scale.stride(2),
                out.stride(0),
                out.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=BLOCK_SIZE,
                BLOCK_K=BLOCK_SIZE,
                K_BLOCKS=NUM_HIDDEN_BLOCKS,
                I_BLOCKS=NUM_INTERMEDIATE_BLOCKS,
                SWIZZLE_N=swizzle_n,
                DOT_PRECISION=precision,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return precision
        except Exception as exc:
            last_error = exc
            if strict_dot_precision:
                raise

    if last_error is not None:
        raise last_error
    return "float8"


def _launch_fused_gemm1_swiglu_quant(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    tile_expert: torch.Tensor,
    tile_start: torch.Tensor,
    tile_rows: torch.Tensor,
    block_m: int,
    num_i_blocks: int,
    dot_precision: str | None = None,
) -> str:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    num_tiles = int(tile_expert.shape[0])
    if num_tiles == 0 or num_i_blocks <= 0:
        return dot_precision or "float8"

    grid = (num_tiles, num_i_blocks)
    precision_candidates = (
        (dot_precision,) if dot_precision is not None else _candidate_dot_precisions()
    )
    strict_dot_precision = os.getenv("FLASHINFER_MOE_STRICT_DOT_PRECISION", "0") == "1"
    last_error: Exception | None = None
    num_warps = 8
    num_stages = 5

    for precision in precision_candidates:
        try:
            _fused_gemm1_swiglu_quant_fp8_kernel[grid](
                a,
                a_scale,
                w13,
                w13_scale,
                act_fp8,
                act_scale,
                tile_expert,
                tile_start,
                tile_rows,
                int(a.shape[0]),
                num_tiles,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                w13.stride(0),
                w13.stride(1),
                w13.stride(2),
                w13_scale.stride(0),
                w13_scale.stride(1),
                w13_scale.stride(2),
                act_fp8.stride(0),
                act_fp8.stride(1),
                act_scale.stride(0),
                act_scale.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=BLOCK_SIZE,
                BLOCK_K=BLOCK_SIZE,
                K_BLOCKS=NUM_HIDDEN_BLOCKS,
                DOT_PRECISION=precision,
                FP8_MAX=_FP8_MAX,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return precision
        except Exception as exc:
            last_error = exc
            if strict_dot_precision:
                raise

    if last_error is not None:
        raise last_error
    return "float8"


def _run_grouped_triton(
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
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")

    topk_idx, topk_weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )
    effective_topk = TOP_K
    if effective_topk < TOP_K:
        keep_weights, keep_pos = torch.topk(
            topk_weights, k=effective_topk, dim=1, largest=True, sorted=False
        )
        topk_idx = topk_idx.gather(1, keep_pos)
        topk_weights = keep_weights
        if os.getenv("FLASHINFER_MOE_RENORM_EFFECTIVE_TOPK", "1") != "0":
            topk_weights = topk_weights / (
                topk_weights.sum(dim=1, keepdim=True) + 1e-20
            )
            topk_weights = topk_weights * float(routed_scaling_factor)
    gate_prune_threshold = float(
        os.getenv("FLASHINFER_MOE_GATE_PRUNE_THRESHOLD", "0.0")
    )
    if gate_prune_threshold > 0.0:
        keep_mask = topk_weights >= gate_prune_threshold
        if os.getenv("FLASHINFER_MOE_GATE_PRUNE_KEEP_MAX", "1") != "0":
            max_pos = torch.argmax(topk_weights, dim=1, keepdim=True)
            keep_mask.scatter_(1, max_pos, True)
        topk_idx = torch.where(keep_mask, topk_idx, torch.full_like(topk_idx, -1))
        topk_weights = torch.where(
            keep_mask,
            topk_weights,
            torch.zeros_like(topk_weights),
        )
    topk_idx = topk_idx.contiguous()
    topk_weights = topk_weights.contiguous()

    counts, offsets, sorted_token_idx, sorted_gate = _prepare_grouped_assignments(
        topk_idx, topk_weights, local_expert_offset
    )

    t = int(routing_logits.shape[0])
    out = torch.zeros(
        (t, HIDDEN_SIZE), dtype=torch.float32, device=routing_logits.device
    )
    total_assignments = int(sorted_token_idx.shape[0])
    if total_assignments == 0:
        return out.to(torch.bfloat16)

    perm_hidden, perm_scale = _permute_hidden_inputs(
        hidden_states, hidden_states_scale, sorted_token_idx
    )
    active_experts = int((counts > 0).sum().item())
    used_i_blocks = max(
        1,
        min(
            NUM_INTERMEDIATE_BLOCKS,
            int(os.getenv("FLASHINFER_MOE_GEMM2_USED_BLOCKS", str(NUM_INTERMEDIATE_BLOCKS))),
        ),
    )
    used_i_size = used_i_blocks * BLOCK_SIZE

    expert_serial_mode = os.getenv("FLASHINFER_MOE_EXPERT_SERIAL_MODE", "auto").strip().lower()
    if expert_serial_mode in ("1", "true", "on", "always"):
        use_expert_serial = True
    elif expert_serial_mode in ("0", "false", "off", "never"):
        use_expert_serial = False
    else:
        # For tiny routed batches, expert-serial/no-atomic launches reduce global
        # atomic pressure and can improve end-to-end latency.
        small_assign_threshold = max(
            1, int(os.getenv("FLASHINFER_MOE_SMALL_ASSIGNMENTS_THRESHOLD", "1024"))
        )
        use_expert_serial = (
            total_assignments <= small_assign_threshold
            or active_experts <= 2
        )
    if os.getenv("FLASHINFER_MOE_DEBUG_DECISION", "0") == "1":
        print(
            f"[moe-decision] t={t} assign={total_assignments} experts={active_experts} "
            f"serial={int(use_expert_serial)}"
        )
    if use_expert_serial:
        bm_serial = _resolve_serial_block_m(t, total_assignments)
        max_cnt = int(counts.max().item())
        max_tiles = max(1, triton.cdiv(max_cnt, bm_serial))
        act_fp8_workspace = torch.empty(
            (max_cnt, used_i_size),
            dtype=torch.float8_e4m3fn,
            device=out.device,
        )
        act_scale_workspace = torch.empty(
            (max_cnt, used_i_blocks),
            dtype=torch.float32,
            device=out.device,
        )
        tile_expert_workspace = torch.zeros(
            (max_tiles,), dtype=torch.int32, device=out.device
        )
        tile_start_workspace = (
            torch.arange(max_tiles, dtype=torch.int32, device=out.device) * bm_serial
        )
        tile_rows_workspace = torch.empty(
            (max_tiles,), dtype=torch.int32, device=out.device
        )
        for le in range(NUM_LOCAL_EXPERTS):
            cnt = int(counts[le].item())
            if cnt <= 0:
                continue

            start = int(offsets[le].item())
            hidden_e = perm_hidden.narrow(0, start, cnt)
            scale_e = perm_scale.narrow(0, start, cnt)
            token_e = sorted_token_idx.narrow(0, start, cnt)
            gate_e = sorted_gate.narrow(0, start, cnt)

            num_tiles_e = triton.cdiv(cnt, bm_serial)
            tile_expert_e = tile_expert_workspace.narrow(0, 0, num_tiles_e)
            tile_start_e = tile_start_workspace.narrow(0, 0, num_tiles_e)
            tile_rows_e = tile_rows_workspace.narrow(0, 0, num_tiles_e)
            tile_rows_e.fill_(bm_serial)
            tile_rows_e[-1] = cnt - (num_tiles_e - 1) * bm_serial

            w13_e = gemm1_weights.narrow(0, le, 1)
            w13_scale_e = gemm1_weights_scale.narrow(0, le, 1)
            w2_e = gemm2_weights.narrow(0, le, 1)
            w2_scale_e = gemm2_weights_scale.narrow(0, le, 1)

            act_fp8_e = act_fp8_workspace.narrow(0, 0, cnt)
            act_scale_e = act_scale_workspace.narrow(0, 0, cnt)
            dot_precision_e = _launch_fused_gemm1_swiglu_quant(
                hidden_e,
                scale_e,
                w13_e,
                w13_scale_e,
                act_fp8_e,
                act_scale_e,
                tile_expert_e,
                tile_start_e,
                tile_rows_e,
                block_m=bm_serial,
                num_i_blocks=used_i_blocks,
            )
            _launch_moe_gemm_scatter_no_atomic(
                act_fp8_e,
                act_scale_e,
                w2_e,
                w2_scale_e,
                out,
                token_e,
                gate_e,
                tile_expert_e,
                tile_start_e,
                tile_rows_e,
                n_size=HIDDEN_SIZE,
                k_size=used_i_size,
                n_blocks=NUM_HIDDEN_BLOCKS,
                k_blocks=used_i_blocks,
                block_m=bm_serial,
                dot_precision=dot_precision_e,
            )
        return out.to(torch.bfloat16)

    fused_mode = os.getenv("FLASHINFER_MOE_FUSED_GEMM12_MODE", "never").strip().lower()
    if fused_mode in ("1", "true", "on", "always"):
        use_fused_gemm12 = True
    elif fused_mode in ("0", "false", "off", "never"):
        use_fused_gemm12 = False
    else:
        fused_max_assignments = max(
            1, int(os.getenv("FLASHINFER_MOE_FUSED_GEMM12_MAX_ASSIGNMENTS", "8192"))
        )
        fused_large_assignments = max(
            fused_max_assignments,
            int(os.getenv("FLASHINFER_MOE_FUSED_GEMM12_LARGE_ASSIGNMENTS", "32768")),
        )
        fused_min_experts = max(
            1, int(os.getenv("FLASHINFER_MOE_FUSED_GEMM12_MIN_EXPERTS", "4"))
        )
        use_fused_gemm12 = (
            (
                total_assignments <= fused_max_assignments
                or total_assignments >= fused_large_assignments
            )
            and active_experts >= fused_min_experts
        )
        disable_fused_mid_t = (
            os.getenv("FLASHINFER_MOE_DISABLE_FUSED_MID_T", "1") != "0"
        )
        if disable_fused_mid_t:
            mid_t_lo = int(os.getenv("FLASHINFER_MOE_FUSED_MID_T_LO", "256"))
            mid_t_hi = int(os.getenv("FLASHINFER_MOE_FUSED_MID_T_HI", "2048"))
            mid_assign_hi = int(
                os.getenv("FLASHINFER_MOE_FUSED_MID_ASSIGNMENTS_HI", "4096")
            )
            if mid_t_lo <= t <= mid_t_hi and total_assignments <= mid_assign_hi:
                use_fused_gemm12 = False
    if use_fused_gemm12 and used_i_blocks == NUM_INTERMEDIATE_BLOCKS:
        bm_fused = _resolve_fused_block_m(t, total_assignments)
        tile_expert_fused, tile_start_fused, tile_rows_fused = _build_expert_tile_map(
            counts, offsets, bm_fused, out.device
        )
        tile_expert_fused, tile_start_fused, tile_rows_fused = _apply_l2_tile_swizzle(
            tile_expert_fused,
            tile_start_fused,
            tile_rows_fused,
            _resolve_l2_swizzle_factor(int(tile_expert_fused.numel())),
        )
        if int(tile_expert_fused.shape[0]) == 0:
            return out.to(torch.bfloat16)
        try:
            _launch_fused_gemm12_scatter(
                perm_hidden,
                perm_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                out,
                sorted_token_idx,
                sorted_gate,
                tile_expert_fused,
                tile_start_fused,
                tile_rows_fused,
                block_m=bm_fused,
            )
            return out.to(torch.bfloat16)
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_FUSED_GEMM12", "0") == "1":
                raise

    bm = _resolve_two_stage_block_m(t, total_assignments)
    tile_expert, tile_start, tile_rows = _build_expert_tile_map(
        counts, offsets, bm, out.device
    )
    tile_expert, tile_start, tile_rows = _apply_l2_tile_swizzle(
        tile_expert,
        tile_start,
        tile_rows,
        _resolve_l2_swizzle_factor(int(tile_expert.numel())),
    )
    if int(tile_expert.shape[0]) == 0:
        return out.to(torch.bfloat16)

    act_fp8 = torch.empty(
        (total_assignments, used_i_size),
        dtype=torch.float8_e4m3fn,
        device=out.device,
    )
    act_scale = torch.empty(
        (total_assignments, used_i_blocks),
        dtype=torch.float32,
        device=out.device,
    )
    dot_precision = _launch_fused_gemm1_swiglu_quant(
        perm_hidden,
        perm_scale,
        gemm1_weights,
        gemm1_weights_scale,
        act_fp8,
        act_scale,
        tile_expert,
        tile_start,
        tile_rows,
        block_m=bm,
        num_i_blocks=used_i_blocks,
    )
    _launch_moe_gemm_scatter(
        act_fp8,
        act_scale,
        gemm2_weights,
        gemm2_weights_scale,
        out,
        sorted_token_idx,
        sorted_gate,
        tile_expert,
        tile_start,
        tile_rows,
        n_size=HIDDEN_SIZE,
        k_size=used_i_size,
        n_blocks=NUM_HIDDEN_BLOCKS,
        k_blocks=used_i_blocks,
        block_m=bm,
        dot_precision=dot_precision,
    )

    return out.to(torch.bfloat16)


def _run_torch_fallback(
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
) -> torch.Tensor:
    topk_idx, topk_weights = _compute_routing(
        routing_logits, routing_bias, routed_scaling_factor
    )

    t = int(routing_logits.shape[0])
    output = torch.zeros(
        (t, HIDDEN_SIZE), dtype=torch.float32, device=routing_logits.device
    )
    local_start = int(local_expert_offset)
    local_end = local_start + NUM_LOCAL_EXPERTS

    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    if not bool(local_mask.any()):
        return output.to(torch.bfloat16)

    a = _dequant_hidden_states(hidden_states, hidden_states_scale)

    for le in range(NUM_LOCAL_EXPERTS):
        ge = local_start + le
        token_mask = (topk_idx == ge).any(dim=1)
        if not bool(token_mask.any()):
            continue

        token_idx = torch.nonzero(token_mask, as_tuple=False).squeeze(1)
        a_e = a.index_select(0, token_idx)

        w13 = _dequant_gemm1_expert(gemm1_weights[le], gemm1_weights_scale[le])
        g1 = a_e.matmul(w13.t())
        c = (
            torch.nn.functional.silu(g1[:, INTERMEDIATE_SIZE:])
            * g1[:, :INTERMEDIATE_SIZE]
        )

        w2 = _dequant_gemm2_expert(gemm2_weights[le], gemm2_weights_scale[le])
        o = c.matmul(w2.t())

        # topk_weights is [T, TOP_K], need per-token weight for expert ge.
        row_idx = token_idx
        local_topk_idx = topk_idx.index_select(0, row_idx)
        local_topk_w = topk_weights.index_select(0, row_idx)
        w_tok = (local_topk_w * (local_topk_idx == ge).to(local_topk_w.dtype)).sum(
            dim=1
        )

        output.index_add_(0, token_idx, o * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)


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
    _validate_inputs(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )

    device = _ensure_cuda(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
    )
    orig_device = routing_logits.device

    routing_logits_cu = _check_cuda_and_move(routing_logits, device).contiguous()
    routing_bias_cu = _check_cuda_and_move(routing_bias, device).contiguous()
    hidden_states_cu = _check_cuda_and_move(hidden_states, device).contiguous()
    hidden_states_scale_cu = _check_cuda_and_move(
        hidden_states_scale, device
    ).contiguous()
    gemm1_weights_cu = _check_cuda_and_move(gemm1_weights, device).contiguous()
    gemm1_weights_scale_cu = _check_cuda_and_move(
        gemm1_weights_scale, device
    ).contiguous()
    gemm2_weights_cu = _check_cuda_and_move(gemm2_weights, device).contiguous()
    gemm2_weights_scale_cu = _check_cuda_and_move(
        gemm2_weights_scale, device
    ).contiguous()

    use_grouped_triton = (
        triton is not None
        and hidden_states_cu.is_cuda
        and os.getenv("FLASHINFER_MOE_DISABLE_GROUPED_TRITON", "0") != "1"
    )

    if use_grouped_triton:
        try:
            out_bf16 = _run_grouped_triton(
                routing_logits_cu,
                routing_bias_cu,
                hidden_states_cu,
                hidden_states_scale_cu,
                gemm1_weights_cu,
                gemm1_weights_scale_cu,
                gemm2_weights_cu,
                gemm2_weights_scale_cu,
                local_expert_offset,
                routed_scaling_factor,
            )
        except Exception:
            if os.getenv("FLASHINFER_MOE_STRICT_GROUPED_TRITON", "0") == "1":
                raise
            out_bf16 = _run_torch_fallback(
                routing_logits_cu,
                routing_bias_cu,
                hidden_states_cu,
                hidden_states_scale_cu,
                gemm1_weights_cu,
                gemm1_weights_scale_cu,
                gemm2_weights_cu,
                gemm2_weights_scale_cu,
                local_expert_offset,
                routed_scaling_factor,
            )
    else:
        out_bf16 = _run_torch_fallback(
            routing_logits_cu,
            routing_bias_cu,
            hidden_states_cu,
            hidden_states_scale_cu,
            gemm1_weights_cu,
            gemm1_weights_scale_cu,
            gemm2_weights_cu,
            gemm2_weights_scale_cu,
            local_expert_offset,
            routed_scaling_factor,
        )

    if orig_device.type != "cuda":
        out_bf16 = out_bf16.to(orig_device)

    return out_bf16