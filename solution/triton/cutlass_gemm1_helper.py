"""CUTLASS GEMM1 helper for Blackwell GPU — drop-in replacement for Triton GEMM1.

Uses the CuTe DSL BlockwiseContiguousGroupedGemmKernel with:
- Software FP32 per-row/per-block scaling (acc_update warps)
- Warp specialization (TMA + MMA + epilogue warps)
- Persistent tile scheduling
- tcgen05.mma (Blackwell native FP8 tensor cores)

Drop-in interface: replaces grouped_gemm1_large_triton.
Handles gather/scatter between MoE sorted layout and CUTLASS contiguous grouped layout.
"""

import ctypes
import os
import sys
import torch
import triton
import triton.language as tl
from typing import NamedTuple

# CUTLASS imports — nvidia-cutlass-dsl and cuda-python are in Modal image deps
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda_drv

# Import the contiguous grouped GEMM kernel from co-located module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockwise_contiguous_grouped_gemm import BlockwiseContiguousGroupedGemmKernel

try:
    from .fast_launch import launch as _launch
except ImportError:
    from fast_launch import launch as _launch


@triton.jit
def _pdl_launch_dependents():
    """Signal that dependent kernels may start (PDL prologue overlap)."""
    tl.inline_asm_elementwise(
        "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
        "=r", [], dtype=tl.int32, is_pure=False, pack=1,
    )

@triton.jit
def _pdl_wait():
    """Wait for prior kernel to complete (PDL synchronization)."""
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; mov.u32 $0, 0;",
        "=r", [], dtype=tl.int32, is_pure=False, pack=1,
    )


# ---- Triton kernel: fused gather_A + gather_SFA + build_gidx ----

@triton.jit
def fused_gather_kernel(
    hidden_states_ptr, a_scale_ptr, sorted_token_ids_ptr,
    expert_offsets_ptr, padded_offsets_ptr,
    A_gathered_ptr, SFA_gathered_ptr, gidx_mapping_ptr,
    stride_as,  # stride of a_scale dim0 (= NUM_K_BLOCKS for transposed [T, 56] layout)
    sfa_stride_m,  # stride for SFA_gathered dim0 (= valid_m for column-major)
    K: tl.constexpr, BLOCK_COPY: tl.constexpr, NUM_K_BLOCKS: tl.constexpr,
    E_LOCAL: tl.constexpr,
):
    """Fused gather: hidden_states → A_gathered, A_scale → SFA_gathered, gidx_mapping fill.
    Combines 3 kernel launches into 1. Shares offset loads and token_id lookups.
    Grid: (valid_m,) — 1D flat grid, uses vectorized padded_offsets lookup to find expert.
    BLOCK_COPY=1024: each thread handles 8 FP8 bytes, 7 iterations (vs 56 with BLOCK_K=128).
    """
    # Wait for prior kernel (routing/scatter) output
    _pdl_wait()

    flat_row = tl.program_id(0)

    # Early exit for rows beyond actual valid_m (padded_offsets[E_LOCAL] = valid_m on GPU)
    valid_m_actual = tl.load(padded_offsets_ptr + E_LOCAL)
    if flat_row >= valid_m_actual:
        return

    # Vectorized expert lookup: load all padded_offsets, count how many <= flat_row
    offs_e = tl.arange(0, 64)  # padded to power-of-2 >= E_LOCAL+1 (33)
    po_mask = offs_e <= E_LOCAL
    po = tl.load(padded_offsets_ptr + offs_e, mask=po_mask, other=2147483647)
    expert_id = tl.sum((po <= flat_row).to(tl.int32)) - 1
    expert_id = tl.maximum(expert_id, 0)

    dst_start = tl.load(padded_offsets_ptr + expert_id).to(tl.int64)
    pid_row = flat_row - dst_start

    src_start = tl.load(expert_offsets_ptr + expert_id).to(tl.int64)
    src_end = tl.load(expert_offsets_ptr + expert_id + 1).to(tl.int64)
    count = src_end - src_start

    # gidx_mapping: single int32 store per row
    tl.store(gidx_mapping_ptr + flat_row, expert_id)

    offs_k = tl.arange(0, BLOCK_COPY)
    offs_kb = tl.arange(0, 64)  # padded to power-of-2 >= NUM_K_BLOCKS (56)
    kb_mask = offs_kb < NUM_K_BLOCKS

    if pid_row < count:
        token_id = tl.load(sorted_token_ids_ptr + src_start + pid_row)
        # gather_A: copy hidden_states row (wide blocks: 7 iters of 1024 vs 56 of 128)
        for i in range(K // BLOCK_COPY):
            k_off = i * BLOCK_COPY
            data = tl.load(hidden_states_ptr + token_id * K + k_off + offs_k)
            tl.store(A_gathered_ptr + flat_row * K + k_off + offs_k, data)
        # gather_SFA: copy scale values (contiguous read from transposed [T, 56] layout)
        scale_vals = tl.load(
            a_scale_ptr + token_id * stride_as + offs_kb,
            mask=kb_mask, other=0.0
        )
    else:
        # Padding rows: zero A, zero SFA
        zero_val = tl.zeros([BLOCK_COPY], dtype=tl.float8e4nv)
        for i in range(K // BLOCK_COPY):
            k_off = i * BLOCK_COPY
            tl.store(A_gathered_ptr + flat_row * K + k_off + offs_k, zero_val)
        scale_vals = tl.zeros([64], dtype=tl.float32)

    tl.store(
        SFA_gathered_ptr + offs_kb * sfa_stride_m + flat_row,
        scale_vals, mask=kb_mask
    )

    _pdl_launch_dependents()


@triton.jit
def fused_swiglu_fp8_padded_kernel(
    C_raw_ptr,             # [1, valid_m, N=4096] fp16 — CUTLASS padded output
    swiglu_out_ptr,        # [1, valid_m, I=2048] fp8 — output in PADDED layout
    block_scale_ptr,       # [1, NUM_I_BLOCKS, valid_m] fp32 — column-major (M contiguous)
    padded_offsets_ptr,    # [E+1] int32
    valid_m_stride,        # int — stride for block_scale M dimension (= valid_m)
    I_dim: tl.constexpr,   # 2048
    N_dim: tl.constexpr,   # 4096
    NUM_I_BLOCKS: tl.constexpr,  # 16
    BLOCK_K: tl.constexpr,       # 128
    BLOCK_M: tl.constexpr,       # rows per thread block (e.g., 4)
    E_LOCAL: tl.constexpr,       # number of local experts
):
    """SwiGLU → FP8 in PADDED layout for CUTLASS FP8 GEMM2.

    Single-pass: per-block scaling means we scale+write each block independently.
    Output stays in padded layout (same row indices as C_raw input).
    Block scales stored column-major: scale[kb, m] = ptr + kb * valid_m_stride + m
    Grid: (cdiv(valid_m, BLOCK_M),)
    """
    _pdl_wait()

    flat_base = tl.program_id(0) * BLOCK_M

    valid_m_actual = tl.load(padded_offsets_ptr + E_LOCAL)
    if flat_base >= valid_m_actual:
        _pdl_launch_dependents()
        return

    # Vectorized expert lookup to check if these rows are padding
    offs_e = tl.arange(0, 64)
    po_mask = offs_e <= E_LOCAL
    po = tl.load(padded_offsets_ptr + offs_e, mask=po_mask, other=2147483647)
    expert_id = tl.sum((po <= flat_base).to(tl.int32)) - 1
    expert_id = tl.maximum(expert_id, 0)

    # Check expert bounds — but for padded layout, padding rows get zero
    # (GEMM1 output is zero for padding rows, so SwiGLU(0,0)=0, scale=1)
    # No need to skip — zeros produce zeros.

    offs_m = tl.arange(0, BLOCK_M)
    row_mask = (flat_base + offs_m) < valid_m_actual
    src_rows = flat_base + offs_m  # padded layout positions

    offs_k = tl.arange(0, BLOCK_K)

    # Single-pass: compute SwiGLU, per-block scale to FP8
    for kb in range(NUM_I_BLOCKS):
        k_off = kb * BLOCK_K
        x1 = tl.load(
            C_raw_ptr + src_rows[:, None] * N_dim + k_off + offs_k[None, :],
            mask=row_mask[:, None], other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            C_raw_ptr + src_rows[:, None] * N_dim + I_dim + k_off + offs_k[None, :],
            mask=row_mask[:, None], other=0.0
        ).to(tl.float32)

        c = x2 * tl.sigmoid(x2) * x1

        block_max = tl.max(tl.abs(c), axis=1)
        scale = block_max / 448.0
        scale = tl.where(scale > 0, scale, 1.0)
        c_fp8 = (c / scale[:, None]).to(tl.float8e4nv)

        # Write FP8 data in padded layout (same position as C_raw input)
        tl.store(
            swiglu_out_ptr + src_rows[:, None] * I_dim + k_off + offs_k[None, :],
            c_fp8, mask=row_mask[:, None]
        )
        # Write scale column-major: [NUM_I_BLOCKS, valid_m] with valid_m contiguous
        tl.store(
            block_scale_ptr + kb * valid_m_stride + src_rows,
            scale, mask=row_mask
        )

    _pdl_launch_dependents()


# ---- CuTe tensor creation helpers ----

def _make_cute(torch_tensor, cutlass_dtype):
    """Create CuTe tensor from a PyTorch tensor with dynamic layout.
    leading_dim = dimension index where stride == 1."""
    ct = from_dlpack(torch_tensor, assumed_align=16)
    ct.element_type = cutlass_dtype
    # Find the dimension with stride 1 (the "leading" dim in CUTLASS convention)
    ld = None
    for i, s in enumerate(torch_tensor.stride()):
        if s == 1:
            ld = i
            break
    ct = ct.mark_layout_dynamic(leading_dim=ld)
    return ct


def _update_cute_data_ptr(cute_tensor, new_data_ptr):
    """Fast update of CuTe tensor data pointer via direct memref manipulation.
    Avoids the full from_dlpack + mark_layout_dynamic overhead (~33us per tensor).

    The CUTLASS DSL memref descriptor has data_ptr at offset 0.
    Offset 8+ contains packed shape/stride data (NOT aligned_ptr).
    """
    c_ptrs = cute_tensor.__c_pointers__()
    memref_addr = c_ptrs[0]
    # Only update data_ptr at offset 0
    ctypes.c_uint64.from_address(memref_addr).value = new_data_ptr


def _init_cute(torch_tensor, cutlass_dtype):
    """Create a CuTe tensor and force memref build."""
    ct = _make_cute(torch_tensor, cutlass_dtype)
    ct.__c_pointers__()
    return ct


# ---- CUTLASS compilation and caching ----

class _BufferCuTe(NamedTuple):
    """CuTe wrappers for buffer tensors (stable pointers per valid_m)."""
    a: object
    sfa: object
    c: object
    gidx: object


class _WeightCuTe(NamedTuple):
    """CuTe wrappers for weight tensors (pointer updated every call)."""
    b: object
    sfb: object


_cutlass_cache = {}  # (valid_m, K, N, E, device_idx, config) -> compiled_gemm
_buffer_cache = {}   # (valid_m, K, N, E, device_idx) -> _Gemm1Buffers
_a_scale_t_cache = {}  # (T, device_idx) -> [T, NUM_K_BLOCKS] fp32 buffer
_buf_ct_cache = {}   # (valid_m, K, N, E, device_idx) -> _BufferCuTe
_weight_ct_cache = {}  # (N, K, E, device_idx) -> _WeightCuTe
_fast_launch_cache = {}  # full runtime launch key -> (capi_func, packed_args)
_stream_handle_cache = {}  # stream pointer -> CUstream object kept alive for packed args
_trace_dummy_cache = None  # cached dummy trace CuTe tensor for production path
_trace_real_torch = None   # real trace buffer (torch) when CUTLASS_TRACE=1
_trace_real_ct = None      # real trace buffer (CuTe) when CUTLASS_TRACE=1


def _compile_cutlass_gemm(valid_m, K, N, E, device, mma_m=128, mma_n=128, cl_m=1, cl_n=4):
    """Compile the CUTLASS contiguous grouped GEMM kernel for given shapes.
    Cached by (valid_m, K, N, E, mma_config). Returns compiled_gemm callable."""
    cache_key = (valid_m, K, N, E, device.index, mma_m, mma_n, cl_m, cl_n)
    if cache_key in _cutlass_cache:
        return _cutlass_cache[cache_key]

    NUM_K_BLOCKS = K // 128
    NUM_N_BLOCKS = N // 128

    # Create representative tensors for compilation with exact CUTLASS-expected
    # layouts. Layouts must match what the CUTLASS reference code produces via
    # cutlass_torch.matrix().
    # Helper to create cute tensor with explicit leading_dim (dimension index
    # where stride==1)
    def _mk(t, dtype):
        ct = from_dlpack(t, assumed_align=16)
        ct.element_type = dtype
        ld = None
        for i, s in enumerate(t.stride()):
            if s == 1:
                ld = i
                break
        ct = ct.mark_layout_dynamic(leading_dim=ld)
        return ct

    # A: [valid_m, K, 1] fp8, K contiguous. Strides (K, 1, valid_m*K)
    a_torch = torch.empty(1, valid_m, K, dtype=torch.int8, device=device)
    a_ct = _mk(a_torch.permute(1, 2, 0), cutlass.Float8E4M3FN)

    # B: [N, K, E] fp8, K contiguous. Strides (K, 1, N*K)
    b_torch = torch.empty(E, N, K, dtype=torch.int8, device=device)
    b_ct = _mk(b_torch.permute(1, 2, 0), cutlass.Float8E4M3FN)

    # C: [valid_m, N, 1] fp16, N contiguous. Strides (N, 1, valid_m*N)
    c_torch = torch.empty(1, valid_m, N, dtype=torch.float16, device=device)
    c_ct = _mk(c_torch.permute(1, 2, 0), cutlass.Float16)

    # SFA: [valid_m, NUM_K_BLOCKS, 1] fp32, column-major. Strides (1, valid_m, valid_m*NUM_K_BLOCKS)
    sfa_torch = torch.empty(1, NUM_K_BLOCKS, valid_m, dtype=torch.float32, device=device)
    sfa_ct = _mk(sfa_torch.permute(2, 1, 0), cutlass.Float32)

    # SFB: [N/128, K/128, E] fp32. Strides (K/128, 1, N/128*K/128)
    sfb_torch = torch.empty(E, NUM_N_BLOCKS, NUM_K_BLOCKS, dtype=torch.float32, device=device)
    sfb_ct = _mk(sfb_torch.permute(1, 2, 0), cutlass.Float32)

    # gidx_mapping: [valid_m] int32
    gidx_torch = torch.zeros(valid_m, dtype=torch.int32, device=device)
    gidx_ct = _mk(gidx_torch, cutlass.Int32)

    # Create kernel instance
    # cluster(1,4): N-multicast only — confirmed best config for grouped FP8 GEMM1
    # Tested: cl(1,8)=slower, 2CTA cl(2,2/2,4)=slower, mma(64,128)=slower
    gemm = BlockwiseContiguousGroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(mma_m, mma_n),
        cluster_shape_mn=(cl_m, cl_n),
    )

    # Compute max active clusters
    hardware_info = cutlass.utils.HardwareInfo()
    cluster_size = cl_m * cl_n
    max_active_clusters = hardware_info.get_max_active_clusters(cluster_size)

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

    # Trace buffer — real buffer if CUTLASS_TRACE=1, else 1-element dummy
    if os.environ.get('CUTLASS_TRACE', '') == '1':
        from blockwise_contiguous_grouped_gemm import TRACE_TOTAL
        global _trace_real_torch, _trace_real_ct
        _trace_real_torch = torch.zeros(TRACE_TOTAL, dtype=torch.int64, device=device)
        _trace_real_ct = from_dlpack(_trace_real_torch, assumed_align=16)
        _trace_real_ct.element_type = cutlass.Int64
        _trace_real_ct = _trace_real_ct.mark_layout_dynamic(leading_dim=0)
        trace_ct = _trace_real_ct
    else:
        trace_torch = torch.zeros(1, dtype=torch.int64, device=device)
        trace_ct = from_dlpack(trace_torch, assumed_align=16)
        trace_ct.element_type = cutlass.Int64
        trace_ct = trace_ct.mark_layout_dynamic(leading_dim=0)

    compiled_gemm = cute.compile(
        gemm,
        a_ct, b_ct, c_ct, sfa_ct, sfb_ct, gidx_ct,
        max_active_clusters,
        current_stream,
        trace_ct,
    )

    _cutlass_cache[cache_key] = compiled_gemm
    return compiled_gemm


class _Gemm1Buffers(NamedTuple):
    A_raw: torch.Tensor       # [1, valid_m, K] fp8
    SFA_raw: torch.Tensor     # [1, NUM_K_BLOCKS, valid_m] fp32
    C_raw: torch.Tensor       # [1, valid_m, N] fp16
    gidx_mapping: torch.Tensor  # [valid_m] int32


def _get_buffers(valid_m, K, N, E, device):
    """Get or create pre-allocated buffers for gather/GEMM."""
    buf_key = (valid_m, K, N, E, device.index)
    if buf_key not in _buffer_cache:
        NUM_K_BLOCKS = K // 128
        _buffer_cache[buf_key] = _Gemm1Buffers(
            A_raw=torch.empty(1, valid_m, K, dtype=torch.float8_e4m3fn, device=device),
            SFA_raw=torch.empty(1, NUM_K_BLOCKS, valid_m, dtype=torch.float32, device=device),
            C_raw=torch.empty(1, valid_m, N, dtype=torch.float16, device=device),
            gidx_mapping=torch.zeros(valid_m, dtype=torch.int32, device=device),
        )
    return _buffer_cache[buf_key]


# ---- Main entry point ----

def run_gemm1(
    hidden_states,      # [T, K=7168] fp8
    A_scale,            # [NUM_K_BLOCKS, T] fp32 (column-major, T contiguous)
    gemm1_weights,      # [E=32, N=4096, K=7168] fp8
    W1_scale,          # [E, N/128, K/128] fp32
    expert_offsets,     # [E+1] int32
    sorted_token_ids,   # [total_routed] int64
    padded_offsets,     # [E+1] int32 — pre-computed padded offsets
    valid_m,            # int — total padded row count (sum of padded expert counts)
    E_local,            # int
    mma_m=128, mma_n=128, cl_m=1, cl_n=4,  # CUTLASS config (64,128,1,1 for medium-T)
):
    """Run GEMM1 pipeline: gather → CUTLASS grouped GEMM → return C_raw.
    Returns C_raw [1, valid_m, N=4096] fp16 in padded layout for fused SwiGLU.
    padded_offsets and valid_m are pre-computed in compute_offsets_kernel to
    avoid D2H sync."""
    K = hidden_states.shape[1]  # 7168
    N = 4096
    device = hidden_states.device
    T = A_scale.shape[1]
    NUM_K_BLOCKS = K // 128
    NUM_N_BLOCKS = N // 128
    # Transpose A_scale [56, T] → [T, 56] into pre-allocated buffer
    # (avoids .T.contiguous() clone allocation ~40us CPU overhead per call)
    a_scale_key = (T, device.index)
    if a_scale_key not in _a_scale_t_cache:
        _a_scale_t_cache[a_scale_key] = torch.empty(
            T, NUM_K_BLOCKS, dtype=torch.float32, device=device
        )
    A_scale_T = _a_scale_t_cache[a_scale_key]
    A_scale_T.copy_(A_scale.T, non_blocking=True)
    stride_as = NUM_K_BLOCKS  # = 56, row-major

    if valid_m == 0:
        return

    # Get or create buffers (cached by valid_m)
    bufs = _get_buffers(valid_m, K, N, E_local, device)
    A_raw = bufs.A_raw
    SFA_raw = bufs.SFA_raw
    C_raw = bufs.C_raw
    gidx_mapping = bufs.gidx_mapping

    # Step 2-4: Fused gather (A + SFA + gidx) — 1D grid, one block per padded row
    _launch(fused_gather_kernel, (valid_m,),
        hidden_states, A_scale_T, sorted_token_ids,
        expert_offsets, padded_offsets,
        A_raw, SFA_raw, gidx_mapping,
        stride_as, valid_m,
        _cache_key=valid_m,
        K=K, BLOCK_COPY=1024, NUM_K_BLOCKS=NUM_K_BLOCKS, E_LOCAL=E_local,
    )

    # Step 5: Prepare CuTe tensors
    # Buffer CuTe tensors: created once per valid_m, pointers are stable.
    buf_ct_key = (valid_m, K, N, E_local, device.index)
    if buf_ct_key not in _buf_ct_cache:
        _buf_ct_cache[buf_ct_key] = _BufferCuTe(
            a=_init_cute(A_raw.view(torch.int8).permute(1, 2, 0), cutlass.Float8E4M3FN),
            sfa=_init_cute(SFA_raw.permute(2, 1, 0), cutlass.Float32),
            c=_init_cute(C_raw.permute(1, 2, 0), cutlass.Float16),
            gidx=_init_cute(gidx_mapping, cutlass.Int32),
        )
    buf_ct = _buf_ct_cache[buf_ct_key]

    # Weight CuTe tensors: shape fixed, data pointer patched every call (~2us vs ~33us).
    B_int8 = gemm1_weights.view(torch.int8)
    b_view = B_int8.as_strided([N, K, E_local], [K, 1, N * K])
    sfb_view = W1_scale.as_strided(
        [NUM_N_BLOCKS, NUM_K_BLOCKS, E_local],
        [NUM_K_BLOCKS, 1, NUM_N_BLOCKS * NUM_K_BLOCKS]
    )
    weight_ct_key = (N, K, E_local, device.index)
    if weight_ct_key not in _weight_ct_cache:
        _weight_ct_cache[weight_ct_key] = _WeightCuTe(
            b=_init_cute(b_view, cutlass.Float8E4M3FN),
            sfb=_init_cute(sfb_view, cutlass.Float32),
        )
    else:
        _update_cute_data_ptr(_weight_ct_cache[weight_ct_key].b, gemm1_weights.data_ptr())
        _update_cute_data_ptr(_weight_ct_cache[weight_ct_key].sfb, W1_scale.data_ptr())
    weight_ct = _weight_ct_cache[weight_ct_key]

    # Run CUTLASS GEMM — fast launch path bypasses generate_execution_args_positional
    # on subsequent calls. CuTe tensor memref pointers are stable (we patch data_ptr
    # in-place via _update_cute_data_ptr), so packed_args can be reused directly.
    torch_stream = torch.cuda.current_stream()
    stream_ptr = torch_stream.cuda_stream
    if stream_ptr not in _stream_handle_cache:
        _stream_handle_cache[stream_ptr] = cuda_drv.CUstream(stream_ptr)
    current_stream = _stream_handle_cache[stream_ptr]
    _fl_key = (
        valid_m,
        K,
        N,
        E_local,
        device.index,
        mma_m,
        mma_n,
        cl_m,
        cl_n,
        stream_ptr,
    )
    if _fl_key in _fast_launch_cache:
        capi_func, packed_args = _fast_launch_cache[_fl_key]
        capi_func(packed_args)
    else:
        compiled_gemm = _compile_cutlass_gemm(valid_m, K, N, E_local, device,
                                               mma_m=mma_m, mma_n=mma_n, cl_m=cl_m, cl_n=cl_n)

        # Trace buffer
        if os.environ.get('CUTLASS_TRACE', '') == '1' and _trace_real_ct is not None:
            trace_ct = _trace_real_ct
        else:
            global _trace_dummy_cache
            if _trace_dummy_cache is None:
                _td = torch.zeros(1, dtype=torch.int64, device=device)
                _td_ct = from_dlpack(_td, assumed_align=16)
                _td_ct.element_type = cutlass.Int64
                _td_ct = _td_ct.mark_layout_dynamic(leading_dim=0)
                _trace_dummy_cache = _td_ct
            trace_ct = _trace_dummy_cache

        compiled_gemm(
            buf_ct.a, weight_ct.b, buf_ct.c, buf_ct.sfa, weight_ct.sfb, buf_ct.gidx,
            current_stream,
            trace_ct,
        )

        # Capture packed_args for subsequent direct calls.
        # JitCompiledFunction.__call__ → _default_executor (JitExecutor) → _tls.packed_args
        executor = compiled_gemm._default_executor
        if executor is not None:
            tls = executor._tls
            packed_args = getattr(tls, "packed_args", None)
            if packed_args is not None:
                _fast_launch_cache[_fl_key] = (executor.capi_func, packed_args)

    return C_raw
