"""CUTLASS FP8 GEMM2 helper — BlockwiseContiguousGroupedGemmKernel for FP8×FP8.

Reuses the same CUTLASS kernel as GEMM1 (BlockwiseContiguousGroupedGemmKernel)
with different dimensions:
  GEMM1: A=[valid_m, K=7168], B=[N=4096, K=7168, E], C=[valid_m, N=4096]
  GEMM2: A=[valid_m, K=2048], B=[N=7168, K=2048, E], C=[valid_m, N=7168]

A = FP8 SwiGLU output in padded layout (written by fused_scatter_swiglu_fp8_kernel)
B = raw FP8 W2 weights (no dequant needed!)
SFA = per-block SwiGLU scale [NUM_I_BLOCKS, valid_m]
SFB = W2 per-block scale [H/128, I/128, E]
"""

import os
import sys
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda_drv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockwise_contiguous_grouped_gemm import BlockwiseContiguousGroupedGemmKernel


# ---- CuTe helpers (same as gemm1) ----

def _init_cute(torch_tensor, cutlass_dtype):
    ct = from_dlpack(torch_tensor, assumed_align=16)
    ct.element_type = cutlass_dtype
    ld = None
    for i, s in enumerate(torch_tensor.stride()):
        if s == 1:
            ld = i
            break
    ct = ct.mark_layout_dynamic(leading_dim=ld)
    ct.__c_pointers__()  # force memref build for _update_cute_data_ptr
    return ct

def _update_cute_data_ptr(ct, new_data_ptr):
    import ctypes
    c_ptrs = ct.__c_pointers__()
    memref_addr = c_ptrs[0]
    ctypes.c_uint64.from_address(memref_addr).value = new_data_ptr


# ---- Caches ----

_cutlass_cache = {}   # (valid_m, E, device_idx, config) -> compiled_gemm
_buffer_cache = {}    # (valid_m, device_idx) -> buffers
_buf_ct_cache = {}    # (valid_m, device_idx) -> CuTe buffer wrappers
_weight_ct_cache = {}  # (H, I, E, device_idx) -> CuTe weight wrappers
_fast_launch_cache = {}  # full runtime launch key -> (capi_func, packed_args)
_stream_handle_cache = {}  # stream pointer -> CUstream object kept alive for packed args
_trace_dummy_cache = None

# Constants
I = 2048   # K dimension for GEMM2
H = 7168   # N dimension for GEMM2
NUM_I_BLOCKS = I // 128   # 16
NUM_H_BLOCKS = H // 128   # 56

# Kernel config: mma/cluster shape (configurable via env for sweep)
_gemm2_fp8_cfg = os.environ.get('SWEEP_GEMM2_FP8_CFG', '128_128_1_1')
_fp8_parts = _gemm2_fp8_cfg.split('_')
_FP8_MMA_M, _FP8_MMA_N = int(_fp8_parts[0]), int(_fp8_parts[1])
_FP8_CL_M, _FP8_CL_N = int(_fp8_parts[2]), int(_fp8_parts[3])
_FP8_CLUSTER_SIZE = _FP8_CL_M * _FP8_CL_N


def _compile(valid_m, E, device, mma_m_ov=None, mma_n_ov=None, cl_m_ov=None, cl_n_ov=None):
    _mm = mma_m_ov or _FP8_MMA_M
    _mn = mma_n_ov or _FP8_MMA_N
    _cm = cl_m_ov or _FP8_CL_M
    _cn = cl_n_ov or _FP8_CL_N
    _ck = (_mm, _mn, _cm, _cn)
    cache_key = (valid_m, E, device.index, _ck)
    if cache_key in _cutlass_cache:
        return _cutlass_cache[cache_key]

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

    # A: [valid_m, K=I, 1] fp8, K contiguous
    a_torch = torch.empty(1, valid_m, I, dtype=torch.int8, device=device)
    a_ct = _mk(a_torch.permute(1, 2, 0), cutlass.Float8E4M3FN)

    # B: [N=H, K=I, E] fp8, K contiguous
    b_torch = torch.empty(E, H, I, dtype=torch.int8, device=device)
    b_ct = _mk(b_torch.permute(1, 2, 0), cutlass.Float8E4M3FN)

    # C: [valid_m, N=H, 1] bf16, N contiguous (acc in FP32, epilogue casts)
    c_torch = torch.empty(1, valid_m, H, dtype=torch.bfloat16, device=device)
    c_ct = _mk(c_torch.permute(1, 2, 0), cutlass.BFloat16)

    # SFA: [valid_m, NUM_I_BLOCKS, 1] fp32, column-major (valid_m stride=1)
    sfa_torch = torch.empty(1, NUM_I_BLOCKS, valid_m, dtype=torch.float32, device=device)
    sfa_ct = _mk(sfa_torch.permute(2, 1, 0), cutlass.Float32)

    # SFB: [H/128, I/128, E] fp32, I/128 contiguous
    sfb_torch = torch.empty(E, NUM_H_BLOCKS, NUM_I_BLOCKS, dtype=torch.float32, device=device)
    sfb_ct = _mk(sfb_torch.permute(1, 2, 0), cutlass.Float32)

    # gidx_mapping: [valid_m] int32
    gidx_torch = torch.zeros(valid_m, dtype=torch.int32, device=device)
    gidx_ct = _mk(gidx_torch, cutlass.Int32)

    # Kernel: same class as GEMM1, different compile-time shapes
    gemm = BlockwiseContiguousGroupedGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(_mm, _mn),
        cluster_shape_mn=(_cm, _cn),
    )

    hardware_info = cutlass.utils.HardwareInfo()
    _cs = _cm * _cn
    max_active_clusters = hardware_info.get_max_active_clusters(_cs)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda_drv.CUstream(torch_stream.cuda_stream)

    trace_torch = torch.zeros(1, dtype=torch.int64, device=device)
    trace_ct = from_dlpack(trace_torch, assumed_align=16)
    trace_ct.element_type = cutlass.Int64
    trace_ct = trace_ct.mark_layout_dynamic(leading_dim=0)

    compiled = cute.compile(
        gemm,
        a_ct, b_ct, c_ct, sfa_ct, sfb_ct, gidx_ct,
        max_active_clusters,
        current_stream,
        trace_ct,
    )

    _cutlass_cache[cache_key] = compiled
    return compiled


class _Gemm2FP8Buffers:
    __slots__ = ('A', 'SFA', 'C', 'gidx')
    def __init__(self, valid_m, device):
        # A (SwiGLU FP8 output) and SFA (block scales) are written by SwiGLU kernel
        # We just need pre-allocated buffers with the right shapes
        self.A = torch.empty(1, valid_m, I, dtype=torch.float8_e4m3fn, device=device)
        self.SFA = torch.empty(1, NUM_I_BLOCKS, valid_m, dtype=torch.float32, device=device)
        self.C = torch.empty(1, valid_m, H, dtype=torch.bfloat16, device=device)
        self.gidx = torch.zeros(valid_m, dtype=torch.int32, device=device)


def _get_buffers(valid_m, device):
    key = (valid_m, device.index)
    if key not in _buffer_cache:
        _buffer_cache[key] = _Gemm2FP8Buffers(valid_m, device)
    return _buffer_cache[key]


def run_gemm2_fp8(
    swiglu_fp8,       # [1, valid_m, I] fp8 — SwiGLU output in PADDED layout
    swiglu_scale,     # [1, NUM_I_BLOCKS, valid_m] fp32 — per-block scales, column-major
    W2_weights,       # [E, H, I] fp8 — raw weights (no dequant!)
    W2_scale,         # [E, H/128, I/128] fp32
    gidx_mapping,     # [valid_m] int32 — group index mapping (same as GEMM1)
    gemm2_out,        # [1, valid_m, H] bf16 — output buffer
    valid_m,          # int
    E_local,          # int
    mma_m=None, mma_n=None, cl_m=None, cl_n=None,  # override CUTLASS config
):
    """Run FP8×FP8 GEMM2 using BlockwiseContiguousGroupedGemmKernel.

    Returns gemm2_out [1, valid_m, H] bf16 in padded layout.
    """
    if valid_m == 0:
        return gemm2_out

    device = W2_weights.device

    # CuTe buffer wrappers are stable for the cached backing buffers.
    buf_ct_key = (valid_m, device.index)
    if buf_ct_key not in _buf_ct_cache:
        bufs = _get_buffers(valid_m, device)
        _buf_ct_cache[buf_ct_key] = {
            'a': _init_cute(bufs.A.view(torch.int8).permute(1, 2, 0), cutlass.Float8E4M3FN),
            'sfa': _init_cute(bufs.SFA.permute(2, 1, 0), cutlass.Float32),
            'c': _init_cute(bufs.C.permute(1, 2, 0), cutlass.BFloat16),
            'gidx': _init_cute(bufs.gidx, cutlass.Int32),
        }
    buf_ct = _buf_ct_cache[buf_ct_key]

    # Patch data pointers to actual runtime buffers
    _update_cute_data_ptr(buf_ct['a'], swiglu_fp8.data_ptr())
    _update_cute_data_ptr(buf_ct['sfa'], swiglu_scale.data_ptr())
    _update_cute_data_ptr(buf_ct['c'], gemm2_out.data_ptr())
    _update_cute_data_ptr(buf_ct['gidx'], gidx_mapping.data_ptr())

    # Weight CuTe tensors: [N=H, K=I, E] view of [E, H, I].
    B_int8 = W2_weights.view(torch.int8)
    b_view = B_int8.as_strided([H, I, E_local], [I, 1, H * I])
    sfb_view = W2_scale.as_strided(
        [NUM_H_BLOCKS, NUM_I_BLOCKS, E_local],
        [NUM_I_BLOCKS, 1, NUM_H_BLOCKS * NUM_I_BLOCKS]
    )
    weight_ct_key = (H, I, E_local, device.index)
    if weight_ct_key not in _weight_ct_cache:
        _weight_ct_cache[weight_ct_key] = {
            'b': _init_cute(b_view, cutlass.Float8E4M3FN),
            'sfb': _init_cute(sfb_view, cutlass.Float32),
        }
    else:
        _update_cute_data_ptr(_weight_ct_cache[weight_ct_key]['b'], W2_weights.data_ptr())
        _update_cute_data_ptr(_weight_ct_cache[weight_ct_key]['sfb'], W2_scale.data_ptr())
    weight_ct = _weight_ct_cache[weight_ct_key]

    torch_stream = torch.cuda.current_stream()
    stream_ptr = torch_stream.cuda_stream
    if stream_ptr not in _stream_handle_cache:
        _stream_handle_cache[stream_ptr] = cuda_drv.CUstream(stream_ptr)
    current_stream = _stream_handle_cache[stream_ptr]

    # Fast launch path
    _fl_key = (valid_m, E_local, device.index, mma_m, mma_n, cl_m, cl_n, stream_ptr)
    if _fl_key in _fast_launch_cache:
        capi_func, packed_args = _fast_launch_cache[_fl_key]
        capi_func(packed_args)
    else:
        compiled = _compile(valid_m, E_local, device,
                           mma_m_ov=mma_m, mma_n_ov=mma_n, cl_m_ov=cl_m, cl_n_ov=cl_n)

        global _trace_dummy_cache
        if _trace_dummy_cache is None:
            _td = torch.zeros(1, dtype=torch.int64, device=device)
            _td_ct = from_dlpack(_td, assumed_align=16)
            _td_ct.element_type = cutlass.Int64
            _td_ct = _td_ct.mark_layout_dynamic(leading_dim=0)
            _trace_dummy_cache = _td_ct

        compiled(
            buf_ct['a'], weight_ct['b'], buf_ct['c'],
            buf_ct['sfa'], weight_ct['sfb'], buf_ct['gidx'],
            current_stream,
            _trace_dummy_cache,
        )

        # Capture for fast launch
        executor = compiled._default_executor
        if executor is not None:
            tls = executor._tls
            packed_args = getattr(tls, "packed_args", None)
            if packed_args is not None:
                _fast_launch_cache[_fl_key] = (executor.capi_func, packed_args)

    return gemm2_out
