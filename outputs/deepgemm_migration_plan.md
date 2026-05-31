# DeepGEMM Migration Plan for FlashInfer MoE Contest

## 0. Current State

- DeepGEMM repo: `/home/ludens/workspace/github/DeepGEMM`
- DeepGEMM HEAD: `714dd1a`
- Submodules initialized:
  - `third-party/cutlass`
  - `third-party/fmt`
- Current submitted solution path: `solution/triton/kernel.py`
- Current Modal result: Triton solution passes all 19 workloads.
- Important packaging constraint: `pack_solution_from_files` only packs first-level source files in
  `solution/triton` or `solution/cuda`; it does not recursively include external repo directories.

## 1. Why We Cannot Directly Import DeepGEMM

DeepGEMM is a Python package with a C++ extension and runtime JIT. The contest solution cannot assume
`deep_gemm` is installed in the evaluator image unless we modify the Modal image, which is not a real
`solution/triton` or `solution/cuda` migration.

Therefore, the practical migration path is:

1. Study DeepGEMM's layout and scheduling design.
2. Re-implement the relevant pieces inside `solution/cuda`.
3. Keep the code self-contained for FlashInfer-Bench packing.

## 2. DeepGEMM Pieces Relevant to This Contest

### 2.1 M-grouped GEMM contiguous layout

Source files:

- `DeepGEMM/csrc/apis/gemm.hpp`
- `DeepGEMM/tests/generators.py`
- `DeepGEMM/deep_gemm/include/deep_gemm/scheduler/gemm.cuh`

DeepGEMM expects MoE expert rows to be concatenated by expert:

```text
expert 0 rows | padding | expert 1 rows | padding | ... | expert 31 rows
```

The grouped layout can be represented as cumulative valid row ends:

```text
grouped_layout[e] = aligned_start[e] + count[e]
aligned_start[e + 1] = align(grouped_layout[e], alignment)
```

This is a better match for the contest than the current token-major layout.

### 2.2 SM100 grouped GEMM kernel configuration

Source files:

- `DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp`
- `DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`
- `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`

Important details to migrate:

- B200 is SM100, so the relevant kernel family is `sm100_*`.
- Grouped MoE uses M-grouped scheduling.
- DeepGEMM sets `block_n = 128`, `block_k = 128 / element_size`, and uses an M alignment chosen by heuristics.
- For m-grouped GEMM, DeepGEMM enables swap-A/B and may use cluster-N multicast.

### 2.3 Scale format mismatch

Contest input scale format:

- hidden scale: FP32, shape `[H / 128, T]`
- W13 scale: FP32, shape `[E_local, 2I / 128, H / 128]`
- W2 scale: FP32, shape `[E_local, H / 128, I / 128]`

DeepGEMM SM100 expected scale format:

- packed UE8M0 int scale
- TMA-aligned MN-major layout

This is the largest migration gap. A full DeepGEMM-style port must add scale-packing kernels before
the grouped GEMMs.

## 3. Migration Stages

### Stage A: Preserve current Triton solution

Keep `config.toml` pointing to `solution/triton/kernel.py` until the CUDA path passes Modal.

Validation:

```bash
conda activate mlsys
modal run ./scripts/run_modal.py
```

Expected baseline: all workloads pass, current large-batch speedup is around 12-13x.

### Stage B: Make `solution/cuda` compile and run as-is

Before adding DeepGEMM logic, switch `config.toml` to:

```toml
[build]
language = "cuda"
entry_point = "main.cpp::run"
```

Then run Modal. This checks whether the FlashInfer-Bench CUDA builder can compile the existing
`main.cpp + kernel.cu + kernel.h` path.

Rollback if it fails.

### Stage C: Convert CUDA routing output to DeepGEMM-compatible psum layout

Modify `solution/cuda/kernel.cu` so local expert row packing uses aligned starts:

```text
aligned_start[0] = 0
grouped_layout[e] = aligned_start[e] + count[e]
aligned_start[e + 1] = align(grouped_layout[e], DG_ALIGNMENT)
total_packed_rows = aligned_start[32]
```

Required buffer changes:

- allocate packed buffers by `total_packed_rows`, not `sum(counts)`
- keep `pair_rows` pointing only to valid rows
- ignore padded rows during final scatter

This stage should preserve BF16 cuBLAS computation first. It is a layout migration, not yet a kernel
replacement.

### Stage D: Replace GEMM1 with direct FP8 block-scale grouped path

Current CUDA path dequantizes A and W13 into BF16 and calls cuBLAS BF16 GEMM. DeepGEMM's direction is
to keep FP8 operands and consume scale in the GEMM kernel.

Minimal self-contained version:

- gather routed hidden states into packed FP8 rows
- gather hidden scales into `[packed_rows, H / 128]`
- call a direct FP8 block-scale matmul path for each expert or expert run

Ideal DeepGEMM-like version:

- replace per-expert calls with one M-grouped persistent SM100 kernel
- use `grouped_layout` for scheduling

### Stage E: Fuse or quantize SwiGLU output for GEMM2

DeepGEMM/MegaMoE style re-quantizes the intermediate activation after SwiGLU.

Required pieces:

- `SwiGLU` kernel
- BF16-to-FP8 quantization per row/block
- scale output in DeepGEMM-compatible layout

Risk:

- This changes numerical behavior more than GEMM1 migration.
- Must validate correctness with FlashInfer-Bench tolerance.

### Stage F: Replace GEMM2 with grouped FP8 block-scale path

Same idea as GEMM1:

- A: FP8 intermediate activation `[packed_rows, I]`
- B: W2 FP8 `[E_local, H, I]`
- output: BF16 `[packed_rows, H]`
- final combine uses `pair_rows` and routing weights.

### Stage G: Optional full SM100/CuTe port

This is the true "complete DeepGEMM migration", but it is much larger:

- copy/adapt SM100 MMA/TMA helpers
- copy/adapt scheduler
- generate fixed-shape kernels instead of runtime JIT
- add UE8M0 scale packing
- add static launch wrappers for this contest shape

This is feasible, but not a one-shot edit. It should be treated as a separate implementation project.

## 4. Recommended Immediate Path

Do these in order:

1. Run Stage B to verify current CUDA build.
2. Implement Stage C aligned psum packing.
3. Run Modal and compare against Triton.
4. Implement Stage D using existing `cublasLt` FP8 block-scale helper as a stepping stone.
5. Only after D passes, attempt E/F.

This gives us a gradual path toward DeepGEMM without breaking the known-good Triton submission.
