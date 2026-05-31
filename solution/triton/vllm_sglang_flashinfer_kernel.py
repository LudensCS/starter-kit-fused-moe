"""vLLM/SGLang/FlashInfer-inspired Triton MoE migration.

This file intentionally builds on ``deepgemm_kernel.py`` instead of copying the
whole implementation.  The kernel structure there already matches the contest
interface and has the DeepGEMM-style expert-major layout.  The migration here
pulls in the most directly portable ideas from production MoE stacks:

- vLLM/SGLang fused-MoE configs: FP8 block-wise kernels generally prefer
  smaller M tiles for decode-like batches and larger grouped-M reuse once the
  batch has enough routed rows.
- SGLang/DeepGEMM contiguous grouped GEMM: keep 128-row expert segment
  alignment for compact paths.
- FlashInfer DeepSeek-V3 routing: keep the fused no-aux grouped top-k routing
  and route-scale normalization in the Triton routing kernel.

The goal is a separate benchmarkable variant under ``solution/triton`` while
keeping ``deepgemm_kernel.py`` as the current stable baseline.
"""

from __future__ import annotations

try:
    from . import deepgemm_kernel as _base
except ImportError:
    import deepgemm_kernel as _base


# vLLM/SGLang block-wise FP8 MoE defaults often use smaller M tiles when tokens
# are spread thin across experts.  On this B200 contest shape, that regressed
# the large prefill workloads, so the migrated variant keeps the stable
# DeepGEMM-style 128-row GEMM1 tile and imports the gentler scheduling changes
# below instead.
_base.GEMM1_BLOCK_M = 128
_base.GEMM1_DENSE_BLOCK_M = 128

# Keep the B200-tested K tile from the DeepGEMM migration.  vLLM/SGLang often
# use K=128 for generic FP8 block-wise MoE, but this workload's H=7168 and
# in-kernel dequant/SwiGLU path was more stable with 64 in the current baseline.
_base.GEMM1_BLOCK_K = 64

# vLLM/SGLang use more aggressive grouped-M reuse for block-wise FP8 configs.
# This wrapper increases the group from the baseline's 8 to 16: enough to reuse
# N tiles without grouping too many different experts into the same scheduling
# window when per-expert M is small.
_base.GEMM_GROUP_SIZE_M = 16

# SGLang's default down-MoE FP8 block-wise path uses K=128.  GEMM2 has the
# shorter K dimension (I=2048) and no SwiGLU epilogue, so it is the safer place
# to import that larger reduction tile.
_base.GEMM2_DENSE_BLOCK_K = 128

# Keep the current baseline threshold.  Raising this to 2048 follows the
# decode-style intuition from FlashInfer/SGLang, but it hurts the large
# workloads here because compact host-planned jobs save more GEMM work.
_base.SYNC_FREE_SEQ_THRESHOLD = 1024


def _select_gemm2_config(max_count: int) -> tuple[int, int]:
    """vLLM/SGLang-style down-MoE M-tile heuristic.

    The baseline uses 32/64.  This variant adds a 16-row tile for very sparse
    local experts and raises the large-tile threshold, mirroring production
    fused-MoE behavior where decode-like batches avoid large masked row blocks.
    """

    if max_count <= 16:
        return 16, _base.GEMM2_SMALL_NUM_STAGES
    if max_count < 96:
        return _base.GEMM2_SMALL_BLOCK_M, _base.GEMM2_SMALL_NUM_STAGES
    return _base.GEMM2_LARGE_BLOCK_M, _base.GEMM2_LARGE_NUM_STAGES


_base._select_gemm2_config = _select_gemm2_config


run = _base.run
