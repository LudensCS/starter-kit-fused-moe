# `solution/triton/baseline.py` 技术溯源报告

生成时间: 2026-04-08 UTC

## 1. 报告范围与环境

本报告溯源的目标文件是 `solution/triton/baseline.py`。它并不自己实现 MoE 计算，而是把参数整理后直接转发给本地安装的 `flashinfer`。

本机确认的运行环境如下:

- Conda 环境: `mlsys`
- Python: `/root/miniconda3/envs/mlsys/bin/python`
- `flashinfer` 版本: `0.6.6`
- `flashinfer` 包目录: `/root/miniconda3/envs/mlsys/lib/python3.12/site-packages/flashinfer`
- FlashInfer C++ 源目录: `/root/miniconda3/envs/mlsys/lib/python3.12/site-packages/flashinfer/data/csrc`
- FlashInfer 生成源码缓存: `/root/.cache/flashinfer/0.6.6/100a/generated`
- FlashInfer cubin/cache 目录: `/root/.cache/flashinfer/cubins`

补充说明:

- 系统默认的 `/usr/bin/python3` 环境里没有 `flashinfer`。
- 因此后续所有复现、调试、源码定位，都应以 `conda activate mlsys` 为前提。

## 2. 一句话结论

`baseline.py` 本质上是一个“参数适配层”。真正的计算发生在 FlashInfer 的 Blackwell/SM100 MoE 实现里，路径是:

`baseline.py`
-> `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`
-> `flashinfer.fused_moe.core.get_trtllm_moe_sm100_module()`
-> JIT 构建并加载 `trtllm_fused_moe_*.cu`
-> C++ launcher
-> DeepSeekV3 路由 kernel
-> grouped GEMM1
-> 单独 SwiGLU activation
-> grouped GEMM2
-> finalize/scatter 回输出

也就是说，`baseline.py` 的“性能”几乎全部来自本地 `flashinfer`，而不是来自这个 Python 文件本身。

## 3. `baseline.py` 自身做了什么

源码位置: `solution/triton/baseline.py:1-83`

### 3.1 固定了任务配置

文件顶部把这条 fused MoE workload 的关键超参数写死了:

- `NUM_EXPERTS_GLOBAL = 256`
- `TOP_K = 8`
- `N_GROUP = 8`
- `TOPK_GROUP = 4`
- `HIDDEN_SIZE = 7168`
- `INTERMEDIATE_SIZE = 2048`
- `BLOCK_SIZE = 128`

这组参数与 DeepSeekV3 分组路由模式完全匹配:

- 256 个全局 expert
- 8 个 group
- 每组 32 个 expert
- 先选 4 组，再从候选 expert 里选 top-8

### 3.2 做了严格 shape 校验

`run()` 在进入 FlashInfer 之前先断言张量形状:

- `routing_logits.shape == (seq_len, 256)`
- `hidden_states.shape == (seq_len, 7168)`
- `hidden_states_scale.shape == (7168 // 128, seq_len) = (56, seq_len)`
- `gemm1_weights.shape == (local_num_experts, 4096, 7168)`
- `gemm1_weights_scale.shape == (local_num_experts, 32, 56)`
- `gemm2_weights.shape == (local_num_experts, 7168, 2048)`
- `gemm2_weights_scale.shape == (local_num_experts, 56, 16)`

这组 shape 与 FlashInfer 底层 DeepSeek FP8 block-scale 检查完全一致。

### 3.3 做了少量 dtype/布局整理

`baseline.py` 只做了三类预处理:

1. 把 Python / tensor 标量归一化为原生 `int` / `float`
2. 把下列张量转成 `float32` 并确保 contiguous
   - `routing_logits`
   - `hidden_states_scale`
   - `gemm1_weights_scale`
   - `gemm2_weights_scale`
3. 把其余输入做 contiguous
   - `hidden_states`
   - `gemm1_weights`
   - `gemm2_weights`
   - `routing_bias`（若存在）

这里最关键的一点是:

- `baseline.py` 没有把 `hidden_states` / `gemm1_weights` / `gemm2_weights` 反量化成 BF16/FP32
- 它保留了 FlashInfer 期望的 FP8 权重/激活格式，只把 block-scale 张量显式转成 FP32

这和 FlashInfer 底层检查是吻合的: 对 `DeepSeekFp8` 路径，`hidden_states` / `gemm*_weights` 必须是 `float8_e4m3fn`，而 scale 必须是 `float32`。

### 3.4 真正的计算只是一行调用

`baseline.py:64-83` 最终调用:

```python
trtllm_fp8_block_scale_moe(
    ...,
    NUM_EXPERTS_GLOBAL,
    TOP_K,
    N_GROUP,
    TOPK_GROUP,
    INTERMEDIATE_SIZE,
    local_expert_offset,
    local_num_experts,
    routed_scaling_factor,
    routing_method_type=2,
    use_shuffled_weight=False,
)
```

其中最重要的两个开关是:

- `routing_method_type=2`
- `use_shuffled_weight=False`

## 4. Python 到 CUDA 的完整调用链

### 4.1 第一层: `baseline.py` -> public Python API

`solution/triton/baseline.py:64-83`

调用 `flashinfer.fused_moe.trtllm_fp8_block_scale_moe(...)`。

对应的 FlashInfer 包导出位于:

- `flashinfer/fused_moe/__init__.py:18-34`

真正实现位于:

- `flashinfer/fused_moe/core.py:2522-2615`

这个 wrapper 做两件事:

1. 分配 `output = torch.empty(hidden_states.shape, dtype=torch.bfloat16, ...)`
2. 调用 `get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(...)`

### 4.2 第二层: Python custom op 封装

`flashinfer/fused_moe/core.py:1629-1788`

这里注册了 custom op:

- `flashinfer::trtllm_fp8_block_scale_moe`

关键行为:

- 若传入 `routing_logits`，则 `topk_ids` / `expert_weights` 用空 tensor 占位，由底层 kernel 自己算 routing
- 根据 `fp8_quantization_type` 推导激活/权重 dtype
- 构造 `MoERunner`
- 通过 `AutoTuner` 选择 tactic
- 最后调用 `moe_op.trtllm_fp8_block_scale_moe(...)`

这说明 `baseline.py` 走的是“routing logits 直接输入”的路径，不是“预先给出 topk ids / weights”的路径。

### 4.3 第三层: JIT 构建并加载 SM100 模块

`flashinfer/fused_moe/core.py:950-1010`

`get_trtllm_moe_sm100_module()` 会:

1. 调用 `gen_trtllm_gen_fused_moe_sm100_module()`
2. `build_and_load()`
3. `setup_cubin_loader(...)`

JIT 规格定义在:

- `flashinfer/jit/fused_moe.py:215-286`

这里能看到实际参与编译/装载的源文件:

- `trtllm_fused_moe_kernel_launcher.cu`
- `trtllm_fused_moe_runner.cu`
- `trtllm_fused_moe_routing_deepseek.cu`
- `trtllm_fused_moe_routing_llama4.cu`
- `trtllm_fused_moe_routing_renormalize.cu`
- `trtllm_fused_moe_dev_kernel.cu`
- `trtllm_batched_gemm_runner.cu`

并且它会从在线/缓存的 cubin 元数据里拉取 `flashinferMetaInfo.h`，用于 TensorRT-LLM batched GEMM kernel 列表和导出接口。

结论:

- `baseline.py` 实际上不是在调用纯 Python / Triton 逻辑
- 它是在调用一个 JIT 装配好的 TensorRT-LLM / CUDA fused MoE 运行时

## 5. `routing_method_type=2` 的精确含义

### 5.1 枚举定义

`flashinfer/fused_moe/core.py:61-75` 定义:

- `RoutingMethodType.DeepSeekV3 = 2`

源码注释直接写明:

- `Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts from the Top4 groups`

因此 `baseline.py` 的 `routing_method_type=2` 不是普通 softmax top-k，而是 DeepSeekV3 分组路由。

### 5.2 高层算法说明

FlashInfer 还在 `flashinfer/fused_moe/fused_routing_dsv3.py:121-196` 给出了同一算法的文字版:

1. `sigmoid(scores) + bias`
2. 对每个 group 取 top-2 expert 分数求和，得到 group score
3. 选 top-k groups
4. 在这些 groups 内再选 top-k experts
5. 最终路由权重用 `sigmoid_scores / sum(sigmoid_scores) * routed_scaling_factor`

注意最后一步非常关键:

- expert 选择依据是 `sigmoid(score) + bias`
- 但最终归一化权重使用的是不带 bias 的 `sigmoid(score)`

这和很多“直接对偏置后分数归一化”的 MoE 实现不同。

### 5.3 CUDA 路由 kernel 的底层实现

对应 CUDA 文件:

- `flashinfer/data/csrc/trtllm_fused_moe_routing_deepseek.cu`

关键代码点:

- `routing_deepseek.cu:97-116`
  - 先把 router logits 读成 `float`
  - 计算 `sigmoid`
  - 再加 `bias`
- `routing_deepseek.cu:128-169`
  - 每组做 top-2 -> group score
  - 再做 top-group -> top-expert
- `routing_deepseek.cu:236-263`
  - 最终权重用 `sigmoid` 分数归一化后乘 `routeScale`
  - 同时写出 expert index / packed topk 结果

这与仓库内 `solution/triton/kernel.py:_build_local_plan()` 完全对应:

- `kernel.py:226-227` 计算 `s = sigmoid(logits)` 和 `s_with_bias = s + bias`
- `kernel.py:229-234` 计算 group score 并选 top groups
- `kernel.py:245-250` 在候选 groups 内选 top-k experts
- `kernel.py:252-253` 用 `sigmoid` 分数归一化并乘 `routed_scaling_factor`

因此可以把 `solution/triton/kernel.py` 视为这条 DeepSeekV3 路由的“可读 Python/Triton 语义展开版”。

## 6. 张量契约、layout 和 dtype 约束

### 6.1 `baseline.py` 显式断言的输入契约

来自 `solution/triton/baseline.py:29-44`:

| 张量 | 形状 |
| --- | --- |
| `routing_logits` | `(seq_len, 256)` |
| `routing_bias` | `(256,)` 或兼容末维 256 |
| `hidden_states` | `(seq_len, 7168)` |
| `hidden_states_scale` | `(56, seq_len)` |
| `gemm1_weights` | `(local_num_experts, 4096, 7168)` |
| `gemm1_weights_scale` | `(local_num_experts, 32, 56)` |
| `gemm2_weights` | `(local_num_experts, 7168, 2048)` |
| `gemm2_weights_scale` | `(local_num_experts, 56, 16)` |

### 6.2 FlashInfer 底层的额外强约束

来自 `trtllm_fused_moe_kernel_launcher.cu:853-980`:

- DeepSeekV3 路径要求:
  - `n_group != 0`
  - `topk_group != 0`
  - `num_experts % n_group == 0`
  - `top_k <= 8`
  - `topk_group <= 4`
  - `top_k < topk_group * num_experts / n_group`
- 通用要求:
  - `num_experts % 4 == 0`
  - `num_experts > top_k`
  - `local_expert_offset + local_num_experts <= num_experts`
- FP8 DeepSeek 量化要求:
  - `hidden_states.dtype == float8_e4m3fn`
  - `gemm1_weights.dtype == float8_e4m3fn`
  - `gemm2_weights.dtype == float8_e4m3fn`
  - `hidden_states_scale.dtype == float32`
  - `gemm1_weights_scale.dtype == float32`
  - `gemm2_weights_scale.dtype == float32`
  - `intermediate_size % 128 == 0`

### 6.3 `use_shuffled_weight=False` 的含义

从接口设计看，`use_shuffled_weight` 会影响底层 GEMM runner 的权重布局/选择策略。`baseline.py` 显式把它设为 `False`，说明它使用未 shuffle 的权重布局。

源码依据:

- Python op 接口有 `use_shuffled_weight` 参数: `core.py:1654`, `core.py:2539-2545`
- 该参数被一路传入 `MoERunner` 和底层 launcher: `core.py:1713-1723`, `core.py:1757-1783`

基于源码的推断:

- 这条 baseline 假设输入权重已经是 FlashInfer 默认可接受的 MajorK 布局
- 它没有依赖额外的 offline weight shuffle 预处理

## 7. 真正的执行流水线

### 7.1 Routing 阶段

对应:

- `trtllm_fused_moe_runner.cu:52-101`

输出不仅有 top-k expert 和权重，还会生成一整套后续 grouped GEMM 需要的辅助索引:

- expert histogram
- `expanded_idx_to_permuted_idx`
- `permuted_idx_to_token_idx`
- CTA 调度相关映射

这说明底层实现并不是“先 topk，再普通 for-loop matmul”，而是 routing 和后续 grouped GEMM 调度紧耦合。

### 7.2 GEMM1 / Permute 阶段

对应:

- `trtllm_fused_moe_runner.cu:586-599`

`mPermuteGemm1.run(...)` 同时做了:

- 输入 token 的按 expert 重排
- FP8 block-scale 反量化所需的 scale 处理
- 第一层 grouped GEMM

所以它并不是先把全部 token 展开成 BF16 再 matmul，而是直接进入专用 batched/grouped GEMM 路径。

### 7.3 激活阶段

对应:

- `trtllm_fused_moe_runner.cu:601-609`

源码注释明确写了两点:

- DeepSeek FP8 路径下，activation 不和 FC1 融合
- 原因是 weights shuffling 约束以及 cubin 没有 fused activation

因此流水线是:

- GEMM1
- 单独 activation
- GEMM2

而不是:

- Fused GEMM1 + SwiGLU + GEMM2

### 7.4 GEMM2 阶段

对应:

- `trtllm_fused_moe_runner.cu:612-619`

这里再次走 grouped GEMM，但输入已经变成激活后的中间结果及其 scale。

### 7.5 Finalize 阶段

对应:

- `trtllm_fused_moe_runner.cu:494-518`
- `trtllm_fused_moe_runner.cu:621-625`

这里负责:

- 用 routing weights 对 expert 输出加权
- scatter/add 回 token 顺序
- 把 padded/permuted 表示恢复成最终 `(seq_len, hidden_size)` 输出

也就是说，MoE 的“combine expert outputs”并不是 Python 侧做的，而是 finalize kernel 负责的。

## 8. 与仓库内 `solution/triton/kernel.py` 的对照

`solution/triton/kernel.py` 不是 FlashInfer 的源码，但它几乎把 FlashInfer baseline 的语义显式重写了一遍。

### 8.1 路由语义几乎一一对应

`kernel.py:212-273` 的 `_build_local_plan()` 对应 FlashInfer 的 DeepSeek routing:

- `sigmoid(logits)` -> `s`
- `s + bias`
- group top-2 求和
- 选 top groups
- 再选 top experts
- 用 `sigmoid` 分数归一化
- 只保留本地 `local_expert_offset : local_expert_offset + NUM_LOCAL_EXPERTS` 范围内的 expert

### 8.2 它把底层隐式反量化显式写出来了

FlashInfer 底层是在专用 kernel / grouped GEMM 里处理 FP8 + block-scale。

而 `kernel.py` 显式做了:

- `_dequant_selected_hidden_states()` -> 反量化选中的 token
- `_dequant_selected_w13_t()` -> 反量化选中的 GEMM1 权重
- `_dequant_selected_w2_t()` -> 反量化选中的 GEMM2 权重

这让语义更容易理解，但性能上显然更重。

### 8.3 它用 `aten._grouped_mm` + Triton SwiGLU 近似底层流水线

对应:

- `kernel.py:379` GEMM1
- `kernel.py:380` SwiGLU
- `kernel.py:381` GEMM2
- `kernel.py:382` `index_add_` 回写最终输出

这正好对应 FlashInfer C++ runner 的:

- `mPermuteGemm1.run`
- `activation::run`
- `mGemm2.run`
- `finalize::run`

### 8.4 它和 baseline/FlashInfer 的关键差异

`solution/triton/kernel.py` 更像“功能等价实现”，但不是“性能等价实现”。主要差异有:

- 它显式反量化 selected weights / hidden states
- 它没有 FlashInfer 的 routing fused kernel
- 它没有 FlashInfer 的 CTA 调度/packed topk/PDL 机制
- 它没有 FlashInfer 的 AutoTuner tactic 选择
- 它的权重缓存只是在 Python 层缓存 active experts 的 dequantized pack
- 它的最终合并使用 `index_add_`，而不是底层 finalize kernel

因此它非常适合“读懂 baseline 在做什么”，但不代表能接近 baseline 的真实性能。

## 9. 为什么 `baseline.py` 很短却仍然很强

### 9.1 因为它把高价值工作都外包给了 FlashInfer

真正昂贵且关键的部分都不在 Python:

- DeepSeekV3 fused routing
- routing 到 grouped GEMM 的调度数据生成
- TensorRT-LLM batched GEMM tactic 选择
- FP8 block-scale 专用路径
- finalize/scatter kernel

### 9.2 因为它保留了底层最喜欢的数据格式

`baseline.py` 没有把 FP8 权重/激活变成 BF16 后再算，而是直接把:

- FP8 data
- FP32 block-scale

原样交给底层专用 kernel。

### 9.3 因为它走的是针对 Blackwell/SM100 的专门实现

从 `gen_trtllm_gen_fused_moe_sm100_module()` 可以看出，这条路径是专门给 SM100/Blackwell 准备的，不是通用 fallback。

## 10. 对后续 Triton 重写最有价值的结论

如果目标是自己写 `solution/triton/kernel.py` 的高性能版本，而不是继续调用 FlashInfer，那么最值得借鉴的是这些语义与系统设计点:

1. 路由必须复现 DeepSeekV3 规则，而不是普通 softmax top-k。
2. 选 expert 用 `sigmoid + bias`，算 combine weight 用纯 `sigmoid` 归一化。
3. 只做本地 expert 的计算，`local_expert_offset/local_num_experts` 是分布式切分边界。
4. 尽量避免“全量反量化全部 expert 权重”；最好只对 active local experts 工作。
5. 真正高性能的关键不只是 GEMM，而是 routing、permutation、grouped scheduling、finalize 整条链路。
6. DeepSeek FP8 路径下，SwiGLU 很可能需要独立阶段，而不是默认假设能完全 fuse 进 FC1。

## 11. FlashInfer 底层是怎样把 MoE 变快的

这一节不再停留在“调用了哪个函数”，而是拆开解释 FlashInfer 底层到底做了哪些性能关键设计。

### 11.1 先把动态稀疏问题变成规则的 tile 问题

MoE 最大的麻烦不是 GEMM 本身，而是:

- 每个 token 去的 expert 不同
- 每个 expert 收到的 token 数量不均匀
- 不均匀 batch 会让 Tensor Core kernel 很难高效利用

FlashInfer 的第一步不是立刻做 matmul，而是先把 routing 结果转成“适合 grouped GEMM 的规则批处理计划”。

具体证据在 `trtllm_fused_moe_kernel_launcher.cu:258-305`:

- 分配 `expanded_idx_to_permuted_idx`
- 分配 `permuted_idx_to_token_idx`
- 分配 `expert_count_histogram`
- 分配 `cta_idx_xy_to_batch_idx`
- 分配 `cta_idx_xy_to_mn_limit`
- 分配 `num_non_exiting_ctas`
- 记录 `workspace.total_max_padded_tokens`
- 记录 `workspace.ProjUpTileN`

这些张量不是“附带产物”，而是后续所有高吞吐执行的核心计划表:

- `expanded_idx_to_permuted_idx`
  - 把 `(token, topk_slot)` 映射到按 expert 打包后的连续位置
- `permuted_idx_to_token_idx`
  - 反向记录打包后位置对应原 token
- `expert_count_histogram`
  - 统计每个 expert 收到多少 token
- `cta_idx_xy_to_batch_idx` / `cta_idx_xy_to_mn_limit`
  - 直接给 grouped GEMM 提供 CTA 级别的批次映射和边界
- `num_non_exiting_ctas`
  - 让后续 kernel 能高效跳过空 CTA

为什么这很重要:

- 朴素实现常常是 “先 topk，再按 token 做小 matmul，再 scatter 回去”
- FlashInfer 不是以 token 为中心执行，而是把 token 先重排成按 expert 聚集的连续块，再让后续 kernel 像处理规则 batch 一样处理

这一步本质上把“动态稀疏计算”改写成了“规则化的分块批处理计算”。

### 11.2 `tile_tokens_dim` 不是写死的，而是按负载自适应选的

`trtllm_fused_moe_kernel_launcher.cu:85-106` 的 `computeSelectedTileN(...)` 会根据

- `num_tokens`
- `top_k`
- `num_local_experts`

估算平均每个 local expert 的 token 数:

- `avg_tokens_per_expert = num_tokens * top_k / num_local_experts`

然后:

1. 取它的下一个 2 的幂
2. clamp 到支持的 tile 集合
3. 不只测一个 tile，还会把相邻几个 tile 一起纳入候选

这意味着 FlashInfer 并不是固定假设 “每个 expert token 数差不多” 或者 “某个 tile 永远最优”。

它的策略是:

- 先用 cheap heuristic 估计合理 tile
- 再把相邻 tile 一并纳入 tactic 候选
- 让后续 autotuner / config selector 选最合适的那一个

为什么快:

- 对 MoE 来说，`seq_len` 和 active expert 分布在不同 batch 间变化很大
- 单一 tile 往往会在某些 batch 上严重 under-utilize SM
- 自适应 tile 可以减少 padding 浪费和 CTA 空转

### 11.2.1 这部分原先报告说得还不够精确，这里补上真实公式

`runner.h:81-113` 给了 routing workspace 上界的明确公式。

#### CTA 上界

`getMaxNumCtasInBatchDim(numTokens, topK, numExperts, tileTokensDim)` 的思路是:

1. 先把 expanded token 数记为:
   - `numRemainingTokens = numTokens * topK`
2. 先给尽可能多的 expert 各分 1 个 token:
   - `numExpertsFilled = min(numExperts, numRemainingTokens)`
   - `maxNumCtasInBatchDim += numExpertsFilled`
3. 剩余 token 再按 `tileTokensDim` 贪心装满 CTA tile:
   - `maxNumCtasInBatchDim += numRemainingTokens / tileTokensDim`

它不是平均值估计，而是一个“为了提前分配 workspace 的保守上界”。

#### Permuted/padded token 上界

`getMaxPermutedPaddedCount(...)` 直接定义为:

- `maxPermutedPaddedCount = getMaxNumCtasInBatchDim(...) * padding`

这里的 `padding` 实际上就是 `tile_tokens_dim`。

对后续自研 Triton kernel 的启发是:

- FlashInfer 宁愿多分配一些排列后的 token 空间，也要保证后面 grouped GEMM 和 finalize 都能在固定 tile 假设下运行
- 如果我们只按“真实 active token 数”做最小化分配，往往会让后面的执行组织退化

### 11.3 Routing kernel 本身就做了非常多事，而不是只输出 top-k

在 DeepSeek 路径下，`trtllm_fused_moe_runner.cu:52-101` 会调用:

- `moe::dev::routing::routingDeepSeek::run(...)`

对应 CUDA 实现在:

- `trtllm_fused_moe_routing_deepseek.cu`

这部分至少做了四层工作:

1. 对 router logits 做 `sigmoid`
2. 加 bias，做分组 top-k
3. 只保留本机 local experts
4. 同时构造后续 grouped GEMM 所需的 permutation / histogram / CTA 调度信息

从 `routing_deepseek.cu:97-169` 可见:

- 用 shared memory 保存 `sigmoid` 分数和加 bias 后分数
- 每个 warp 对应一个 expert group
- 先做组内 top-2，再做组间 top-group，再做最终 top-expert

从 `routing_deepseek.cu:236-263` 可见:

- 在同一个 kernel 里直接算出最终 `finalScore`
- 直接写 packed topk 结果或 topk weights

这说明 routing 不是一个“只负责找下标”的轻量操作，而是一个融合了:

- 分数变换
- group top-k
- 权重归一化
- local expert 过滤
- 输出布局准备

的前端管线。

### 11.4 Routing 后半段还会根据规模选择不同执行模式

`routing_deepseek.cu:620-684` 很值得注意。

它并不是只写了一种 routing permutation kernel，而是有多种执行模式:

- `routingIndicesClusterKernel`
- `routingIndicesCoopKernel`
- `routingIndicesHistogramKernel`
- `routingIndicesOffsetsKernel`

选择逻辑大致是:

1. 如果能用 single-cluster，就走 cluster kernel
2. 如果 token 数不大，走 cooperative launch kernel
3. 否则退回两步式 histogram + offsets kernel

这背后的设计含义是:

- 小规模或适中规模时，用更强同步语义的 cooperative/cluster 路径，减少中间访存和 launch 开销
- 更大规模时，退回更稳妥的两阶段方法，避免 cooperative launch 的规模约束

为什么快:

- 同一套 kernel 模式并不能覆盖所有 token 规模
- FlashInfer 明确做了规模分层，按 problem size 选不同组织方式
- 这是典型的系统层优化，不是单个数学 kernel 能解决的

### 11.5 PDL 让阶段间依赖更轻

在多个 kernel 中都能看到:

- `cudaTriggerProgrammaticLaunchCompletion()`
- `cudaGridDependencySynchronize()`

出现位置包括:

- `routing_deepseek.cu:89-94`
- `routing_deepseek.cu:280-285`
- `activationDeepSeekKernel` 中的 `dev_kernel.cu:214-219`
- finalize kernel 中的 `dev_kernel.cu:643-647`, `855-859`, `912-916`

这意味着 FlashInfer 在 SM90+/SM100 上利用了 PDL 风格的设备端依赖协调。

基于源码的工程推断:

- 这样做可以减少 host 侧频繁同步/串行 launch 带来的气泡
- 某些后继 kernel 不必完全等 host 再次介入，就能在设备端等待前序结果并继续
- 对像 MoE 这种多阶段流水线，PDL 有利于把 routing / activation / finalize 的阶段衔接做得更紧

### 11.6 一个容易误判的点: 这条 baseline 路径里并没有显式 scale-layout 转换开销

前面报告提到 `convertSfData`，但继续往下查后可以更精确地说:

- `trtllm_fused_moe_runner.cu:466-476` 确实准备了 `convertSfData`
- 但在 `Runner::run(...)` 的实际执行路径里，没有看到 `moe::dev::convertsf::run(convertSfData, ...)`
- 反而在 `586` 处直接令:
  - `void* hidden_states_scale_linear{args.hidden_states_scale};`
- 随后把这个指针直接传给 `mPermuteGemm1.run(...)`

这意味着至少在当前 `flashinfer 0.6.6` 的 DeepSeek FP8 block-scale baseline 路径上:

- `hidden_states_scale` 没有额外走一次显式 layout conversion kernel
- `convertSfData` 更像为别的路径或未来路径预留的基础设施

这对后续优化很重要，因为它能避免我们把精力花在一个并不处于关键路径的点上。

### 11.7 `MoERunnerArgs` / `MoEWorkspace` 把流水线结构写死在数据结构里

这一点对“怎么重写才可能快”很关键，之前报告提到得还不够细。

来自 `runner.h:265-358`:

#### `MoERunnerArgs` 暴露出的关键信息

- `routing_logits`
- `routing_bias`
- `hidden_states`
- `hidden_states_scale`
- `gemm1_weights` / `gemm1_weights_scale`
- `gemm2_weights` / `gemm2_weights_scale`
- `num_tokens`
- `num_experts`
- `hidden_size`
- `hidden_size_output`
- `top_k`
- `n_group`
- `topk_group`
- `routed_scaling_factor`
- `intermediate_size`
- `local_expert_offset`
- `local_num_experts`
- `mUseRoutingScalesOnInput`
- `mUseDeepSeekFp8`
- `output1_scales_scalar`
- `output1_scales_gate_scalar`
- `output2_scales_scalar`
- `do_finalize`

其中对 baseline 最关键的策略位是:

- `mUseDeepSeekFp8`
- `mUseRoutingScalesOnInput`
- `do_finalize`

#### `MoEWorkspace` 暴露出的关键信息

`MoEWorkspace` 明确分成几类缓冲区:

1. Routing 产物
   - `routing_expert_indexes`
   - `expanded_idx_to_permuted_idx`
   - `permuted_idx_to_token_idx`
   - `expert_weights`
   - `cta_idx_xy_to_batch_idx`
   - `cta_idx_xy_to_mn_limit`
   - `num_non_exiting_ctas`
   - `total_num_padded_tokens`
   - `total_max_padded_tokens`
2. 中间表示
   - `gemm1_output`
   - `gemm1_output_scale`
   - `activation_output`
   - `activation_output_scale`
   - `gemm2_output`
   - `gemm2_output_scale`
3. 调度辅助
   - `ProjUpTileN`
   - `bmm1_workspace`
   - `bmm2_workspace`

这说明 FlashInfer 不是把 MoE 当成单 kernel，而是当成一条依赖显式 workspace 的多阶段执行图。

## 12. GEMM 子系统为什么快

### 12.1 它不是自己写 matmul，而是复用 TensorRT-LLM 导出的 batched GEMM 内核族

证据:

- `jit/fused_moe.py:215-286` 会下载/缓存 `flashinferMetaInfo.h`
- `trtllm_batched_gemm_runner.cu` 使用 `BatchedGemmInterface`

从 `trtllm_batched_gemm_runner.cu:86-119` 可以看出:

- 它会遍历 `BatchedGemmInterface` 提供的所有 config
- 按 dtype / routeAct / fusedAct / tileSize / useShuffledMatrix / layout 等条件筛 passing configs

也就是说 FlashInfer 并不是“运行时现场生成一个 kernel”。

更准确地说，它是在一个预先准备好的高性能 kernel family 里，为当前问题形状筛选、排序、验证合适 config。

### 12.2 它把 FC1 和 FC2 映射成不同属性的 grouped GEMM

从 `trtllm_fused_moe_runner.cu:227-269` 和 `349-365` 可见:

- FC1 `PermuteGemm1`
  - `routeAct = true`
  - gated activation 场景下可配置 `actType`
  - `transposeMmaOutput = true`
  - DeepSeek FP8 时 `fusedAct = !useDeepSeekFp8`，也就是禁用 fused activation
  - `epilogueTileM = 64`（DeepSeek FP8）或 `128`
- FC2 `Gemm2`
  - `routeAct = false`
  - `fusedAct = false`
  - 同样 `transposeMmaOutput = true`
  - DeepSeek FP8 时也是较小 `epilogueTileM = 64`

这里至少有三个性能信号:

1. FC1 和 FC2 分别建模
   - 因为两者输入输出结构不同，不能简单复用同一配置
2. `transposeMmaOutput = true`
   - 说明底层 GEMM 数据流是围绕某种更适合 Tensor Core/epilogue 的存储方式组织的
3. `epilogueTileM` 会根据 DeepSeek FP8 改变
   - 这通常意味着针对 FP8 block-scale/输出量化路径，epilogue 的 tile 粒度也需要专门调节

### 12.3 它不是盲目 autotune，而是“先筛，再排序，再验”

`trtllm_batched_gemm_runner.cu:323-420` 的 `getValidConfigIndices()` 很关键。

它对通过初筛的 config 做排序时使用了分层 heuristic:

1. 优先考虑 `tileK`
   - 如果 `K < tileK`，优先较高利用率
   - 否则优先较大 `tileK`
2. 再考虑 `mUseUnrollLoop2xForMma`
3. 再比较 `tileM`
4. 再比较 `tileN`
5. 再根据估计 CTA 数与 SM 数关系，偏向 persistent scheduler

这说明 FlashInfer 的 tactic 选择不是只靠黑盒 benchmark。

它先利用问题形状和硬件特征进行有约束的排序，再由 `isValidConfig` 过滤非法 config。

为什么快:

- 大幅缩小了要考虑的 kernel 空间
- 避免把明显不合适的 kernel 拉进运行时试错
- 对动态形状场景，能降低 autotune 成本和 cache 污染

### 12.4 `prepare_moe_common()` 做的是“Runner + tactic + workspace”一体化准备

在 `trtllm_fused_moe_kernel_launcher.cu:322-356`:

- 根据 dtype / quantization path 构造 `MoE::Runner`
- 取默认或指定 tactic
- 验证 tactic 是否仍在当前 valid config 集合中
- 分别为 FC1 / FC2 分配 workspace

源码里甚至专门处理了一个重要细节:

- `324-330` 对 DeepSeek FP8 block-scale 路径，使用 weights-only Runner constructor，以匹配原始 kernel path 和 numerics

这说明 FlashInfer 很重视“同一条数值路径”的一致性，不愿意为了代码统一而破坏底层 kernel 原始行为。

### 12.5 它会故意把 padded token 数抬高到一个最小阈值，换吞吐

这一点是之前报告里缺失的关键工程细节。

`runner.h:55-59` 定义了:

- `maybeGetMinTokenCount(numPaddedTokens, hiddenSize, dtypeSizeBits)`

公式是:

- `minNumTokensRequired = divUp(128 * 1024 * 8, hiddenSize * dtypeSizeBits)`
- 返回 `max(numPaddedTokens, minNumTokensRequired)`

直白解释:

- 如果实际 padded token 太少，导致总工作集不到 128 KiB，就会人为抬高到至少 128 KiB 对应的 token 数

这个函数在 DeepSeek FP8 block-scale path 的 `prepare_moe()` 中被真正使用了:

- `trtllm_fused_moe_kernel_launcher.cu:998-1007`

分别用于:

- `max_num_padded_tokens_gemm1`
- `max_num_padded_tokens_gemm2`

这说明 FlashInfer 为了吞吐，会接受额外 padding 和中间空间开销。

对后续自研 kernel 的启发是:

- 如果我们只按“真实 token 数”做极致精简，短序列场景可能天然吃亏
- 让工作集达到更适合 Tensor Core/SM 调度的规模，可能本身就是 speedup 来源

## 13. DeepSeek FP8 路径的专门优化

### 13.1 激活不是简单 `SwiGLU(x)`，而是“反量化 + 激活 + 重新量化”合成阶段

`activationDeepSeekKernel` 位于:

- `trtllm_fused_moe_dev_kernel.cu:202-335`

它做的事情远多于普通 activation:

1. 读取 GEMM1 输出的两个半段
2. 读取各自 block scale
3. 先反量化成 `float`
4. 计算 `silu(x2) * x1`
5. 对输出块做绝对值最大值归约
6. 计算新的输出 scale
7. 再把结果量化回输出 dtype，并把新 scale 写出

关键代码点:

- `268-278` 读两路输入及其 scale
- `283-288` 反量化并做 SwiGLU
- `291-313` block reduce 求 amax，生成新的输出 scale
- `328-330` 用 `out / scaleOut` 写回量化输出

这说明 DeepSeek FP8 路径根本不是“先 GEMM1 得到 BF16，再调个 SwiGLU”。

它是:

- FP8 block-scale 中间结果
- 局部反量化
- 激活
- 再量化

全部在专用 activation kernel 中完成。

为什么快:

- 避免把整个中间张量扩成更高精度长期驻留
- 输出马上恢复成更紧凑格式，减轻写带宽和后续 GEMM2 输入带宽
- amax/scale 直接在 kernel 内归约得到，无需额外 pass

### 13.2 它还会按 token 密度调节 activation kernel 的并行方式

在 `dev_kernel.cu:347-360`，DeepSeek activation 会根据:

- `numSms`
- `numTokens`
- `topK`
- 输出维度对应的 scale block 数

估计工作量，然后选 `numTokensPerCta = 1 / 2 / 4`。

这意味着 activation 也不是固定 launch policy，而是根据当前 problem size 自适应决定一个 CTA 同时处理几个 token。

为什么快:

- 小 batch 时避免 block 过重导致 occupancy 差
- 大 batch 时提高 CTA 复用和吞吐

### 13.2.1 这条路径的中间张量 dtype 也很值得记住

在 DeepSeek FP8 path 的 `prepare_moe()` 里:

- `gemm1_output` 分配成 `uint8`
- `gemm1_output_scale` 分配成 `float32`
- `activation_output` 分配成 `uint8`
- `activation_output_scale` 分配成 `float32`
- `gemm2_output` 才是 `bfloat16`

证据:

- `trtllm_fused_moe_kernel_launcher.cu:1009-1034`

这说明前半段流水线的真实中间表示不是 BF16，而是:

- 量化值 + block-scale

这也是 baseline 快的重要原因之一，因为它避免了 FC1 后就扩成大块 BF16 激活。

### 13.3 Finalize 也针对不同场景分路径

`trtllm_fused_moe_dev_kernel.cu:638-1003` 展示了 finalize 的三条路径:

1. `finalizeKernel`
   - 普通标量式合并
2. `finalizeKernelVecLoad`
   - 向量化 load + topk unroll 的高速路径
3. `finalizeDeepSeekKernel`
   - DeepSeek FP8 路径，融合了 block-scale 和输出重标定

#### DeepSeek FP8 finalize

`904-968` 的 `finalizeDeepSeekKernel` 会:

- 按 token / hidden 遍历
- 对所有 top-k expert 结果做加权求和
- 同时读取每个 expert 输出的 block scale
- 计算结果块的 amax
- 写回新的输出 scale
- 再量化输出

也就是说，它把:

- combine expert outputs
- block scale 应用
- 输出重标定

融合到了同一个 finalize kernel 中。

#### 非 DeepSeek 路径的向量化 finalize

`802-900` 的 `finalizeKernelVecLoad` 则进一步做了:

- 128-bit 对齐/向量化读写
- `TopKUnrollFactor` 展开
- 把 `permutedIdx` 和 `scale` 先搬到 shared memory
- 用 `float4` PTX load 读输入

而在 `971-1000`，它还根据 grid 大小选择是否使用 vectorized finalize。

代码里的解释非常直白:

- 当 block 数较多、会出现多 wave 时，倾向用 vectorized loading kernel

为什么快:

- 把 combine/scatter 这种常被忽视的“尾部操作”也做成高吞吐 kernel
- 避免最终阶段成为整条流水线的瓶颈

## 14. 从系统角度看，FlashInfer 为什么明显快于语义等价重写

### 14.1 它避免了“大量中间张量在高精度下长期存活”

朴素重写常见做法:

- 先把输入/权重反量化成 BF16
- 做 matmul
- 保留大中间张量
- 再做 activation / combine

FlashInfer 的做法则更接近:

- routing 先做重排和调度
- grouped GEMM 尽量直接消费压缩格式
- activation/finalize 在需要时局部反量化并立刻重新量化

好处是:

- 带宽压力更低
- 显存占用更小
- cache 命中更友好

### 14.2 它把多个“低算强度但高开销”的步骤都 kernel 化了

不仅 matmul 是专用实现，下面这些也是:

- grouped routing
- permutation
- activation
- finalize
- scale layout 处理基础设施

而且这些 kernel 之间共享同一套 permutation / padded-token / CTA 计划。

这和“数学上做对了”完全不是一个层面的优化。

### 14.3 它真正做到了“路由感知的 GEMM”

从 `PermuteGemm1` 的 options 可以看到:

- `routeAct = true`

从 batched GEMM runner 的输入也能看到:

- `routeMap`
- `totalNumPaddedTokens`
- `ctaIdxXyToBatchIdx`
- `ctaIdxXyToMnLimit`
- `numNonExitingCtas`

这说明 GEMM kernel 并不是对一堆普通连续矩阵做乘法，而是显式感知 routing 后的批次结构。

换句话说，FlashInfer 的 GEMM 不是“MoE 外部的通用 GEMM”，而是“MoE 调度信息驱动的 batched GEMM”。

### 14.4 它的优化单位是整条流水线，而不是某一个 kernel

从源码可以看到至少有这些跨阶段协同:

- routing 决定 padded token 布局
- padded token 布局决定 CTA 批次调度
- CTA 调度决定 grouped GEMM 的 config 选择
- GEMM1 输出格式影响 activation 是否能 fuse
- activation 输出 scale 直接为 GEMM2 服务
- finalize 直接消费前面所有阶段的 permutation 和 scale

因此 FlashInfer 快，不是因为“其中某个 kernel 特别神”，而是因为它把 MoE 看作一条端到端的数据流问题。

## 15. 这份报告此前仍然偏粗的地方，现在补成对提速更有用的结论

这一节专门面向“后面怎么提升 speedup”，总结哪些地方以前说得太粗，现在哪些已经补齐。

### 15.1 之前太粗: 只说了有 padding，没有说 padding 怎么算

现在补齐为:

- `max_num_ctas` 的上界公式
- `max_permuted_padded_count = max_ctas * tile_tokens_dim`
- `maybeGetMinTokenCount` 的 128 KiB 最小工作集规则

这三条直接决定:

- 中间缓冲区大小
- 是否值得强行 padding
- 小 batch 时为什么 baseline 仍能保持较高吞吐

### 15.2 之前太粗: 只说了有 workspace，没有拆字段含义

现在补齐为:

- `MoERunnerArgs` 明确告诉我们有哪些策略开关
- `MoEWorkspace` 明确告诉我们各阶段是靠哪些索引和缓冲区耦合起来的

这使得后续优化不再是“看到哪里慢就局部改”，而是能按数据流结构来重构。

### 15.3 之前太粗: 把 scale layout conversion 也写进了可能路径

现在补齐为:

- 当前 baseline 的 DeepSeek FP8 路径里，没有看到显式调用 `convertsf::run`
- `hidden_states_scale` 直接被当作 `hidden_states_scale_linear` 传给 GEMM1 路径

这能避免后续把时间花在不存在的热点上。

### 15.4 之前太粗: 没把“已证实事实”和“待验证假设”分开

当前已经可以视为“源码证实”的结论:

- DeepSeekV3 routing 规则
- routing/permutation/CTA workspace 结构
- grouped GEMM config 的筛选和排序逻辑
- activation/finalize 的 DeepSeek FP8 特化路径
- 128 KiB 最小 token 数启发式

当前仍然属于“需要 profile/实验验证”的优化假设:

- `use_shuffled_weight=False` 是否已经明显限制了上界性能
- `mUseRoutingScalesOnInput` 改成输入前乘后，能否显著减轻 finalize 压力
- 哪一种 `tile_tokens_dim` 在你的 workload 分布上最有利
- 哪一段最值得优先优化: routing、grouped GEMM、activation 还是 finalize

## 16. 面向下一步 speedup 的直接行动建议

如果这份报告的读者是“后面要继续写更快算子”的我，那么最值得优先盯的点不是均匀的。

### 16.1 第一优先级: 复刻 routing -> permutation -> CTA 调度语义

理由:

- 这是 FlashInfer 把 MoE 变成规则批处理的核心
- 一旦这层不对，后续 GEMM 再快也会被 irregular access 拖垮

最需要模仿的产物是:

- `expanded_idx_to_permuted_idx`
- `permuted_idx_to_token_idx`
- `cta_idx_xy_to_batch_idx`
- `cta_idx_xy_to_mn_limit`
- `num_non_exiting_ctas`

### 16.2 第二优先级: 不要过早把中间结果扩成 BF16

理由:

- baseline 的前半段真实中间格式仍然是 `uint8/fp8 + float32 scale`
- 这对带宽和中间显存压力都更友好

如果我们的 Triton kernel 现在是:

- FC1 后直接得到 BF16
- SwiGLU 后仍保留 BF16

那它大概率天然吃亏。

### 16.3 第三优先级: 短序列场景要认真对待“最小工作集”策略

理由:

- `maybeGetMinTokenCount` 表明 FlashInfer 愿意牺牲一部分额外 padding 来换硬件利用率
- 这在小 batch / 短序列场景很可能是非常关键的 speedup 来源

### 16.4 第四优先级: finalize 不能只当尾巴，它可能是竞争力差距的一大来源

理由:

- baseline 的 finalize 不只是一个简单 `index_add_`
- 它有向量化加载、top-k unroll、DeepSeek FP8 特化和输出重标定

如果我们后续 kernel 末尾还是:

- Python 侧 `index_add_`
- 或一个很普通的 scatter/reduce kernel

那即使 GEMM 本体接近，整体 speedup 也可能追不上。

### 16.5 第五优先级: 两个值得单独做实验的优化旋钮

1. `use_shuffled_weight`
   - baseline 明确传的是 `False`
   - 但别的路径和部分默认实现倾向 `true`
   - 它可能带来更好的权重访存/布局匹配
   - 代价是需要准备匹配布局的输入权重

2. `mUseRoutingScalesOnInput`
   - 从 `finalizeData.expertWeightsPtr = nullptr` 的分支可以看出，系统支持把 routing scale 作用前移
   - 这可能减少 finalize 的乘法和读权重开销
   - baseline 当前没有开启

注意:

- 这两点都不能直接从源码推断“打开就一定更快”
- 但它们已经是这份报告里能明确指认出来、值得后续实验验证的优化旋钮

## 17. 最终结论

`solution/triton/baseline.py` 不是一个“baseline kernel 实现”，而是一个“baseline 调用入口”。它的角色是:

- 固定这条比赛 workload 的结构参数
- 校验输入 shape
- 把 router/log-scale 张量整理成 FlashInfer 需要的 dtype/layout
- 触发 `flashinfer` 的 DeepSeekV3 FP8 block-scale fused MoE 运行时

如果你的目标是“理解 baseline 背后的真实算法”，最值得对照阅读的是:

- `solution/triton/baseline.py`
- `solution/triton/kernel.py`
- `flashinfer/fused_moe/core.py`
- `flashinfer/data/csrc/trtllm_fused_moe_runner.cu`
- `flashinfer/data/csrc/trtllm_fused_moe_kernel_launcher.cu`
- `flashinfer/data/csrc/trtllm_fused_moe_routing_deepseek.cu`

其中:

- `baseline.py` 说明入口参数是什么
- `kernel.py` 说明语义长什么样
- FlashInfer C++/CUDA 源码说明真正高性能实现是如何组织的
