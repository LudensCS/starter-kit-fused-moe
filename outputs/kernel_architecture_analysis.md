# `solution/triton/kernel.py` 结构剖析报告

生成时间: 2026-04-18 UTC

目标文件: `solution/triton/kernel.py`

## 1. 一句话总览

这个 `kernel.py` 实现的是一个“先做路由与分桶，再做两段局部 expert GEMM，最后回写聚合”的 Triton 版 fused MoE。

它的整体执行顺序是:

1. 对每个 token 做 DeepSeek-V3 no-aux routing
2. 只保留本 rank 对应的 32 个 local experts
3. 把 `(token, local_expert, weight)` 重新整理成按 expert 连续的工作队列
4. 对每个 local expert 做 `GEMM1 -> SwiGLU`
5. 对每个 local expert 做 `GEMM2 -> 按 routing weight 累加回 token 输出`

从架构上看，它不是“一个大 kernel 把所有事情一次做完”，而是“多个小 kernel/包装函数串起来的流水线”。

## 2. 顶层调用图

对应代码位置:

- 常量与 tile 参数: `7-30`
- 路由 kernel: `33-112`
- prefix sum kernel: `114-124`
- token/expert 分桶 kernel: `126-169`
- GEMM1 + SwiGLU kernel: `172-283`
- GEMM2 + scatter-accumulate kernel: `286-369`
- Python 包装层:
  - `_as_cuda_contiguous`: `372-379`
  - `_routing_and_scatter`: `382-457`
  - `_gemm1_swiglu_triton`: `460-515`
  - `_gemm2_scatter_triton`: `518-566`
  - `run`: `569-699`

整体调用关系:

```text
run(...)
  -> _routing_and_scatter(...)
       -> _routing_topk_kernel
       -> _prefix_sum32_kernel
       -> _scatter_local_kernel
  -> _gemm1_swiglu_triton(...)
       -> _gemm1_swiglu_kernel
  -> _gemm2_scatter_triton(...)
       -> _gemm2_scatter_kernel
  -> 返回 output
```

## 3. 这份代码在算什么

它实现的数学逻辑和 definition/reference 基本一致:

1. `routing_logits` 做 `sigmoid`
2. 加 `routing_bias`
3. 按 8 个 group 分组，每组 32 个 experts
4. 每组取 top-2 分数求和，选 top-4 groups
5. 只在这 4 个组内做全局 top-8 expert 选择
6. 最终归一化权重用的是 `sigmoid(logits)`，不是 `sigmoid(logits)+bias`
7. 对于本地 expert:
   - `hidden_states * hidden_states_scale`
   - `gemm1_weights * gemm1_weights_scale`
   - 计算 `GEMM1`
   - 做 `SwiGLU`
   - `gemm2_weights * gemm2_weights_scale`
   - 计算 `GEMM2`
   - 用 routing weight 累加回 token 输出

## 4. 关键输入输出与中间张量

### 4.1 固定几何

来自 `7-18`:

- `NUM_EXPERTS_GLOBAL = 256`
- `NUM_LOCAL_EXPERTS = 32`
- `TOP_K = 8`
- `N_GROUP = 8`
- `TOPK_GROUP = 4`
- `HIDDEN_SIZE = 7168`
- `INTERMEDIATE_SIZE = 2048`
- `BLOCK_SIZE = 128`

这意味着:

- 每组 experts 数量: `256 / 8 = 32`
- `hidden_states_scale` 的 block 数: `7168 / 128 = 56`
- `gemm1` 输出维度是 `4096 = 2 * 2048`
- `gemm2` 输入维度是 `2048`

### 4.2 关键中间张量

在 `_routing_and_scatter` 和 `run` 中分配:

- `topk_ids`: `[T, 8]`
  - 每个 token 选出来的 8 个全局 expert id
- `topk_weights`: `[T, 8]`
  - 对应 8 个 expert 的归一化权重
- `counts`: `[32]`
  - 每个 local expert 被选中的 token-expert pair 数
- `offsets`: `[32]`
  - 每个 local expert 在重排缓冲区中的起始位置
- `sorted_token`: `[T * 8]`
  - 重排后的 token id
- `sorted_weight`: `[T * 8]`
  - 重排后的 routing weight
- `inter`: `[T * 8, 2048]`
  - 每个选中 pair 经 `GEMM1 + SwiGLU` 后的激活
- `output_buffer`: `[T, 7168]`
  - 最终输出缓冲区

要注意:

- `sorted_token`/`sorted_weight` 的物理长度总是 `T * TOP_K`
- 其中只有落在本地 expert 范围内的 pair 会被真正写入
- `counts` 和 `offsets` 决定了哪些区段有效

## 5. 常量区与调参入口

### 5.1 算法常量

`7-18` 是不能随便改的定义级参数。

如果你改这里，相当于改了算子的语义契约，不只是“优化”。

### 5.2 Triton tile 参数

`20-30` 是最常见的手动调参入口:

- `GEMM1_BLOCK_M/N/K`
- `GEMM1_NUM_WARPS/STAGES`
- `GEMM2_BLOCK_M/N/K`
- `GEMM2_NUM_WARPS/STAGES`

这些参数直接决定:

- 每个 Triton program 处理多少行 token-pairs
- 每次处理多少输出列
- K 维切块大小
- 寄存器压力
- shared memory 使用量
- occupancy

如果你想先做性能调优，通常先从这里开始，而不是先改算法逻辑。

## 6. `_routing_topk_kernel` 详细剖析

代码位置: `33-112`

### 6.1 这个 kernel 干了什么

它是“每个 token 一个 program”的路由 kernel。

`pid = tl.program_id(0)` 表示当前 program 对应一个 token。

它完成了三件事:

1. 为当前 token 计算 top-k routing 结果
2. 统计当前 token 是否命中本地 32 个 experts，并原子加到 `counts`
3. 顺手把该 token 对应的输出行清零

第三点很容易忽略: 输出初始化不是单独的 memset，而是融合在路由 kernel 末尾的 `104-111`。

### 6.2 路由选择过程

关键逻辑:

- `58-61`
  - 读取 256 个 logits 和 256 个 bias
  - 计算 `sig = sigmoid(logits)`
  - 计算 `score = sig + bias`

- `62-67`
  - reshape 成 `[8, 32]`
  - 每组取 top-1 和 top-2，求和得到 `group_scores`

- `69-75`
  - 通过 4 次迭代 max，选出 top-4 groups

- `77-95`
  - 只在选中 group 里继续选 top-8 experts
  - 存 `topk_ids`
  - 暂时把 `sig` 存进 `topk_weights`
  - 同时对命中的 local expert 做 `atomic_add(count_ptr + local, 1)`

- `97-102`
  - 对 `topk_weights` 做归一化并乘 `routed_scaling_factor`

### 6.3 tie-break 处理

`55-56`:

- `expert_tiebreak = expert_offs * -1e-6`
- `group_tiebreak = group_offs * -1e-4`

作用:

- 避免 `tl.max` 在完全相等分数下选不稳定
- 让较小下标的 group/expert 略占优

这是一个很重要的小细节。

如果你改 routing 逻辑，最好保留 deterministic tie-break，不然 reference 对比时容易出现“只有 top-k id 不同，但误差看起来很奇怪”的情况。

### 6.4 为什么 `topk_weights` 先存 `expert_sig`

`87-89` 先把每个专家对应的 `sig` 存进去，后面 `97-102` 再统一归一化。

这样恰好对应 reference:

- 选 expert 用 `sig + bias`
- 算权重用 `sig`

这是 DeepSeek-V3 no-aux routing 的关键语义。

### 6.5 输出清零被融合进来了

`104-111` 对 `[T, H]` 输出的当前 token 行做清零。

这意味着:

- `output` 在进入后续 GEMM2 scatter 前已经是 0
- 不需要额外的 `torch.zeros`
- 但也意味着 routing kernel 除了路由，还承担了“输出初始化”的副作用

如果你以后想把 routing 换成预计算 topk ids/weights 的路径，这段初始化逻辑别忘了补回来。

### 6.6 这个 kernel 的修改风险

高风险点:

- `65` 和 `73/85` 里的相等判断基于 float
- `90-93` 中 local expert 范围判断不能错
- `97-102` 的归一化必须继续用 `sig`，不能误用 `score`
- `104-111` 是输出初始化，不是无关代码

## 7. `_prefix_sum32_kernel` 详细剖析

代码位置: `114-124`

这是一个非常简单的串行 prefix sum:

- 输入: `counts[32]`
- 输出: `offsets[32]`

含义:

- `offsets[e]` 表示 local expert `e` 的工作队列在 `sorted_token`/`sorted_weight` 里的起始位置

例如:

```text
counts  = [3, 0, 5, 2, ...]
offsets = [0, 3, 3, 8, ...]
```

它只有一个 block，完全不是性能热点。

## 8. `_scatter_local_kernel` 详细剖析

代码位置: `126-169`

### 8.1 它的作用

这个 kernel 做的是“把 `(token, topk_slot)` 扁平化，再按 local expert 分桶”。

它把逻辑上的二元关系:

```text
token t
  -> topk slot k
  -> global expert ge
  -> local expert le = ge - local_expert_offset
```

整理成后续 GEMM 更好处理的一维工作队列。

### 8.2 关键步骤

- `145-148`
  - 把 `[0, T*8)` 的线性 slot 映射回 `token_ids = offs // 8`
  - `k_slot = offs % 8`

- `150-159`
  - 从 `topk_ids/topk_weights` 取出对应 expert 和 weight

- `161-166`
  - 判断是不是 local expert
  - 用 `cursor_ptr` 原子加，得到当前 pair 在本 expert 桶里的相对位置
  - `dest = offsets[local] + pos`

- `168-169`
  - 把 token id 和 weight 写到重排缓冲区

### 8.3 这一步为什么必要

如果不做这一步，后面的 GEMM1/GEMM2 很难按 expert 连续取权重。

这个设计的核心收益是:

- 同一个 local expert 的所有 pair 被放到一段连续区间
- 后面 `pid_e` 可以直接认为自己只处理 expert `e`

### 8.4 代价

这一步使用了原子操作和离散写入。

因此它的特征是:

- 逻辑简单
- 访存不连续
- 更像“调度重排”而不是“数值计算”

## 9. `_gemm1_swiglu_kernel` 详细剖析

代码位置: `172-283`

这是第一个真正的重计算 kernel。

### 9.1 它计算的数学形式

对每个 local expert、每个被该 expert 选中的 token:

```text
A = dequant(hidden_states)
W13 = dequant(gemm1_weights)
G1 = A @ W13^T
out = gate * silu(up)
```

其中:

- 前 2048 维是 `gate`
- 后 2048 维是 `up`
- 代码中实际公式是 `acc_gate * (acc_up / (1 + exp(-acc_up)))`

也就是:

```text
gate * silu(up)
```

### 9.2 grid 语义

`202-204`:

- `pid_n`: 输出列块
- `pid_m`: token-pair 行块
- `pid_e`: local expert id

所以它是一个 3D grid:

```text
(N tiles, M tiles, 32 experts)
```

这意味着:

- 每个 expert 都会被 launch
- 即使某个 expert 一个 token 都没有，也只是靠 `206-208` 早退

### 9.3 它如何取数据

`210-218`:

- 先通过 `offset_ptr` 找到当前 expert 对应的数据段起点
- 再通过 `token_ptr` 找到这段里每行对应的原始 token id

这里非常关键:

- `slot_ids` 是“重排后”的行号
- `token_ids` 才是“原始 hidden_states 的 token 行号”

### 9.4 hidden_states 的 block-scale 反量化

`227-240`:

- `offs_k` 是当前 K tile
- `scale_k = (kb * BLOCK_K) // 128`
- 从 `hidden_scale_ptr[scale_k, token_id]` 取当前 token 在这个 128-wide block 上的 scale
- `hidden = hidden * hidden_scale`

所以 hidden 的 scale layout 是:

```text
hidden_states_scale[hidden_block, token]
```

不是 `[token, hidden_block]`。

### 9.5 `gemm1_weights_scale` 的索引方式

`221-223` 和 `259-270`:

- `gate_scale_n = (pid_n * BLOCK_N) // 128`
- `up_scale_n = gate_scale_n + 16`

因为:

- `gemm1` 输出总长是 4096
- 前 2048 是 gate，对应 16 个 128-block
- 后 2048 是 up，再对应 16 个 128-block

所以 `+16` 是把索引切换到 up 半区。

这是个非常容易改错的点。

### 9.6 计算精度路径

这一段的精度路径是:

1. 从 fp8 tensor 读出来后转成 `fp32`
2. 乘 block scale
3. 再转成 `bf16`
4. `tl.dot` 累加到 `fp32 acc`
5. 最终 `out` 存成 `bf16`

也就是说:

- scale 乘法在 `fp32`
- tensor core 乘加更接近 `bf16 x bf16 -> fp32 accumulate`
- 中间输出 `inter` 最后落地成 `bf16`

### 9.7 手改时最容易踩坑的地方

- `gate` 和 `up` 的半区不要对调
- `hidden_states_scale` 的索引是 `[block, token]`
- `gemm1_weights_scale` 的索引是 `[expert, out_block, k_block]`
- `pid_e` 是 local expert，不是 global expert
- `out` 的每一行对应的是“一个 token-expert pair”，不是“一个 token”

## 10. `_gemm2_scatter_kernel` 详细剖析

代码位置: `286-369`

这是第二个重计算 kernel，也是最终把局部 expert 输出加回 token 维度的地方。

### 10.1 它计算的数学形式

对每个 local expert 的每个 token-pair:

```text
O = inter @ W2^T
output[token_id] += O * routing_weight
```

### 10.2 输入是什么

- `act_ptr`: `GEMM1 + SwiGLU` 的结果
- `token_ptr`: 重排后的 token id
- `sorted_weight_ptr`: 重排后的 routing weight
- `offset_ptr/count_ptr`: 每个 expert 的数据段信息
- `weight_ptr/weight_scale_ptr`: `gemm2` 的 FP8 权重和 scale

### 10.3 grid 语义

`315-317`:

- `pid_n`: 输出 hidden 维列块
- `pid_m`: 当前 expert 下的 token-pair 行块
- `pid_e`: local expert

grid 也是 3D:

```text
(HIDDEN_SIZE tiles, max_count tiles, 32 experts)
```

### 10.4 为什么这里必须 `atomic_add`

`364-368`:

多个 experts 可能同时给同一个 token 输出贡献值。

例如一个 token 选了 8 个 experts，那么这 8 个 experts 最终都会把各自的 `O * w` 加到同一行 `output[token]`。

因此不能简单 `store`，必须 `atomic_add`。

### 10.5 当前版本的精度路径

这里的路径是:

1. `act` 从 `bf16 inter` 读出
2. `gemm2_weights` 读成 `fp32`
3. 乘 scale 后转 `bf16`
4. `tl.dot` 累加到 `fp32 acc`
5. 乘 `row_weights`
6. 再转 `bf16`
7. `atomic_add` 到 `output_buffer`

这意味着当前版本的误差来源里，这里占比通常很高，因为:

- `inter` 已经是 `bf16`
- `routing weight` 最终加回输出前会再经历一次 `bf16`
- `atomic_add` 的目标缓冲区也是 `bf16`

如果后续你想专门压误差，这一段和输出 dtype 是首选观察点。

### 10.6 为什么 `row_weights` 在这里乘

而不是在 `GEMM1` 之后先乘掉，是因为:

- 权重属于 expert 对 token 的最终贡献系数
- 放在 `GEMM2` 之后乘，逻辑更直接
- 也减少了中间激活被权重缩放后的数值损失传播

## 11. Python 包装层如何把这些 kernel 串起来

### 11.1 `_as_cuda_contiguous`

代码位置: `372-379`

作用很简单:

- 如果 tensor 不在 CUDA，搬到 CUDA
- 如果不连续，转连续

这个函数不参与算法，只负责输入预处理。

### 11.2 `_routing_and_scatter`

代码位置: `382-457`

这是 routing 阶段的 Python 调度器。

它完成:

1. 分配 `topk_ids/topk_weights/counts/offsets/sorted_token/sorted_weight`
2. 准备好 `bias`
3. 调 `_routing_topk_kernel`
4. 调 `_prefix_sum32_kernel`
5. 调 `_scatter_local_kernel`
6. 返回重排结果

这个函数是“逻辑阶段切分”的关键点。

如果你之后想把 route plan 单独拿出来调试，这里是最方便插桩的位置。

### 11.3 `_gemm1_swiglu_triton`

代码位置: `460-515`

它主要做两件事:

1. 分配中间缓冲 `out`
2. 构造 grid 并调 `_gemm1_swiglu_kernel`

注意:

- `grid.z = NUM_LOCAL_EXPERTS`
- `grid.y = ceil(max_count / BLOCK_M)`
- 即使只有少数 experts 有数据，也还是对 32 个 experts 全 launch

### 11.4 `_gemm2_scatter_triton`

代码位置: `518-566`

与上面类似:

1. 构造 `gemm2` 的 3D grid
2. 调 `_gemm2_scatter_kernel`

同样地:

- 它按 `NUM_LOCAL_EXPERTS` 全 launch
- 空 expert 依赖 kernel 内部早退

## 12. `run` 入口函数详细剖析

代码位置: `569-699`

### 12.1 这是评测真正调用的入口

它必须:

- 兼容 CPU tensor 输入
- 自动搬运到 CUDA
- 支持 destination passing style
- 最终返回和输入设备一致的结果

### 12.2 输入设备与连续性处理

`586-621`:

- 如果任一关键输入不在 CUDA，就逐个搬到 CUDA
- 否则只做 `.contiguous()`

这是为了保证 Triton kernel 能直接用 stride 做索引，不碰奇怪 layout。

### 12.3 标量规范化

`623-632`:

- `local_expert_offset` 转 Python `int`
- `routed_scaling_factor` 转 Python `float`

这样 Triton launch 时不会因为 0-D tensor 标量带来额外问题。

### 12.4 输出缓冲区策略

`636-661`:

- 如果用户传进来的 `output` 已经是 CUDA 上连续的 `bf16` tensor，就直接复用
- 否则内部新建一个 `output_buffer`

然后 `_finalize_output` 负责:

- 没有显式输出 tensor 时，按原设备返回
- 有显式输出 tensor 时，把结果 copy 回去

### 12.5 真正的流水线

`666-697`:

1. 调 `_routing_and_scatter`
2. 取 `max_count`
3. 如果没有任何 local expert 命中，直接返回
4. 调 `_gemm1_swiglu_triton`
5. 调 `_gemm2_scatter_triton`
6. 返回结果

这一段就是整个文件的“主控流”。

## 13. 这份实现的整体架构特点

### 13.1 它是“分阶段流水线”，不是“单核极限融合”

从工程结构看，这份实现最鲜明的特点是:

- 路由单独做
- 分桶单独做
- GEMM1 单独做
- GEMM2 + scatter 单独做

优点:

- 结构清晰
- 每一段更容易定位问题
- 更容易手动修改某一段

缺点:

- 中间张量较多
- launch 次数多
- 全局内存往返更多
- 对短序列 workload 不够友好

### 13.2 它的核心设计不是“把所有专家一起 GEMM”

它选择的是:

- 先按 expert 把 pair 重排
- 再对每个 local expert 按桶处理

这比“把 top-k 展平成一个大稀疏矩阵再做统一 GEMM”更直接，但也带来:

- atomic
- 分桶
- 空 expert launch

### 13.3 输出初始化被塞进 routing 阶段

这是一种小融合。

好处:

- 省一次单独 kernel

坏处:

- 读代码时不直观
- 以后换 routing 流程容易漏掉输出清零

## 14. 如果你准备手动改，这些位置最值得优先盯

### 14.1 想改路由算法

看:

- `58-67`
- `69-75`
- `77-102`

尤其注意:

- 选 expert 的分数和归一化权重不是同一个量
- tie-break 不建议删

### 14.2 想改分桶/调度方式

看:

- `90-93`
- `126-169`
- `392-457`

这是“token 如何被整理成 expert 连续工作流”的核心。

### 14.3 想改 GEMM1 性能

看:

- 参数: `20-24`
- kernel: `202-283`
- wrapper grid: `479-514`

重点关注:

- `BLOCK_M/N/K`
- `num_warps`
- `num_stages`
- `gate/up` 的权重布局与 scale 索引

### 14.4 想改 GEMM2 性能或精度

看:

- 参数: `26-30`
- kernel: `315-369`
- wrapper grid: `532-565`

重点关注:

- `atomic_add`
- `output_buffer` dtype
- `row_weights` 的乘法位置

### 14.5 想压误差

先看:

- `278-282`
- `346`
- `361-368`
- `642-660`

因为当前数值路径里，最明显的低精度点就是:

1. `GEMM1` 输出落 `bf16`
2. `GEMM2` 输入是 `bf16`
3. 最终按权重后再转 `bf16`
4. 输出缓冲区本身也是 `bf16`

### 14.6 想减少无效 launch

先看:

- `206-208`
- `319-321`
- `479-483`
- `532-536`

当前做法是:

- 对 32 个 local experts 全 launch
- 空 expert 依赖 early return

这在工程上简单，但对 occupancy 和调度都不算友好。

## 15. 这份代码的几个“隐含约定”

### 15.1 `token_sorted` 里存的是原始 token id

不是重排后的连续编号。

### 15.2 `counts/offsets` 描述的是 local expert 桶

不是 global expert。

### 15.3 `inter` 的行数是 `T * TOP_K`

但有效区间由 `counts/offsets` 决定。

### 15.4 `run_local.py` 评测时真正打包的是 `kernel.py`

不是你手动改出来的 `solution.json` 快照。

也就是说:

- 改 `solution/triton/kernel.py` 才是根本
- `solution.json` 只是 `pack_solution()` 之后的产物

## 16. 建议的阅读顺序

如果你要手动改算子，建议按这个顺序读:

1. 先看 `run`，理解整体控制流
2. 再看 `_routing_and_scatter`，理解工作队列怎么形成
3. 再看 `_gemm1_swiglu_kernel`，理解第一段 GEMM 的数据布局
4. 再看 `_gemm2_scatter_kernel`，理解为什么要 atomic 加回输出
5. 最后回头看常量与 tile 参数，开始做针对性修改

## 17. 最后给你的修改策略建议

如果你的目标是“先能稳妥地手改”，推荐顺序不是一下子大改，而是:

1. 只改 tile 参数，先看性能变化
2. 再改单个阶段的 dtype 路径，先看误差变化
3. 再改 launch/grid 策略，减少空 expert
4. 最后再尝试更激进的融合或重排方案

最不建议的做法是:

- 同时改 routing、分桶、GEMM tile、dtype、输出累加方式

因为那样一旦错了，很难判断错在语义还是错在数值路径。

## 18. 速查表

最常改的区域:

- 路由逻辑: `33-112`
- 分桶逻辑: `126-169`
- GEMM1 计算: `172-283`
- GEMM2 计算与回写: `286-369`
- Triton tile 参数: `20-30`
- 顶层执行流: `666-697`

最容易改错的区域:

- routing 归一化权重来源: `97-102`
- `gate/up` 半区索引: `221-223`
- `gemm1_weights_scale` 索引: `259-270`
- `atomic_add` 回写逻辑: `364-368`
- 输出初始化隐藏在 routing kernel: `104-111`
