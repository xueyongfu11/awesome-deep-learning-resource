# 从混合 Batch 到 GPU Kernel：读懂 vLLM 与 SGLang 的 Attention 执行

一个请求正在逐 token 生成，另一个请求还在处理长 prompt。vLLM 把它们放进同一轮计算后，QKV 投影能合起来做，Attention 又怎样区分请求？只有一个 query 的 decode，会不会和几百个 query 的 prefill 使用同样的 tile？FlashAttention、FlashInfer 和 CUDA Graph 分别参与了哪一层决策？

本文沿着一轮推理的执行路径回答这些问题：先看调度与输入形状，再看请求如何映射为 GPU 任务，最后讨论 CUDA Graph、后端选择和性能证据。贯穿全文的是同一个包含两个 decode 请求和两个 prefill 请求的例子。

默认场景是普通自回归 Transformer 的稠密因果注意力，不开启投机解码；MoE、MLA、多模态和投机解码在对应章节单独说明。文中的形状与伪代码用于解释执行机制，不是可直接运行的实现。

> 版本说明：资料核对日期为 2026-11-07。关键源码链接固定到提交快照，见文末；这些快照用于定位实现，并不代表本文在同一套安装环境中运行了全部实验。官方 `stable`/`latest` 文档仍会变化，部署行为应以所用版本、配置和启动日志为准。本文引用的性能数据均来自公开资料，没有新增本地 benchmark。

阅读路线：第 1～3 节解释调度、Batch 和张量形状；第 4～7 节解释后端调用、任务划分及 FA3 源码案例；第 8 节解释 CUDA Graph；第 9～10 节比较后端选择规则；第 11～12 节讨论性能与调优。

先记住三个层次：**调度器决定这轮算哪些 token，Attention 后端决定怎样组织注意力计算，kernel 决定 GPU 怎样执行这些任务。** 一次合批不会把这三个层次合并成一个决策。

## 1. 调度：Prefill chunk 如何与 decode 合批

核心是：vLLM 把不同请求本轮需要计算的 token 拼成一个 batch。有些请求贡献几十、几百个 prefill token，有些请求只贡献一个 decode token。

### 1.1 哪些 chunk 可以一起执行

- 不同请求的 chunk：可以放进同一轮 batch，彼此独立。
- 同一个请求的连续 chunk：通常分不同轮执行，后一个 chunk 读取前面 chunk 已生成的 KV cache。

Prefill 处理输入 prompt；decode 则在已有 KV cache 的基础上，反复输入刚生成的 token，推进后续生成。Chunked prefill 将长 prompt 分成多轮处理，为其他请求提供穿插执行的机会。

例如请求 A 的 prompt 有 3,000 个 token：

```text
A1 = token    0～999
A2 = token 1000～1999
A3 = token 2000～2999
```

执行 A2 时：

```text
Q：A2 的 token
K/V：缓存中的 A1 + 本轮计算的 A2
```

A2 内部仍使用 causal mask：位置 1,200 只能看位置 0～1,200。A1 的 KV 直接复用，不必重新做其 K/V 投影。

Chunking 切的是每轮推进多少 token，没有切断上下文。

### 1.2 混合 batch 的调度

例如一轮包含：

| 请求 | 阶段          | 本轮处理 token 数 |
| ---- | ------------- | ----------------: |
| D1   | decode        |                 1 |
| D2   | decode        |                 1 |
| A    | prefill chunk |               600 |
| B    | prefill chunk |               400 |

逻辑输入为：

```text
[D1 的 1 token | D2 的 1 token | A 的 600 tokens | B 的 400 tokens]
共 1002 tokens
```

V1 调度器通过 `num_scheduled_tokens` 等信息描述各请求本轮推进多少 token，同时受 token budget、请求数量、KV cache 空间等约束。

这里 decode 的输入 token，是上一步采样得到、这一步送入模型的 token；本轮结束后再采样下一个。

多轮推进可以示意为：

```text
第 1 轮：[D decode | A chunk1 | B chunk1]
第 2 轮：[D decode | A chunk2 | B chunk2]
第 3 轮：[D decode | A decode | B decode]
第 4 轮：[D decode | A decode | B decode]
```

实际能否同时推进 A、B、每个 chunk 多大，取决于预算和调度配置。

中间 prompt chunk 主要推进计算、保存 KV。最后一个 prompt chunk 完成时，可以利用 prompt 最后位置的输出采样第一个生成 token；下一轮把该 token 送回模型，继续 decode。

参考：[V1 调度器](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/core/sched/scheduler.py)、[Chunked prefill 文档](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)。

## 2. 算子：线性层合批，Attention 保留请求边界

QKV 投影、普通 dense FFN 都是逐 token 使用相同权重的运算，可以沿 token 维合批：

```text
X: [本轮总 token 数, hidden_size]

QKV = X @ W_qkv
FFN(X)
```

Attention 则要沿序列聚合信息，必须知道每个 query 属于哪个请求、能读取哪些 KV：

```text
Q_D1 → KV_D1
Q_D2 → KV_D2
Q_P1 → KV_P1，遵守 causal mask
Q_P2 → KV_P2，遵守 causal mask
```

一层的逻辑流程是：

```text
所有 token 合起来做 QKV 投影
              ↓
位置编码、KV cache 更新等
              ↓
Attention 按请求访问各自的 KV，批量执行
              ↓
所有 token 合起来做输出投影与普通 FFN
```

按请求隔离，不等于 Python 遍历每个请求并单独调用 Attention。一个 kernel 可以执行来自多个请求、多个 head、多个 tile 的任务。

若 FFN 是 MoE，还会根据 token 选择的 expert 重新分组。

## 3. 张量：从请求数量到变长 Attention 输入

### 3.1 Batch 到底指什么：请求数、输入 token 数与上下文长度

Batch 是这次一起送进模型计算的一组输入。“一轮 forward”就是调用一次模型，让这些输入经过所有 Transformer 层。它不是单独一层的调用，也不是一个请求从收到到生成结束的全过程。

描述 batch 时，需要区分三个数量：

| 数量                    | 含义                                                |
| ----------------------- | --------------------------------------------------- |
| 本轮请求数 `B`        | 这次实际参与计算的请求数量                          |
| 本轮输入 token 数 `T` | 各请求这轮新输入的 query token 总数，`T = Σ qᵢ` |
| 各请求 KV 长度 `Lᵢ`  | Attention 可读取的上下文长度，包含历史缓存          |

服务器的并发请求中可能有请求仍在排队，因此并发请求数不一定等于本轮请求数 B。

例如服务器当前有三个请求：A、B 正在生成回答，C 刚收到一个 100-token 的问题。调度器让它们一起执行：

```text
[A 上一步生成的 1 token | B 上一步生成的 1 token | C 的 100 tokens]
                              ↓
                         调用一次模型
                              ↓
                   依次经过所有 Transformer 层

B = 3，T = 102
```

C 的 prompt 完成后，下一轮可能变成每个请求各输入一个新生成的 token，此时仍是 `B=3`，但 `T=3`。A、B 的历史上下文通过 KV cache 读取，不必将历史 token 全部重新输入模型。

普通单 token decode 中，`B=T`，例如 32 个请求各输入一个 token，对应 `B=32、T=32`。混合 prefill/decode 时则通常不同，下面的例子是 `B=4、T=386`。

这里的 T 是展平输入的长度，不是一条合并后的真实序列。普通 dense QKV 投影和 FFN 可以处理同一个 `[T, hidden_size]` 张量；Attention 仍通过元数据隔离请求，各自读取上下文。

### 3.2 一个贯穿后文的例子

| 请求 | 阶段            | 已缓存 token 数 | 本轮输入 token 数 | 本轮后的 KV 长度 |
| ---- | --------------- | --------------: | ----------------: | ---------------: |
| D1   | decode          |             100 |                 1 |              101 |
| D2   | decode          |             200 |                 1 |              201 |
| P1   | chunked prefill |             512 |               128 |              640 |
| P2   | 首次 prefill    |               0 |               256 |              256 |

设：

```text
Hq  = 32      # query heads
Hkv = 8       # KV heads，GQA
d   = 128     # head dimension
T   = 386     # 1 + 1 + 128 + 256
```

本轮的逻辑张量为：

```text
Q:     [386, 32, 128]
K_new: [386,  8, 128]
V_new: [386,  8, 128]

索引       0      1       2 … 129       130 … 385
请求      D1     D2      P1 的 chunk      P2 的 prompt
```

通常不需要组织成 `[batch, padded_seq_len, heads, dim]`。本轮 token 展平成一个维度，变长信息通过元数据描述。

`K_new/V_new` 只含新增 token。KV cache 写入可以是独立步骤；Attention 计算读取 Q 和已经更新的 cache，具体函数签名取决于后端。

### 3.3 边界和分页元数据

```text
query_start_loc / cu_seqlens_q:
[0, 1, 2, 130, 386]

相邻边界之差，即各请求本轮 Q 长度：
[1, 1, 128, 256]

seq_lens:
[101, 201, 640, 256]

block_table:
[4, 每个请求最多需要的 KV block 数]

slot_mapping:
[386]
```

- `query_start_loc`：定位每个请求在 Q 张量中的 span。
- `seq_lens`：各请求本轮的 KV 总长度；单个 query 实际可见的位置还受 causal mask 等规则限制。
- `block_table`：请求的逻辑 KV block 到物理 cache block 的映射。
- `slot_mapping`：本轮新增 token 的 KV 写入位置。

KV cache 的逻辑布局可以理解为：

```text
K_cache: [总物理 block 数, block_size, Hkv, d]
V_cache: [总物理 block 数, block_size, Hkv, d]
```

实际维度顺序、K/V 是否合并存储，由后端决定。

### 3.4 数学上仍是独立的 Attention

对每个 query head：

```text
D1: Q [  1, 128] × Kᵀ [128, 101] → scores [  1, 101]
D2: Q [  1, 128] × Kᵀ [128, 201] → scores [  1, 201]
P1: Q [128, 128] × Kᵀ [128, 640] → scores [128, 640]
P2: Q [256, 128] × Kᵀ [128, 256] → scores [256, 256]
```

scores 是数学上的形状，高效实现通常不把完整分数矩阵写入显存。

P1 有 512 个历史 token，其内部第 `i` 个 query 可见的 KV 位置满足：

```text
j <= 512 + i
```

所以它需要带历史前缀的 causal mask，不能直接使用左上角对齐的 `128×640` 下三角 mask。

## 4. 实现：FlashAttention 与 FlashInfer 各自负责什么

FlashAttention 是高效计算精确 Attention 的算法及实现；FlashInfer 是面向 LLM 推理的 GPU 算子库，也包含高效 Attention，以及采样、量化等其他算子。

| 方面       | FlashAttention                     | FlashInfer                                   |
| ---------- | ---------------------------------- | -------------------------------------------- |
| 定位       | 高效 Attention 算法和实现          | LLM 推理算子库                               |
| 关注点     | 分块、online softmax、减少显存读写 | prefill、decode、分页 KV、变长请求等推理需求 |
| 在 vLLM 中 | 一种 Attention 后端                | 另一种 Attention 后端                        |

两者的 Attention 实现共享分块、减少 HBM 读写等思想。FlashInfer 不是 FlashAttention 的 vLLM 封装，也不能简单视为其升级版。

参考：[FlashAttention 论文](https://arxiv.org/abs/2205.14135)、[FlashInfer 论文](https://arxiv.org/abs/2501.01005)。

### 4.1 变长能力不是 FlashAttention 独有

两者都支持：

- 同一 batch 中各请求的 KV 长度不同。
- 同一 prefill batch 中各请求的 query 长度不同。

例如 Q 长度 `[128,256,64]`，历史 KV 长度 `[1000,2000,500]`，两者都可以批量处理，不必把请求统一 padding 到最长长度。

需要区分“库支持变长”与“vLLM 当前后端如何组织混合 batch”。

参考：[FlashInfer Attention API](https://docs.flashinfer.ai/api/attention.html)。

## 5. 调用：统一变长接口与分阶段接口

### 5.1 FlashAttention 的统一变长路径

在本文引用的 vLLM V1 普通 FlashAttention 路径中，可以将整个 batch 交给变长接口：

```text
Q:           [386, 32, 128]
Q 累计边界:   [0, 1, 2, 130, 386]
KV 长度:      [101, 201, 640, 256]
block_table: 4 个请求
                    ↓
output:      [386, 32, 128]
```

参考：[vLLM FlashAttention 后端](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/attention/backends/flash_attn.py)。

### 5.2 FlashInfer 的分类调用路径

本文引用的 vLLM 普通 FlashInfer wrapper 路径把 decode 放在前面、prefill 放在后面，分别调用。FlashInfer 后端还有其他分支，下面聚焦这一种组织方式：

```text
Decode:
Q:           [2, 32, 128]
KV 长度:      [101, 201]
结果写入:     output[0:2]

Prefill:
Q:           [384, 32, 128]
Q 累计边界:   [0, 128, 384]
KV 长度:      [640, 256]
结果写入:     output[2:386]
```

简化代码：

```python
# 示意：省略量化、并行及其他分支。
n = num_decode_tokens

if num_prefill_tokens > 0:
    prefill_wrapper.run(query[n:], kv_cache, out=output[n:])

if num_decode_tokens > 0:
    decode_wrapper.run(query[:n], kv_cache, out=output[:n])
```

切片通常是 view，结果写入同一个输出张量，不必再复制拼接。这里每次调用批量处理一组请求，不是逐请求调用。

这一组织方式不能用来断言 FlashInfer 库本身不具备统一处理能力。

参考：[vLLM FlashInfer 后端](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/v1/attention/backends/flashinfer.py)。

### 5.3 一次调用不等于一个 CUDA kernel，也不保证并行重叠

需要区分：

```text
调度 batch → 模型算子 → 后端接口调用 → CUDA kernel
```

一次 Attention 调用可能包含预处理、主计算、split-KV 归并等多个 kernel。

两条 Attention 路径若提交到同一 CUDA stream，通常按提交顺序执行。上述 FlashInfer 普通代码先调用 prefill，再调用 decode。

“调度时优先给 decode 分配预算”不意味着“decode kernel 必须先执行”。混合 batch 也不自动保证 prefill 的计算与 decode 的访存互相掩盖耗时。

## 6. 任务：span、分页 KV 与 split-KV

### 6.1 span 定位 Q，block table 定位 KV

任务可以逻辑上表示为：

```text
(request_id, head/group_id, query_tile_id)
```

这是一种解释方式，不要求实现真的存储三元组。实际可以使用多维 grid、任务表、persistent scheduler 等。

对于一个请求的 query tile，其 Q 地址由请求起点和 tile 内位置确定。不会把所有请求拼成一个跨请求的 Attention。

例如 P1 在 Q 中从全局行 2 开始。请求内部 token 64～127 对应全局 token 行 66～129。这里描述的是 token 索引；启用 PackGQA 后，计算 tile 的 M 行还包含 head 维，详见下一节。

假设 KV block size 为 16，P1 的 block table 是：

```text
逻辑 block:   0   1   2   3 ...
物理 block:  37   8  91  12 ...
```

读取 P1 中索引为 35 的 KV token（索引从 0 开始）：

```text
逻辑 block = 35 // 16 = 2
块内偏移   = 35 % 16  = 3
物理 block = block_table[P1, 2] = 91

读取 K_cache[91, 3, kv_head, :]
读取 V_cache[91, 3, kv_head, :]
```

各请求使用自己的 block table，逻辑上不会读取无关请求的上下文。前缀缓存可以让兼容请求共享相同的物理 KV 页，但不会改变请求各自的可见范围。不同请求的任务可以同时在 GPU 上执行。

### 6.2 分块计算与 split-KV

一个 query tile 可以沿 KV 方向逐块加载，通过 online softmax 更新输出累积值，不保存完整 scores。

为了增加长 KV、少 query 时的并行度，也可以沿 KV 维分段：

```text
(request_id, head/group_id, query_tile_id, kv_split_id)
```

多个任务分别产生局部输出和 softmax 统计量，随后归并。局部结果不能直接相加，必须考虑各段的归一化关系。

## 7. 源码案例：单 query decode 为什么有四个有效计算行

“Decode 只有一个 query，所以一个 tile 只有一行有效”听起来直观，却忽略了 GQA 的 head 打包。下面沿 vLLM 维护的 FlashAttention 仓库中的 FA3 Hopper 路径，计算本文混合 batch 的实际 tile 组织。

**适用范围：Hopper、FA3、BF16、普通 causal attention，Q/K/V head dimension 均为 128，Hq=32、Hkv=8，query 长度为 `[1,1,128,256]`，正常启用 PackGQA 的路径。不是对 FA4、FlashInfer 或所有版本的统一结论。**

### 7.1 tile 配置按整次调用选择

`use_one_mma_wg()` 中的判断包括：

```cpp
params.arch >= 90
&& (params.d == 128 || params.d == 64)
&& !params.is_local
&& params.seqlen_q * (params.h / params.h_k) <= 64
```

变长输入中的 `params.seqlen_q` 是调用的最大 Q 长度。此例：

```text
max_query_len = 256
Hq / Hkv = 4
256 × 4 = 1024 > 64
```

因此不启用 one-MMA-warpgroup 分支。判断发生在整次调用的 dispatch 阶段，不是每个请求单独 dispatch。

参考：[heuristics.h](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/heuristics.h)、[flash_api.cpp](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/flash_api.cpp)、[flash_fwd_launch_template.h](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/flash_fwd_launch_template.h)。

### 7.2 本例的 kBlockM 为 128

`tile_size_fwd_sm90()` 对两字节类型、head dimension 128 的对应分支：

```cpp
if (use_one_mma_wg) {
    return {64, ..., true, true};
} else {
    return {128, ..., true, true};
}
```

第一项是 `kBlockM`。上述混合 batch 选择 M=128。

若整批都是单 token decode，最大 Q 长度变成 1，`1×4<=64`，在同样限定条件下会选 one-MMA-warpgroup 分支，M=64。

参考：[tile_size.h](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/tile_size.h)。

### 7.3 PackGQA：计算行不是纯 token 维

32 个 Q heads、8 个 KV heads，表示每 4 个 Q heads 共享一个 KV head。

PackGQA 把同一 KV-head 组内的 `(token, Q head)` 组织到计算矩阵的行维。变长 GQA 的默认判断会启用打包，但构建选项、显式覆盖和其他路径仍须看具体版本。

参考：[打包选择逻辑](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/heuristics.h)、[PackGQA 实现](https://github.com/vllm-project/flash-attention/blob/af5422c5c24e2f464d394295c5b2f659f9332217/hopper/pack_gqa.h)。

### 7.4 D1：一个 token，每个 KV-head 组有四个有效计算行

只看 KV head 0 对应的 Q heads 0～3：

```text
D1 token 0, Q head 0 ─┐
D1 token 0, Q head 1 ─┼─ 共同读取 D1 的 KV head 0
D1 token 0, Q head 2 ─┤
D1 token 0, Q head 3 ─┘
```

它们是不同 Q 向量：

```text
Q_group: [4, 128]

Q_group [4,128] × K_head0ᵀ [128,101] → scores [4,101]
```

因此混合 batch 中，D1 每个 KV-head 组占一个 M=128 的 tile，但只有 4 行对应有效数据：

```text
行 0：token 0，Q head 0   有效
行 1：token 0，Q head 1   有效
行 2：token 0，Q head 2   有效
行 3：token 0，Q head 3   有效
行 4～127：              无有效数据
```

这是计算 tile 的容量与有效数据的区别，不表示输入显存中为 D1 分配了 128 个 token，也不表示 kernel 中每条指令的成本都能简单按有效行比例折算。

三个数字要分清：

```text
256：P2 的 query token 数，影响整次调用的 tile 选择
128：选择出的计算 tile 的 M 维大小
  4：D1 每个 KV-head 组的有效计算行数 = 1 token × 4 heads
```

### 7.5 P1：128 个 token，每组有 512 行，分成四个 tile

```text
P1 的 Q： [128 tokens, 32 heads, 128 head_dim]

只看一个 KV-head 组：
打包前：  [128 tokens, 4 heads, 128 head_dim]
打包后：  [512 行, 128 head_dim]
```

有效行可按 token/head 理解为：

```text
行 0～3：    token 0 的 Q heads 0～3
行 4～7：    token 1 的 Q heads 0～3
……
行 508～511：token 127 的 Q heads 0～3
```

每个 M=128 tile 容纳 32 个 token × 4 个 heads：

| Tile | 打包后的行 | 对应 P1 token          |
| ---- | ---------- | ---------------------- |
| 0    | 0～127     | 0～31，各 4 个 heads   |
| 1    | 128～255   | 32～63，各 4 个 heads  |
| 2    | 256～383   | 64～95，各 4 个 heads  |
| 3    | 384～511   | 96～127，各 4 个 heads |

P1 有 8 个 KV-head 组，共有 `8×4=32` 个有效 query tiles，未计入 KV split 等进一步任务拆分。

| 请求 | 每个 KV-head 组有效行数 | 每组 query tile 数 |
| ---- | ----------------------: | ------------------ |
| D1   |              `1×4=4` | 1，未填满          |
| P1   |          `128×4=512` | 4，填满            |

每行的 causal mask 仍按其 token 位置确定。同一 token 的四个 heads 具有相同的可见 KV 范围。

**这条 FA3 路径中，整次调用选定 kernel 配置，D1 不会仅因自身 Q 长度为 1，就自动另派一个专用 decode kernel。各请求任务数量和 KV 分段可以不同，但不能由此断言它们使用不同的 query tile 配置。**

## 8. 动态 Batch 与 CUDA Graph

理解实际输入形状后，还需要区分调度量与 CUDA Graph 的捕获尺寸。图复用属于执行层优化，不能只根据请求数或 token 数判断。

### 8.1 CUDA Graph 的 1、2、4、8……128、512 是什么

这些数字通常指预先捕获的执行尺寸档位，传统配置名为 `cudagraph_capture_sizes`。vLLM 为若干固定尺寸捕获 GPU 操作及依赖，推理时选择适用的图重放，减少重复提交 kernel 的开销。

传统配置中的尺寸主要按本轮输入 token 总数理解。纯 decode 时，每个请求输入一个 token，因此尺寸为 128 的图可以对应 128 个请求；它不是每个请求的上下文长度。

实际输入不必刚好等于已捕获尺寸，可以补齐到适用档位。例如，假设有 `[1,2,4,8,16,…]` 这些图，且对应模式和元数据允许：

```text
实际 T = 3 → 补齐到 4 个输入位置 → 重放尺寸为 4 的图
实际 T = 5 → 补齐到 8 个输入位置 → 重放尺寸为 8 的图
```

结果中只使用真实请求所需的部分。不必等到凑够 4 个或 8 个真实请求才执行，也不是把每个请求的历史上下文补齐到这个长度。

必须区分实际调度量和图的执行尺寸：

```text
实际调度输入 token 数：3
图的补齐后执行尺寸：  4
```

核对时的 stable 配置文档给出的默认候选模式包括：`1、2、4`，随后是小于 256 的 8 的倍数，以及从 256 起的 16 的倍数，截断到配置上限；具体列表与版本、平台及配置有关，并不是只按 2 的幂捕获。

小批次的 GPU 计算时间短，CPU/kernel 提交开销占比容易较高，因此 CUDA Graph 往往更有价值。为小尺寸保留较细档位，还能减少 padding 浪费；捕获更多尺寸则会增加启动时间及图相关内存开销。

参考：[捕获尺寸配置](https://docs.vllm.ai/en/stable/configuration/engine_args/#--cudagraph-capture-sizes)、[最大捕获尺寸](https://docs.vllm.ai/en/stable/configuration/engine_args/#--max-cudagraph-capture-size)。

### 8.2 相同 token 数不保证复用同一张完整 CUDA Graph

```text
批次 A：32 个请求 × 1 token   → B=32，T=32，普通 decode
批次 B： 1 个请求 × 32 tokens → B=1， T=32，prefill
```

两者线性层的展平输入形状相同，但 Attention 的请求边界、访问范围和任务组织不同。是否能使用同一张完整图，还取决于捕获模式、后端及元数据约束。

vLLM CUDA Graph 设计文档中的 `BatchDescriptor` 包含 `num_tokens`、`num_reqs`、`uniform` 等信息；token 数可以是 padding 后的尺寸。纯 decode、投机验证、混合 batch 可能采用不同的 full/piecewise 图或 eager 路径。不能仅凭 T 相同判断图一定兼容，也不能认为每种 KV 长度都必须重新捕获一张图。

新版 Model Runner V2 进一步分别提供 prefill 的 token 数档位和 decode 的请求数档位配置。因此查看具体参数时，应确认版本和单位，不能把所有名为 capture size 的参数一律解释成 token 数。

看到“batch size = 32”时，首先确认它指的是 32 个请求、32 个实际输入 token，还是补齐后为 32 的执行档位。

参考：[CUDA Graph BatchDescriptor](https://docs.vllm.ai/en/stable/design/cuda_graphs/#batchdescriptor)、[新版捕获配置](https://docs.vllm.ai/en/latest/configuration/engine_args/#--cudagraph-capture-sizes)。

### 8.3 图可以复用，输入数据仍然每轮更新

CUDA Graph 重放复用的是已捕获的 GPU 操作及依赖。对通常的 PyTorch 图重放，需要保持适用的执行形状及所引用缓冲区的地址稳定；token ID、position 等输入值则可以在重放前写入这些缓冲区。

```text
捕获：为某个执行尺寸准备固定缓冲区，记录 GPU 操作
下一轮：把新的输入和适用的元数据写入缓冲区
重放：通过相同地址读取这轮的新数据
```

KV 长度增长、请求对应的物理页变化，可以由支持图重放的 Attention 实现通过元数据处理。前提是这轮数据仍符合已捕获路径的约束；如果需要切换 kernel 路径、工作区或执行模式，框架还需要选择另一张适用的图，或回到 eager 执行。

因此，输入内容动态变化和图可复用并不矛盾。固定缓冲区也不表示把历史 KV 复制到一份新的、定长的上下文中。

参考：[PyTorch CUDA Graphs](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)、[vLLM CUDA Graph 设计](https://docs.vllm.ai/en/stable/design/cuda_graphs/)。

## 9. 选择后端：vLLM 的兼容性与优先级

官方入口：[Attention Backend Feature Support](https://docs.vllm.ai/en/stable/design/attention_backends/)，其中包括功能支持、自动选择和 CUDA 后端优先级。

自动选择的规则：

1. 按预设优先级检查候选后端。
2. 校验 dtype、KV dtype、head size、GPU 架构及功能等条件。
3. 选第一个兼容后端。
4. 全部不兼容则报错。

显式指定后端时，会验证配置，不兼容则报错，而不是任意静默替换。

### 9.1 普通 causal MHA/MQA/GQA

本文源码快照中，普通因果注意力的主要优先级为：

| GPU                   | 从高到低的主要候选                                                        |
| --------------------- | ------------------------------------------------------------------------- |
| Ampere / Ada / Hopper | `FLASH_ATTN` → `FLASHINFER` → `TRITON_ATTN` → `FLEX_ATTENTION` |
| Blackwell SM10.x      | `FLASHINFER` → `FLASH_ATTN` → `TRITON_ATTN` → `FLEX_ATTENTION` |

上述候选之后还有 `TURBOQUANT` 等更低优先级候选。兼容的多模态 prefix 配置可能优先组合后端；SM10.x 非 causal attention 会优先尝试 FlashAttention。

优先级表和功能表由后端注册信息及 `AttentionBackend.validate_configuration()` 的检查逻辑自动生成。功能表主要用于确认模型 dtype、KV dtype、block size、head size、attention sink、非因果注意力、多模态 prefix、DCP 和注意力类型等支持范围，不能当作性能排名。

### 9.2 MLA 与其他硬件

- Hopper MLA 的主要优先方向是 `FLASH_ATTN_MLA`。
- Blackwell SM10.x MLA 的主要优先方向是 `FLASHINFER_MLA`。
- 稀疏 MLA、SM120、ROCm、CPU 等使用不同分支。

具体以对应版本源码为准：[CUDA `_get_backend_priorities()`](https://github.com/vllm-project/vllm/blob/dfebb42d096d5c6a04aa5a1722195e37cb8e0b2d/vllm/platforms/cuda.py)。

### 9.3 手动选择与日志确认

```bash
# 自动选择
vllm serve <model>

# 手动选择
vllm serve <model> --attention-backend FLASH_ATTN
vllm serve <model> --attention-backend FLASHINFER
```

上面是命令示例，`<model>` 需要替换为实际模型。部署时看启动日志的 `Using ... attention backend` 才能确认最终结果。

安装了 FlashInfer，或采样/MoE 使用 FlashInfer 算子，不表示 Attention 后端也选了 FlashInfer。

也可以通过 `--attention-config.backend` 或 `-ac.backend` 设置基础后端，但不要与 `--attention-backend` 同时使用。Python API 支持 `LLM(..., attention_backend="FLASH_ATTN")` 或 `AttentionConfig`。

### 9.4 后端名称下面还有具体实现

`FLASH_ATTN` 不等于固定使用某一代 FlashAttention。文档列出的默认版本方向是：SM100+ 使用 FA4，SM90 使用 FA3，其他适用架构使用 FA2；具体形状和功能仍有例外与回退条件。可以用 `--attention-config.flash_attn_version` 指定版本。

`FLASHINFER` 也包含不同执行路径：Native、SM90 的 XQA decode，以及 SM100 上的 trtllm-gen。性能比较应确认具体版本、执行路径和配置，不能只记一个库名。

MLA 则分别选择 prefill 和 decode 后端，prefill 可用 `-ac.mla_prefill_backend` 覆盖；不能仅根据基础 MLA 后端名推断两阶段的全部实现。

参考：[普通 Attention 后端及实现说明](https://docs.vllm.ai/en/stable/design/attention_backends/#standard-attention-mha-mqa-gqa-backends)、[MLA 后端](https://docs.vllm.ai/en/stable/design/attention_backends/#mla-multi-head-latent-attention-backends)。

### 9.5 为什么这样选：运行规则与排序依据分开理解

```text
硬件平台 + 注意力类型 + 特殊功能需求
                 ↓
          确定候选后端顺序
                 ↓
检查 dtype、head size、KV 格式、block size 等
                 ↓
          采用第一个兼容后端
```

“怎样选”可以直接由上述规则回答；“为什么把某个后端排在前面”则需要对应的工程或实验依据。

| 考量                 | 对选择的影响                                    |
| -------------------- | ----------------------------------------------- |
| 硬件适配             | GPU 的计算、访存和指令能力不同，需要对应 kernel |
| 功能覆盖             | 必须支持模型要求的注意力语义、KV 格式和并行方式 |
| 默认性能与工程成熟度 | 默认策略需要兼顾常见负载表现、稳定性和可用性    |

硬件及功能约束可以从文档验证。默认性能与成熟度是对工程策略的解释；这份 vLLM 文档没有给出每项排序的完整实验依据，不能将优先级直接解释为所有负载下的速度排名。

参考：[vLLM 选择行为](https://docs.vllm.ai/en/stable/design/attention_backends/#backend-selection-behavior)。

### 9.6 具体例子：Blackwell 上 Gemma 4 为什么用组合后端

官方文档明确给出的例子是：在 Blackwell 上运行 Gemma 4，兼容的多模态 prefix 配置优先使用 `TRITON_FLASHINFER`。

```bash
vllm serve google/gemma-4-31B-it
```

兼容配置下无须手动覆盖，自动选择组合后端。原因是不同计算内容需要不同的注意力规则：

| 当前计算内容                  | 所需语义                   | 实现       |
| ----------------------------- | -------------------------- | ---------- |
| 当前 query 需要图像双向注意力 | 按多模态规则允许双向可见   | Triton     |
| 因果文本 prefill              | 按因果规则读取 KV          | FlashInfer |
| 因果文本 decode               | 新 token 读取允许的历史 KV | FlashInfer |

两套实现共享 KV cache，由组合后端路由。判断依据是当前 query 是否需要图像双向注意力；仅仅历史 KV 中含有图像 token，不会触发 Triton。

这说明后端选择可以由当前所需的注意力语义决定。文档能证明上述功能与路由设计，但没有给出该配置下的统一对比数据，不能据此宣称 FlashInfer 比 Triton 快多少，或在 Blackwell 上全面快于 FlashAttention。

Hopper 也有对应的 `TRITON_FLASH_ATTN` 组合路径，受其因果子后端必须解析到 FA4 等兼容条件限制。

参考：[Triton/FlashInfer 组合后端](https://docs.vllm.ai/en/stable/design/attention_backends/#tritonflashinfer-composite)、[Triton/FlashAttention 组合后端](https://docs.vllm.ai/en/stable/design/attention_backends/#tritonflashattention-composite)。

## 10. 选择后端：SGLang 的默认策略与阶段组合

SGLang 官方入口：[Attention Backend Selection Guide](https://github.com/sgl-project/sglang/blob/043cfc4d3a2cc2a931d15862f08c4b60b692c957/docs/docs/advanced_features/attention_backend.mdx#attention-backend-selection-guide-cuda)。

对常见 MHA/GQA 模型，文档概括为：

| 硬件                     | 常见默认方向                                       |
| ------------------------ | -------------------------------------------------- |
| Ampere / Ada，如 A100    | FlashInfer，可用性不足时回退 Triton                |
| Hopper，如 H100/H200     | FA3，需 CUDA 12.3+ 和配置支持                      |
| Blackwell SM100，如 B200 | `trtllm_mha`，受 speculative decoding 等条件限制 |

MLA、混合模型、SM120、非对称 K/V head dimension 等有额外分支。文档概括与当前主分支源码可能不同，不能将所有 Blackwell 型号混为一类。

指南对 MLA 的概括是：Hopper 默认 FA3，要求 CUDA 12.3+；Blackwell 通常默认 FlashInfer，对 DeepSeek V3 专门自动选择 `trtllm_mla`；其他架构默认 Triton。这是文档层面的概括，特殊模型仍需查看对应分支。

### 10.1 设计目标与实际依据

维护者说明：优先性能，其次易用性，同时考虑兼容性；性能变化后默认策略也会调整。

参考：[默认策略讨论 #5064](https://github.com/sgl-project/sglang/issues/5064)。

本文引用的源码快照中，Hopper MHA 分支旁保留了一条选择理由：FlashInfer 0.6.1 在 Hopper Attention kernel 上出现性能回归，因此默认选择 FA3，引用 issue #17411。

该回归报告使用 8×H20、Qwen3-235B-A22B-Instruct-2507-FP8：

| 指标     | FlashInfer 0.5.3 | FlashInfer 0.6.1 |
| -------- | ---------------: | ---------------: |
| 中位 ITL |         39.44 ms |        283.52 ms |
| 输出吞吐 |  1015.60 token/s |   443.42 token/s |

这些数值来自 issue 报告者提供的历史测试，本文没有独立复现。它比较的是 FI 版本，不是 FA3 与 FI 的直接比较；其意义在于，它被默认选择源码引用，提供了这项工程决策的历史依据。不能据此断言后续版本仍存在同样回归。

参考：[回归报告 #17411](https://github.com/sgl-project/sglang/issues/17411)、[默认选择源码](https://github.com/sgl-project/sglang/blob/043cfc4d3a2cc2a931d15862f08c4b60b692c957/python/sglang/srt/arg_groups/model_override_base.py)。

功能也会影响默认选择，例如 speculative topk、非对称 K/V 维度、SM120 支持范围，以及 Whisper cross-attention 的 CUDA Graph 支持。

上述指南没有提供覆盖全部分支、统一版本与负载的完整横向测试。默认规则综合了实测经验、功能支持和历史回归，不意味着每种工作负载下都最快，也不是启动时现场 benchmark。

### 10.2 功能矩阵与 page size 的取舍

选择后端时，需要同时检查 KV 量化格式、原生分页、滑动窗口、投机解码和多模态实现。一个后端支持这些功能中的几项，不代表支持它们的任意组合。

Page size 也会影响选择：小页有利于细粒度前缀复用，大页通常有利于 Attention kernel 性能。例如 page size 为 64 时，65 个 token 的前缀只有完整的前 64 个 token 可作为一页匹配；余下 1 个 token 尚未构成完整可匹配页，并不是从请求上下文删除。

部分后端要求固定原生页大小，例如 FlashMLA 为 64，Cutlass MLA 为 128，TRTLLM MLA 为 32 或 64。包装层通过展开索引模拟分页，与 kernel 原生支持分页也要区分。

参考：[SGLang 功能矩阵与分页说明](https://github.com/sgl-project/sglang/blob/043cfc4d3a2cc2a931d15862f08c4b60b692c957/docs/docs/advanced_features/attention_backend.mdx#support-matrix)。

### 10.3 Prefill 与 decode 可以分别配置

SGLang 支持实验性的 hybrid attention：

```text
--attention-backend            基础后端
--prefill-attention-backend    覆盖 prefill 后端
--decode-attention-backend     覆盖 decode 后端
```

没有覆盖的阶段继承基础后端。两个阶段使用不同实现时，由 hybrid wrapper 分阶段调用。文档给出的 Blackwell 示例是 DeepSeek-R1-FP4 使用 FA4 做 prefill、TRTLLM MLA 做 decode。

投机解码还要看 `--speculative-attention-mode`：`decode` 模式让 draft/verify 使用 decode 后端，`prefill` 模式使用 prefill 后端。因此“验证多个 token”不能直接等同于“必定使用 prefill 后端”。

这里的 hybrid 指阶段采用不同 Attention 实现，不能与“同一模型混合线性注意力层和 full attention 层”混淆。GDN 等线性注意力层另由 `--linear-attn-backend` 一类参数控制。

参考：[Hybrid attention](https://github.com/sgl-project/sglang/blob/043cfc4d3a2cc2a931d15862f08c4b60b692c957/docs/docs/advanced_features/attention_backend.mdx#hybrid-attention-different-backends-for-prefill-vs-decode-experimental)。

### 10.4 具体例子：分页 KV + 多分支投机解码为什么选 FlashInfer

考虑 B200 上的普通 MHA 模型，需要如下功能组合。这是根据官方约束构造的配置例子，不是一次实测记录：

```text
page_size = 64
speculative_eagle_topk = 4
```

普通情况下，B200 优先使用 TRTLLM MHA；但文档明确规定，在所述 hybrid/speculative 组合中，任何阶段采用 `trtllm_mha` 都只支持 `topk=1`，不能满足这里的多分支需求。

多分支投机验证涉及树状可见关系，例如：

```text
已有上下文
├── A → A1
└── B → B1
```

验证 A1 时可以看上下文和祖先 A，不能看另一分支的 B。因此实现需要正确处理分支可见关系，并与分页 KV 寻址配合。

官方文档在 hybrid attention 的投机解码约束中明确写出：对分页 MHA，`page_size>1` 且 `topk>1` 时只支持 FlashInfer。因此这组需求应选择兼容的 FlashInfer 路径；不能继续套用普通 B200 的 TRTLLM MHA 默认值。

这里能证明的是当前 SGLang 集成的功能兼容性，不是 FlashInfer 在所有场景下更快，也不是其他库从算法上永远无法支持树状 Attention。

参考：[投机解码与 hybrid attention 的约束](https://github.com/sgl-project/sglang/blob/043cfc4d3a2cc2a931d15862f08c4b60b692c957/docs/docs/advanced_features/attention_backend.mdx#speculative-decoding-with-hybrid-attention)。

## 11. 性能：先看瓶颈，再看对比数据

不能仅根据“统一调用”或“分开调用”推断性能。两个后端还同时改变 kernel、任务划分、布局和 CPU 开销。

Prefill 通常有较多 query，一块加载进来的 K/V 可以服务多个 query，较容易发挥矩阵计算能力。普通 decode 的 query 很少，却仍需读取历史 KV，因此更关注 KV 读取效率、head 间复用和任务并行度。这解释了为什么 PackGQA 和 split-KV 对少 query 场景有价值：前者让共享 KV 的多个 Q heads 一起组织计算，后者在任务不足时沿 KV 方向增加并行。

这些是优化方向，不是固定的瓶颈分类。Batch、上下文长度、GQA 比例、KV 精度、GPU 和实现方式都会改变计算与访存的比例；split-KV 也会引入局部结果写入和归并开销，并非分得越多越快。

### 11.1 H20 的 Attention 算子对比

vLLM 2026 年 7 月官方博客表 4 中的部分结果，耗时越低越好：

| 工作负载                          | FlashAttention | FlashInfer |
| --------------------------------- | -------------: | ---------: |
| Prefill，1 请求，Q=2K             |       0.574 ms |   0.609 ms |
| Prefill，1 请求，Q=8K             |       7.957 ms |   8.084 ms |
| Extend，2 请求，Q=1K、序列长度=4K |       1.830 ms |   1.829 ms |
| Decode，16 请求，序列长度=2K      |       0.098 ms |   0.052 ms |
| Decode，64 请求，序列长度=4K      |       0.620 ms |   0.340 ms |

这是 Attention 算子耗时，不是整模型生成耗时。原文表 3 的另一组混合 KV 长度、FP8 decode 测试中，FlashAttention 又快于 FlashInfer。两张表的具体测试条件不能当成同一受控实验直接混用。

参考：[vLLM × HPC-Ops，表 3、4](https://vllm-project.github.io/2026/07/06/vllm-hpc-ops.html)。

### 11.2 历史端到端集成结果

FlashInfer MLSys 2025 论文附录 G.4，在固定请求率 16 请求/秒的 vLLM 测试中：

| KV 精度  | vLLM 默认后端中位 ITL | FlashInfer 中位 ITL |
| -------- | --------------------: | ------------------: |
| BF16     |              10.42 ms |            10.63 ms |
| FP8 E4M3 |              12.56 ms |            10.92 ms |

该结果中 BF16 略慢，FP8 ITL 降低约 13%。论文标注的是当时 vLLM 默认后端，不能直接替换成当前 FA3/FA4 的比较结论。

参考：[论文附录 G.4、表 8](https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Supplemental-Conference.pdf)。

### 11.3 实际部署应比较什么

保持模型、版本、GPU、KV 精度、chunk 大小和请求分布一致，只切换后端，关注：

- TTFT：首 token 等待时间。
- ITL：相邻输出 token 的时间间隔，关注分布和 P99；TPOT 通常按请求计算首 token 之后的平均每 token 耗时，具体口径以测试工具为准。
- 满足延迟约束时的输出吞吐。

算子更快，不代表整体同比例加速。例如 Attention 只占耗时 20%，即使加速两倍，总耗时也只是从 100 降到 90，整体约加速 1.11 倍。

测试时还应固定前缀缓存命中条件、并发或到达率、输入/输出长度分布、并行配置与预热过程，并记录实际命中的后端、CUDA Graph 模式和依赖版本。先验证输出正确性，再分别比较算子耗时、整模型执行和在线服务指标，才能判断收益来自哪里。

## 12. 调优：把执行机制转化为排查顺序

### 12.1 token budget 不等于执行成本

```text
A：decode 1 token，已有上下文 1K
B：decode 1 token，已有上下文 128K
```

两者消耗同样数量的 query token 预算，但 B 需要读取更多 KV。

对普通 full attention，粗略地：

```text
线性层计算量 ∝ 本轮 token 数
Attention 计算量 ∝ 各请求 query 数 × 可见 KV 长度
```

相同 scheduled token 数的 batch，耗时可以相差很大。

### 12.2 chunking 不把完整 prefill 的 Attention 复杂度变成线性

后续 chunk 仍需与历史 KV 做 Attention。复用 KV 避免的是旧 token 的 K/V 投影重算，不是省略新 query 与旧 key 的计算。

普通 dense causal attention 的完整 prefill 总计算量仍大致为 `O(N²)`。Chunking 改变执行粒度、调度机会和中间内存需求。

### 12.3 三种“块”不要混淆

| 概念           | 含义                             |
| -------------- | -------------------------------- |
| KV page/block  | 显存分配、定位、复用单位         |
| Attention tile | kernel 搬运和计算 Q/K/V 的单位   |
| Prefill chunk  | 调度器本轮推进的 prompt token 段 |

它们可能有对齐约束，但不是同一个参数。Paged KV 也不意味着稀疏 Attention；普通 full attention 仍访问所有可见 KV。

### 12.4 chunk 大小是吞吐与响应延迟的取舍

```text
chunk 更大：矩阵规模更大、轮次更少，可能提高 prefill 效率；单轮更长。
chunk 更小：更频繁提供调度机会；启动和调度开销更多，计算效率可能下降。
```

decode 优先获得预算后，仍要等待相关计算完成。混合 batch 中 prefill 太多，仍可能拉高 ITL。

### 12.5 并发、KV 容量与排队

增加并发有助于做大 batch，但也增加 KV 占用。长输入和长输出都可能造成缓存压力，影响请求准入、抢占和恢复。

排查可以按以下方向：

```text
主要在排队？
  → 看负载、并发、KV 容量。

已经执行，但每轮很慢？
  → 看 batch 形状、Attention / MLP / 通信占比。

平均速度正常，偶尔卡顿？
  → 看长 prefill、混合长度、抢占和 P99。
```

### 12.6 用同一组配置把问题定位到对应层次

先记录本轮请求数 B、实际输入 token 数 T、各请求的 query/KV 长度，以及图重放时的补齐尺寸。再结合 profiler 判断时间主要花在 Attention、线性层、通信，还是 CPU 提交与调度。

如果主要耗时来自长 KV 的 decode，就检查后端的访存与并行划分；如果是小批次提交开销，就检查 CUDA Graph 是否命中；如果是长 prefill 拉高单轮耗时，就检查 chunk 预算与延迟目标。只有确认瓶颈所在，切换后端或调整尺寸才有明确的验证方向。
