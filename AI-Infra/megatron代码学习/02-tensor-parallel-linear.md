# 02. Tensor Parallel 线性层带读

这一篇承接 `01-gpt-mainline.md`。GPT 主线里会反复看到这两个模式：

```text
QKV projection / MLP FC1: ColumnParallelLinear
attention projection / MLP FC2: RowParallelLinear
```

这一轮只读 tensor parallel 在线性层里的主线。先跳过 MoE expert、embedding wgrad deferral、CPU offload、inference optimized all-gather、sequence parallel 的细节。目标是回答两个问题：

```text
ColumnParallelLinear 为什么适合扩维？
RowParallelLinear 为什么适合收回 hidden size？
```

## A. 不看代码版：Tensor Parallel 线性层运行过程

Tensor Parallel 解决的问题是：单张 GPU 算不下一个很大的线性层时，把 hidden 相关的矩阵维度切到多张 GPU 上一起算。

它不主要切 batch，也不主要切 sequence，而是切线性层里的大矩阵：

```text
输入 hidden
  -> 多张 GPU 各算一部分矩阵乘法
  -> 必要时通信合并结果
  -> 输出 hidden
```

GPT 里最常见的是一前一后两种线性层：

```text
ColumnParallelLinear:
  适合把输出维度变大。
  例如 QKV projection、MLP FC1。

RowParallelLinear:
  适合把已经切开的中间结果收回 hidden size。
  例如 attention output projection、MLP FC2。
```

ColumnParallelLinear 的直觉是“每张卡负责一部分输出列”：

```text
hidden [S, B, H]
  -> GPU0 算 output 的前 1/p
  -> GPU1 算 output 的后 1/p
  -> 得到 local output [S, B, O/p]
```

如果下游能继续消费这份分片，就不用马上合并；如果下游需要完整输出，就 all-gather。

RowParallelLinear 的直觉是“每张卡拿一部分输入，算出对完整输出的贡献”：

```text
input partition [S, B, I/p]
  -> 每张卡算 partial output [S, B, O]
  -> 多张卡把 partial output 相加
  -> output [S, B, O]
```

所以 RowParallelLinear 通常需要 all-reduce，因为每张卡算的是完整输出的一部分贡献，不是完整输出本身。

放回 GPT 主线就是：

```text
Attention:
  hidden
    -> ColumnParallelLinear 生成一部分 Q/K/V heads
    -> attention core 在本卡分片上计算
    -> RowParallelLinear 合并各卡贡献
    -> attention output

MLP:
  hidden
    -> ColumnParallelLinear 扩到 ffn_hidden 的一部分
    -> activation
    -> RowParallelLinear 收回 hidden size
    -> mlp output
```

不看代码时，只需要记住：

> ColumnParallelLinear 负责“分头扩出去”；RowParallelLinear 负责“把分片贡献加回来”。

读完这一节应该能回答：

1. Tensor Parallel 为什么切 hidden/矩阵维度，而不是简单切 batch？
2. ColumnParallelLinear 的输出为什么是分片？
3. RowParallelLinear 为什么需要把各卡结果相加？
4. 为什么 QKV / FC1 用 ColumnParallelLinear？
5. 为什么 attention projection / FC2 用 RowParallelLinear？

## B. 代码带读版：Tensor Parallel 线性层实现路径

主线文件：

- `megatron/core/tensor_parallel/layers.py`
- `megatron/core/tensor_parallel/mappings.py`
- `megatron/core/models/gpt/gpt_layer_specs.py`

### B1. 先看 GPT 里哪里用它们

入口：

- `megatron/core/models/gpt/gpt_layer_specs.py`
- 重点函数：`get_gpt_layer_local_submodules()`

GPT dense local spec 里，attention 的典型组合是：

```text
SelfAttention
  linear_qkv  = ColumnParallelLinear
  linear_proj = RowParallelLinear
```

MLP 的典型组合是：

```text
MLP
  linear_fc1 = ColumnParallelLinear
  linear_fc2 = RowParallelLinear
```

这就是 Megatron Core 里很核心的并行直觉：

```text
先把 hidden 扩出去：ColumnParallelLinear
再把分片结果收回来：RowParallelLinear
```

### B2. Tensor Parallel 切的是什么

先用普通线性层表示：

```text
Y = X A + b
```

在 PyTorch 实现里，`torch.nn.functional.linear` 实际使用的是 `weight` 的转置关系，所以代码里的 weight shape 会看起来像：

```text
weight: [out_features, in_features]
```

Tensor Parallel 的核心不是切 batch，也不是切 sequence，而是切 hidden 相关的矩阵维度。

两种切法：

```text
ColumnParallelLinear:
  切 weight 的 output dimension
  每个 TP rank 只算一部分 output features

RowParallelLinear:
  切 weight 的 input dimension
  每个 TP rank 只消费一部分 input features
```

### B3. ColumnParallelLinear：切输出维

入口：

- `megatron/core/tensor_parallel/layers.py`
- 重点类：`ColumnParallelLinear`

类注释里的数学表达是：

```text
A = [A_1, ..., A_p]
Y_i = X A_i
```

代码里对应：

```text
self.output_size_per_partition = divide(output_size, world_size)
weight shape = [output_size_per_partition, input_size]
bias shape   = [output_size_per_partition]
```

也就是说，如果全量线性层是：

```text
input:  [S, B, H]
output: [S, B, O]
```

TP world size 为 `p` 时，每个 rank 只产生：

```text
local output: [S, B, O / p]
```

这非常适合 QKV 和 MLP FC1：

```text
QKV:
  [S, B, H] -> [S, B, qkv_projection]
  输出维变大，可以按输出维切开。

MLP FC1:
  [S, B, H] -> [S, B, ffn_hidden]
  ffn_hidden 通常比 H 大，也适合按输出维切开。
```

### B4. ColumnParallelLinear.forward 主线

简化后的 forward：

```text
input [S, B, H]
  -> copy_to_tensor_model_parallel_region(input)
  -> local linear with weight [O/p, H]
output_parallel [S, B, O/p]
  -> optional gather_from_tensor_model_parallel_region
output [S, B, O] 或 [S, B, O/p]
```

关键点是 `gather_output`：

```text
gather_output = False:
  每个 rank 保留自己的 output partition。

gather_output = True:
  all-gather 最后一维，让每个 rank 都拿到完整 output。
```

在 GPT 主线里，很多地方不立刻 gather，因为下一个模块可以继续消费 parallel output。例如：

```text
linear_qkv 输出的是每个 rank 的 QKV partition；
attention core 就在这个 partition 上继续算。
```

### B5. RowParallelLinear：切输入维

入口：

- `megatron/core/tensor_parallel/layers.py`
- 重点类：`RowParallelLinear`

类注释里的数学表达是：

```text
A = transpose([A_1, ..., A_p])
X = [X_1, ..., X_p]
```

代码里对应：

```text
self.input_size_per_partition = divide(input_size, world_size)
weight shape = [output_size, input_size_per_partition]
bias shape   = [output_size]
```

如果全量线性层是：

```text
input:  [S, B, I]
output: [S, B, O]
```

TP world size 为 `p` 时，每个 rank 消费：

```text
local input: [S, B, I / p]
```

每个 rank 都会算出一个局部贡献：

```text
local partial output: [S, B, O]
```

这些 partial output 需要做 sum-reduce：

```text
output = sum(partial_output over TP ranks)
```

这就是为什么 RowParallelLinear 的 forward 末尾会调用：

```text
reduce_from_tensor_model_parallel_region(output_parallel)
```

### B6. RowParallelLinear.forward 主线

简化后的 forward：

```text
input [S, B, I]
  -> 如果 input_is_parallel=False，先 scatter last dim
input_parallel [S, B, I/p]
  -> local linear with weight [O, I/p]
output_parallel [S, B, O]
  -> reduce_from_tensor_model_parallel_region
output [S, B, O]
```

关键点是 `input_is_parallel`：

```text
input_is_parallel = True:
  输入已经是按最后一维切好的 partition，不再 scatter。

input_is_parallel = False:
  先把输入按最后一维 scatter 到 TP ranks。
```

在 GPT 主线里，attention 的 `linear_proj` 和 MLP 的 `linear_fc2` 通常接在 parallel output 后面，所以很自然地使用：

```text
input_is_parallel = True
```

### B7. mappings.py：通信语义速查

入口：

- `megatron/core/tensor_parallel/mappings.py`

先记住这几个 wrapper 的 forward 语义：

```text
copy_to_tensor_model_parallel_region:
  forward: copy
  backward: all-reduce

gather_from_tensor_model_parallel_region:
  forward: all-gather last dim
  backward: split last dim

scatter_to_tensor_model_parallel_region:
  forward: split/scatter last dim
  backward: all-gather last dim

reduce_from_tensor_model_parallel_region:
  forward: all-reduce
  backward: copy
```

对应到两个线性层：

```text
ColumnParallelLinear:
  forward 主要产生 [S, B, O/p]
  如需完整输出，使用 gather_from_tensor_model_parallel_region

RowParallelLinear:
  forward 每个 rank 产生 [S, B, O] 的 partial output
  使用 reduce_from_tensor_model_parallel_region 把 partial outputs 求和
```

### B8. 放回 Attention 主线

GPT 主线里的 attention 可以重新理解成：

```text
hidden_states [S, B, H]
  -> ColumnParallelLinear(linear_qkv)
qkv partition [S, B, QKV/p]
  -> split Q/K/V
  -> DotProductAttention
context partition [S, B, H/p]
  -> RowParallelLinear(linear_proj)
attention output [S, B, H]
```

所以 `linear_qkv` 和 `linear_proj` 是一对配合：

```text
ColumnParallelLinear:
  把 QKV 的输出维切开，让每张卡只算一部分 heads。

RowParallelLinear:
  把每张卡的 head/context contribution reduce 回完整 hidden 输出。
```

再把 `linear_qkv` 的输出组织方式补完整。没有 output gate 时，SelfAttention 里常见的 QKV 投影输出可以理解成：

```text
[sq, b, h] -> [sq, b, ng * (np/ng + 2) * hn]
```

符号含义：

```text
sq: sequence length
b: batch size
h: hidden size
np: 当前 TP rank 上的 attention heads 数
ng: 当前 TP rank 上的 query groups 数
hn: 每个 attention head 的 hidden size，也就是 head_dim
```

为什么是 `np/ng + 2`？

```text
每个 query group 里有：
  np/ng 个 query heads
  1 个 key head
  1 个 value head
```

所以每个 group 的内容可以理解成：

```text
q...q | k | v
```

`linear_qkv` 之后，代码会先把最后一维 reshape 成 group 结构：

```text
mixed_qkv:
  [sq, b, hp]
    -> [sq, b, ng, (np/ng + 2) * hn]
```

然后按照最后一维拆开：

```text
query: [sq, b, ng, np/ng * hn]
key:   [sq, b, ng, hn]
value: [sq, b, ng, hn]
```

接着 query 会被进一步 reshape：

```text
query:
  [sq, b, ng, np/ng * hn]
    -> [sq, b, np, hn]
```

key/value 保持 group 维度：

```text
key:   [sq, b, ng, hn]
value: [sq, b, ng, hn]
```

这正是 GQA 的形状特征：query 有 `np` 个 heads，key/value 只有 `ng` 个 groups。后面进入 `DotProductAttention` 时，如果 `np > ng`，key/value 会被虚拟重复到和 query head 数匹配。

还有一个容易卡住的特殊分支：

```text
if config.num_query_groups < world_size:
```

这个情况表示 tensor parallel rank 数比全局 query groups 还多。此时单个 rank 上拿到的 QKV 分片不一定刚好是完整的 `q...q | k | v` group。

代码做了四件事：

```text
1. all-gather 拼回更完整的 mixed_qkv
2. 根据 TP rank 选择自己负责的 query group slice
3. split 出 query/key/value
4. 再从 query 中切出当前 rank 真正负责的 query heads
```

这个分支本质上是在处理 TP 切分和 GQA/MQA group 不整齐对齐的问题。普通配置下，可以先专注 `ColumnParallelLinear -> QKV partition -> split Q/K/V -> attention core -> RowParallelLinear` 这条主线。

### B9. 放回 MLP 主线

MLP 也一样：

```text
hidden_states [S, B, H]
  -> ColumnParallelLinear(linear_fc1)
intermediate [S, B, ffn_hidden/p]
  -> activation / GLU
  -> RowParallelLinear(linear_fc2)
mlp output [S, B, H]
```

直觉是：

```text
FC1 扩维，所以按输出维切。
FC2 收回 hidden，所以按输入维切，并 reduce partial outputs。
```

### B10. 第三轮总图

```text
TransformerLayer

SelfAttention:
  hidden [S, B, H]
    -> ColumnParallelLinear QKV
  qkv partition [S, B, QKV/p]
    -> attention core
  context partition [S, B, H/p]
    -> RowParallelLinear projection
  attention output [S, B, H]

MLP:
  hidden [S, B, H]
    -> ColumnParallelLinear FC1
  intermediate partition [S, B, ffn_hidden/p]
    -> activation
    -> RowParallelLinear FC2
  mlp output [S, B, H]
```

### B11. 建议带着这些问题复读

1. `ColumnParallelLinear` 的 weight 为什么是 `[output_size_per_partition, input_size]`？
2. `RowParallelLinear` 的 weight 为什么是 `[output_size, input_size_per_partition]`？
3. `gather_output=False` 时，下游模块必须满足什么条件？
4. `RowParallelLinear` 为什么需要 all-reduce？
5. attention 的 heads 如何天然适合被 TP 切开？
6. MLP 的 `ffn_hidden` 为什么也天然适合被 TP 切开？
7. GQA/MQA 下为什么 `query` heads 数可以多于 `key/value` groups？
8. `num_query_groups < world_size` 为什么需要先 all-gather 再 split？

### B12. 下一步带读建议

下一段建议读 `TransformerLayer` 里的 residual、bias-dropout-add 和 layer norm 顺序。

建议入口：

- `megatron/core/transformer/transformer_layer.py`
- `megatron/core/transformer/custom_layers/transformer_engine.py` 暂时不读
- `megatron/core/fusions/fused_bias_dropout.py`

原因是现在你已经知道 attention 和 MLP 内部怎么并行，下一步应该看一个 decoder block 如何把：

```text
LayerNorm
SelfAttention
Bias-Dropout-Add
LayerNorm
MLP
Bias-Dropout-Add
```

串成一个完整的 Transformer layer。
