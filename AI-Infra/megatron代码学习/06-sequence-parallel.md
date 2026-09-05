# 06. Sequence Parallel 带读

这一节接在 Tensor Parallel、TransformerLayer、普通 DDP 和 Distributed Optimizer 后面。

前面你已经读到：

```text
TP:
  把 linear 权重和 hidden 维相关计算切到 tensor parallel ranks 上

DDP / Distributed Optimizer:
  处理 data parallel ranks 之间的梯度同步和 optimizer state 分片
```

现在补一个和 TP 绑定很紧的能力：

```text
Sequence Parallel
```

这一节只读 dense GPT 主线里的 SP。先不要展开：

- context parallel
- MoE expert parallel
- inference 专用 SP 路径
- Transformer Engine 特化实现
- FP8 / NVFP4
- distributed checkpointing

建议打开这些文件对照读：

- `megatron/core/tensor_parallel/mappings.py`
- `megatron/core/models/common/embeddings/language_model_embedding.py`
- `megatron/core/tensor_parallel/layers.py`
- `megatron/core/models/gpt/gpt_model.py`
- `megatron/core/distributed/finalize_model_grads.py`

---

## A. 不看代码版：Sequence Parallel 运行过程

Sequence Parallel 解决的问题是：Tensor Parallel 已经切了矩阵，但很多中间 activation 仍然在每张 TP 卡上重复保存，序列长时很占显存。

它的做法是在 TP group 内把 sequence 维切开：

```text
原始 hidden: [S, B, H]
TP=4 后每张卡只保留:
  [S/4, B, H]
```

注意，SP 不是新的模型并行组，它依赖 TP group。可以理解成 TP 的补充：TP 切 hidden/矩阵，SP 切 sequence activation。

运行时会在需要完整 sequence 的地方 gather，在可以分片保存的地方 reduce-scatter 回来：

```text
sequence shard
  -> gather 成完整 sequence 去做某些线性/attention 相关计算
  -> 计算后 reduce-scatter 回 sequence shard
  -> 每张卡继续只保存自己那段 token
```

它主要收益是降低 activation 显存，而不是改变模型数学结果。

和 Context Parallel 的区别也要记住：

```text
SP:
  在 TP group 内切 layer 间 activation 的 sequence 维。
  主要为省 activation 显存。

CP:
  把长上下文 token 分到 CP ranks。
  主要为长序列 attention 计算和存储服务。
```

不看代码时记住：SP 是“同一层内/层间 activation 少存一部分 sequence”，常和 TP 绑定出现。

## B. 代码带读版：Sequence Parallel 实现路径

### B1. 先把 SP 要解决的问题说清楚

Megatron Core 里 transformer 主线常用张量布局是：

```text
[S, B, H]
```

其中：

```text
S: sequence length
B: batch size
H: hidden size
```

Tensor Parallel 主要切的是权重矩阵和 hidden 维相关计算。

比如 `ColumnParallelLinear` 把输出 hidden 维切开：

```text
Y = X A

A 按列切:
  A = [A0, A1, A2, A3]

rank 0 输出 H/4
rank 1 输出 H/4
rank 2 输出 H/4
rank 3 输出 H/4
```

但是很多 activation 仍然可能在每个 TP rank 上保留完整 sequence：

```text
每个 TP rank:
  [S, B, H]
```

Sequence Parallel 的目标是把一些 activation 沿 sequence 维也切开：

```text
TP size = 4

非 SP:
  每个 TP rank 持有 [S, B, H]

SP:
  每个 TP rank 持有 [S/4, B, H]
```

所以 SP 主要是省 activation 显存，不是 optimizer state 显存。

---

### B2. SP 依赖 TP

先看 `megatron/core/model_parallel_config.py`。

配置检查里有一个直接约束：

```python
if self.sequence_parallel:
    if self.tensor_model_parallel_size <= 1:
        raise ValueError("Cannot use sequence parallelism without tensor parallelism")
```

这说明 SP 不是单独的一种 parallel group。它复用 TP group，在 TP ranks 之间切 sequence。

可以先记住这个关系：

```text
TP group:
  同一层 linear 的不同 hidden/weight partition ranks

SP:
  在同一个 TP group 内，把 activation 的 sequence 维切开
```

因此这节读 SP，本质上还是在读 tensor parallel 通信。

---

### B3. 三个核心通信函数

SP 的基础操作在 `megatron/core/tensor_parallel/mappings.py`。

先只看三个 wrapper：

```python
scatter_to_sequence_parallel_region(input_)
gather_from_sequence_parallel_region(input_)
reduce_scatter_to_sequence_parallel_region(input_)
```

它们都沿第 0 维操作，也就是 `[S, B, H]` 里的 `S` 维。

#### 3.1 scatter

`scatter_to_sequence_parallel_region()` 的 forward 是：

```text
split along first dim
```

也就是：

```text
input:
  [S, B, H]

output on each TP rank:
  [S/TP, B, H]
```

它的 backward 是 gather：

```text
forward:  split S
backward: gather S
```

#### 3.2 gather

`gather_from_sequence_parallel_region()` 的 forward 是：

```text
all-gather along first dim
```

也就是：

```text
input on each TP rank:
  [S/TP, B, H]

output:
  [S, B, H]
```

它的 backward 有两个分支。默认 `tensor_parallel_output_grad=True` 时：

```text
forward:  gather S
backward: reduce-scatter S
```

如果后续计算不是 TP 模式，则 backward 可以退化成普通 split。

#### 3.3 reduce-scatter

`reduce_scatter_to_sequence_parallel_region()` 的 forward 是：

```text
reduce + scatter along first dim
```

也就是把 TP ranks 上的结果先求和，再按 sequence 维切回每个 rank。

它的 backward 是 gather：

```text
forward:  reduce-scatter S
backward: gather S
```

这三个函数就是 SP 的基本积木。

---

### B4. embedding 后把 sequence 切开

看 `megatron/core/models/common/embeddings/language_model_embedding.py`。

embedding 输出先从 `[B, S, H]` 转成 `[S, B, H]`：

```python
embeddings = embeddings.transpose(0, 1).contiguous()
```

如果打开了 `config.sequence_parallel`，并且允许 embedding scatter：

```python
embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
    embeddings, group=self.tp_group
)
```

所以 GPT dense 主线进入 transformer block 前，hidden states 可以已经是 sequence shard：

```text
非 SP:
  hidden_states = [S, B, H]

SP:
  hidden_states = [S/TP, B, H]
```

这里有一个容易混淆的点：

```text
SP 切的是 token/sequence 维；
TP linear 仍然会继续切 hidden/weight 相关维度。
```

这两个切法叠在一起，才是 Megatron 里常见的 TP + SP 组合。

---

### B5. padding mask 也要跟着切

看 `megatron/core/models/gpt/gpt_model.py` 的 preprocess。

如果传入了 `padding_mask`，并且启用了 SP：

```python
padding_mask = (
    tensor_parallel.scatter_to_sequence_parallel_region(
        padding_mask.transpose(0, 1).contiguous()
    )
    .transpose(0, 1)
    .contiguous()
)
```

因为 hidden states 的 sequence 已经被切成 `[S/TP, B, H]`，mask 也必须切到对应 token shard。

否则当前 rank 看到的 tokens 和 mask 对不上。

---

### B6. ColumnParallelLinear：SP 时不再 all-reduce dgrad

进入 `megatron/core/tensor_parallel/layers.py`。

先看 `ColumnParallelLinear.__init__()` 里的几个判断：

```python
self.sequence_parallel = config.sequence_parallel

self.allreduce_dgrad = (
    world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
)

if self.allreduce_dgrad and self.sequence_parallel:
    raise RuntimeError(...)
```

普通 TP 下，`ColumnParallelLinear` backward 可能需要对输入梯度做 all-reduce。

SP 打开后，这条路改掉了：

```text
非 SP:
  dgrad all-reduce across TP ranks

SP:
  用 sequence 维的 gather / reduce-scatter 组合替代
```

再看 forward 里传给底层 linear 的参数：

```python
output_parallel = self._forward_impl(
    input=input_parallel,
    weight=weight,
    ...
    sequence_parallel=self.sequence_parallel,
    tp_group=self.tp_group,
)
```

真正的 autograd 细节在 `linear_with_grad_accumulation_and_async_allreduce()`，第一次读不用深入 CUDA overlap 分支。

先抓住行为：

```text
ColumnParallelLinear 在 SP 模式下知道 input 是 sequence shard；
需要完整 sequence 参与某些 wgrad/dgrad 计算时，会配合 gather/reduce-scatter。
```

---

### B7. RowParallelLinear：输出 reduce-scatter 回 sequence shard

再看 `RowParallelLinear.forward()`。

它先要求 SP 模式下输入已经是 parallel 的：

```python
if self.sequence_parallel and not self.input_is_parallel:
    raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")
```

forward 做完本地 matmul 后，普通 TP 和 SP 的输出同步不同：

```python
if self.sequence_parallel:
    output_ = reduce_scatter_to_sequence_parallel_region(
        output_parallel, group=self.tp_group
    )
else:
    output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
```

对比一下：

```text
非 SP:
  output_parallel
  -> all-reduce across TP
  -> 每个 rank 得到完整 [S, B, H]

SP:
  output_parallel
  -> reduce-scatter along S
  -> 每个 rank 得到 [S/TP, B, H]
```

这就是 SP 的关键闭环：

```text
需要完整信息时 gather；
算完后尽快 reduce-scatter 回 sequence shard。
```

---

### B8. 一层 transformer 里的直觉

把 attention/MLP 先抽象成两类 linear：

```text
ColumnParallelLinear:
  通常产生按 hidden 维切开的输出

RowParallelLinear:
  通常把 hidden partition 的结果合回来
```

不开 SP 时，可以粗略理解为：

```text
[S, B, H]
  -> ColumnParallelLinear
  -> [S, B, H/TP]
  -> RowParallelLinear
  -> [S, B, H]
```

开 SP 后，主线更像：

```text
[S/TP, B, H]
  -> 必要时 gather sequence
  -> ColumnParallelLinear / local compute
  -> RowParallelLinear
  -> reduce-scatter sequence
  -> [S/TP, B, H]
```

不要把这理解成每个算子永远只看 `[S/TP, B, H]`。

更准确的说法是：

```text
SP 让层与层之间尽量保存 sequence shard；
遇到需要完整 sequence 的计算，再临时 gather；
算完后再 scatter 或 reduce-scatter 回 shard。
```

---

### B9. output layer 前后的 gather

再看 `megatron/core/models/gpt/gpt_model.py`。

GPT forward 最后会调用：

```python
logits, _ = self.output_layer(
    hidden_states,
    weight=output_weight,
    runtime_gather_output=runtime_gather_output,
)
```

`output_layer` 通常也是 tensor parallel linear。SP 打开时，它可能持有 sequence shard。

在一些 inference 场景里，代码会显式 gather：

```python
hidden_states = gather_from_sequence_parallel_region(
    hidden_states, group=self.pg_collection.tp
)
self.output_layer.sequence_parallel = False
```

这里的注释说明了原因：要从完整 packed logits 视图里切 last token logits。

训练主线里不需要先展开 inference 分支，只需要知道：

```text
如果后续逻辑需要完整 sequence 视图，就必须 gather SP shard。
```

loss、logits、packed sequence 这些位置，经常会触发这种需求。

---

### B10. LayerNorm 参数梯度为什么要额外 all-reduce

SP 还有一个非常容易漏掉的点：LayerNorm 的 weight / bias 参数没有按 sequence 切。

看 `megatron/core/fusions/fused_layer_norm.py`：

```python
self.sequence_parallel = self.config.sequence_parallel
setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
```

LayerNorm 参数是按 hidden 维的向量：

```text
weight: [H]
bias:   [H]
```

它们不是 `[S, B, H]` activation，所以不会按 sequence shard 拆成不同参数。

但是在 SP 模式下，每个 TP rank 只处理一部分 tokens：

```text
rank 0: tokens 0 ... S/TP
rank 1: tokens S/TP ... 2S/TP
...
```

每个 rank 算出来的 LayerNorm weight/bias 梯度只覆盖自己那部分 tokens 的贡献。

所以在 `megatron/core/distributed/finalize_model_grads.py` 里，需要额外同步这类参数梯度：

```python
elif config.sequence_parallel and getattr(param, "sequence_parallel", False):
    grads_sum.append(grad.data)
...
torch.distributed.all_reduce(coalesced, op=torch.distributed.ReduceOp.SUM, group=tp_group)
```

这不是 DP 梯度同步，而是 TP group 内的额外同步。

可以这样记：

```text
activation 被 sequence 切开；
LayerNorm 参数没有被 sequence 切开；
所以 LayerNorm 参数梯度要在 TP ranks 间求和。
```

---

### B11. SP 和 DP / Distributed Optimizer 的边界

第 05 节讲的是 DP 方向：

```text
不同 DP rank 看到不同 data batch；
梯度要在 DP group 内同步。
```

这一节的 SP 是 TP 方向：

```text
同一个 batch 的 sequence 被切到 TP ranks；
activation 和部分参数梯度要在 TP group 内通信。
```

两者不要混在一起：

```text
DP / DDP:
  group = data parallel group
  目标 = replicas 梯度一致

SP:
  group = tensor parallel group
  目标 = activation 沿 sequence 维分片以省显存
```

Distributed Optimizer 的 reduce-scatter 是沿 DP group 做 optimizer shard。

Sequence Parallel 的 reduce-scatter 是沿 TP group 做 activation shard。

名字相同，语义不同：

```text
DistOpt reduce-scatter:
  grad buffer -> DP shard

SP reduce-scatter:
  activation/output -> sequence shard
```

---

### B12. 把整条链串起来

SP 在 dense GPT 主线里可以先这样串：

```text
input ids
  |
  v
word + position embedding
  |
  v
[S, B, H]
  |
  v
scatter_to_sequence_parallel_region
  |
  v
[S/TP, B, H] on each TP rank
  |
  v
TransformerLayer
  |
  +--> ColumnParallelLinear / RowParallelLinear
  |      |
  |      +--> gather when full sequence view is needed
  |      +--> reduce-scatter back to sequence shard
  |
  +--> LayerNorm
         |
         +--> param grads marked sequence_parallel
         +--> finalize_model_grads all-reduce across TP group
  |
  v
output / loss
  |
  +--> gather sequence when full view is required
```

从显存角度看，核心收益是：

```text
层与层之间的大 activation:
  [S, B, H]

变成:
  [S/TP, B, H]
```

---

### B13. 读源码时的最短路径

第一次读 SP，不建议全仓库搜索后一路追到 MoE、inference、TE。

按这个顺序读就够：

```text
1. mappings.py
   - scatter_to_sequence_parallel_region
   - gather_from_sequence_parallel_region
   - reduce_scatter_to_sequence_parallel_region

2. language_model_embedding.py
   - embedding 后 scatter sequence

3. tensor_parallel/layers.py
   - ColumnParallelLinear.sequence_parallel
   - RowParallelLinear.reduce_scatter_to_sequence_parallel_region

4. gpt_model.py
   - padding_mask 如何跟着 scatter
   - output 前何时需要 gather

5. finalize_model_grads.py
   - sequence_parallel 参数梯度在 TP group 内 all-reduce
```

读的时候只盯住两个坐标系：

```text
activation shape:
  [S, B, H] 或 [S/TP, B, H]

communication group:
  TP group，不是 DP group
```

---

### B14. 自检问题

读完后可以用这些问题检查自己有没有读通：

1. Sequence Parallel 切的是哪个维度？
2. 为什么 `sequence_parallel=True` 要求 `tensor_model_parallel_size > 1`？
3. `scatter_to_sequence_parallel_region` 的 forward 和 backward 分别做什么？
4. `gather_from_sequence_parallel_region` 默认 backward 为什么是 reduce-scatter？
5. `RowParallelLinear` 在 SP 模式下为什么用 reduce-scatter，而不是 all-reduce？
6. LayerNorm weight/bias 没有按 sequence 切，为什么它们的梯度还要在 TP group 内 all-reduce？
7. Distributed Optimizer 的 reduce-scatter 和 Sequence Parallel 的 reduce-scatter 有什么区别？

---

### B15. 下一步带读建议

下一节有两个自然方向。

如果继续补并行策略，建议进入：

```text
Context Parallel
```

因为 CP 也切 sequence/token 相关维度，但它解决的是更长上下文下 attention 计算和 KV 相关通信，不要和 SP 混为一谈。

如果回到训练系统完整性，建议进入：

```text
Distributed Checkpointing
```

因为第 05 节已经引入 optimizer state shard，后面必须理解 shard state 如何保存、加载和 reshard。

建议优先读 Context Parallel。这样可以先把 TP / SP / CP 的边界分清，再回头看 checkpoint 会更稳。

