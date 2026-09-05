# Megatron Core 并行训练主线：不看代码版

Megatron Core 的代码里同时交织着模型结构、并行切分、通信调度、优化器状态和低精度后端。本文按训练主线梳理这些模块各自解决的问题，以及它们如何接到同一条 GPT 训练流程上。

## 目录
- 01. GPT 模型主线：ModuleSpec 设计带读
- 02. Tensor Parallel 线性层带读
- 03. TransformerLayer 带读
- 04. Pipeline Parallel 与 1F1B Schedule 带读
- 05. Optimizer、Distributed DDP 和 Distributed Optimizer 带读
- 06. Sequence Parallel 带读
- 07. Context Parallel 带读
- 08. Communication Overlap 带读
- 09. MoE / Expert Parallel 带读
- 10. Transformer Engine / FP8

---

## 01. GPT 模型主线：ModuleSpec 设计带读

### 不看代码版：ModuleSpec 解决了什么

如果没有 `ModuleSpec`，`TransformerLayer` 里可能会写满这种判断：

```text
if use_te:
  用 TE Linear
elif use_moe:
  用 MoE MLP
elif inference:
  用 inference linear
else:
  用普通 Linear
```

这样模型结构、后端选择、并行实现、低精度实现都会混在一个类里。

Megatron 的做法是把它们拆开：

```text
GPTModel:
  负责 embedding、decoder、output/loss 的总结构。

TransformerBlock:
  负责一组 TransformerLayer。

TransformerLayer:
  负责执行一层 decoder 的固定顺序。

ModuleSpec:
  告诉 TransformerLayer 每个位置具体装哪个模块。
```

所以 `TransformerLayer` 的主结构仍然是：

```text
input_layernorm
  -> self_attention
  -> self_attn_bda
  -> pre_mlp_layernorm
  -> mlp
  -> mlp_bda
```

但这些位置里的具体实现可以替换：

```text
local:
  ColumnParallelLinear / RowParallelLinear / DotProductAttention

transformer_engine:
  TEColumnParallelLinear / TERowParallelLinear / TEDotProductAttention

moe:
  MoELayer / experts / token dispatcher

inference_optimized:
  inference 专用 linear 和 attention
```

一句话：

> `ModuleSpec` 是 Megatron 的“装配说明书”：结构不变，零件可换。

---

## 02. Tensor Parallel 线性层带读

### 不看代码版：Tensor Parallel 线性层运行过程

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

---

## 03. TransformerLayer 带读

### 不看代码版：TransformerLayer 运行过程

这一节只解决一个问题：单个 decoder layer 内部怎么把 attention 和 MLP 串起来。

单层 TransformerLayer 的输入输出都是 hidden states：

```text
hidden [S, B, H]
  -> LayerNorm
  -> SelfAttention
  -> bias/dropout/residual add
  -> LayerNorm
  -> MLP
  -> bias/dropout/residual add
  -> hidden [S, B, H]
```

Attention 负责 token 之间交换信息，MLP 负责每个 token 自己的特征加工，residual 让每一层只是在原表示上做增量修改，LayerNorm 稳定数值。

不看代码时记住：TransformerLayer 是“一个 decoder block 怎么算”。很多 decoder blocks 如何切到不同 GPU 上形成 Pipeline Parallel，放到下一节单独读。

---

## 04. Pipeline Parallel 与 1F1B Schedule 带读

### 不看代码版：Pipeline Parallel 1F1B 运行过程

1F1B 解决的问题是：pipeline stage 不能只等一个完整 batch 全部 forward 完再 backward，否则很多 GPU 会空等。

Megatron 会把一个 batch 切成多个 microbatch，让它们像流水线一样经过不同 stage：

```text
microbatch 0 -> stage0 -> stage1 -> stage2
microbatch 1 -> stage0 -> stage1 -> stage2
microbatch 2 -> stage0 -> stage1 -> stage2
```

1F1B 的意思是稳定阶段里每个 stage 尽量交替做：

```text
1 个 forward
1 个 backward
1 个 forward
1 个 backward
```

整个 schedule 分三段：

```text
warmup:
  先塞入足够多的 forward，让 pipeline 填起来。

steady 1F1B:
  一边继续 forward 新 microbatch，一边 backward 旧 microbatch。

cooldown:
  不再塞新 forward，把剩下的 backward 做完。
```

stage 之间传的是 activation 和 activation gradient：

```text
forward:  当前 stage 输出 hidden，发给下一个 stage
backward: 当前 stage 收到输出梯度，算完后把输入梯度发回上一个 stage
```

不看代码时记住：1F1B 的核心不是新模型结构，而是安排 microbatch 的前后向顺序，减少 pipeline 空泡。

---

## 05. Optimizer、Distributed DDP 和 Distributed Optimizer 带读

### 不看代码版：Optimizer 和 Distributed DDP 运行过程

这一节解决的问题是：多张 GPU 各自算了梯度，怎么把它们同步成一致的模型更新。

Data Parallel 下，每个 DP rank 都有一份相同模型，但拿到不同数据：

```text
rank0: batch shard 0 -> forward/backward -> grads0
rank1: batch shard 1 -> forward/backward -> grads1
rank2: batch shard 2 -> forward/backward -> grads2
```

这些梯度必须合并，否则每张卡会把模型更新到不同方向。普通 DDP 的核心动作是 all-reduce：

```text
grads0, grads1, grads2
  -> all-reduce / average
  -> 每个 rank 得到相同 grads
  -> optimizer step
  -> 每个 rank 参数继续一致
```

Megatron 会把参数和梯度组织进连续 buffer，再按 bucket 同步。这样做是为了减少碎片化的小通信，把很多小参数的梯度合并成较大的通信单位。

训练 step 可以简化成：

```text
forward 得到 loss
  -> backward 产生梯度
  -> 梯度进入 main_grad / bucket
  -> DDP 同步梯度
  -> finalize_model_grads 做收口处理
  -> optimizer 消费梯度并更新参数
  -> zero_grad 清理下一轮状态
```

如果有 microbatch 梯度累积，中间若干个 microbatch 先不急着同步，最后再统一同步。

不看代码时记住：DDP 保证“同一个 DP group 内，每张卡看到不同数据，但更新同一份模型”。


### 不看代码版：Distributed Optimizer 运行过程

Distributed Optimizer 解决的问题是：普通 DDP 每张卡都保存完整 optimizer state，显存浪费很大。

普通 DDP 的状态更像这样：

```text
每个 DP rank:
  完整参数
  完整梯度
  完整 optimizer state
```

Distributed Optimizer 把 optimizer 相关状态按 DP rank 切片：

```text
rank0 负责一段参数 shard 的 optimizer state
rank1 负责另一段参数 shard 的 optimizer state
rank2 负责另一段参数 shard 的 optimizer state
```

梯度同步也从 all-reduce 完整梯度，变成 reduce-scatter：

```text
所有 rank 的梯度
  -> reduce-scatter
  -> 每个 rank 只拿到自己负责的梯度 shard
  -> 只更新自己负责的参数 shard
```

更新完后，下一轮 forward 又需要完整模型参数参与计算，所以会在需要时把参数 shard all-gather 回来。

整体流程是：

```text
backward 产生完整梯度贡献
  -> reduce-scatter 得到 grad shard
  -> optimizer 只更新本 rank 的 param shard
  -> all-gather 参数供下一轮 forward 使用
```

不看代码时记住：Distributed Optimizer 不是换优化算法，而是把 optimizer state 和梯度更新责任分摊到 DP ranks 上，主要目的是省显存。

---

## 06. Sequence Parallel 带读

### 不看代码版：Sequence Parallel 运行过程

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

---

## 07. Context Parallel 带读

### 不看代码版：Context Parallel 运行过程

Context Parallel 解决的问题是：上下文太长时，单张 GPU 放不下完整序列的 attention 计算和 KV 信息。

它把长 sequence 按 token 维切到多个 CP rank：

```text
完整序列: [token0 ... token8191]
CP=4:
  rank0: token0    ... token2047
  rank1: token2048 ... token4095
  rank2: token4096 ... token6143
  rank3: token6144 ... token8191
```

每个 rank 先只持有自己那段 token 的 hidden states、position 信息和 mask。问题是 attention 不能只看本地 token，它还需要历史上下文里的 K/V。

所以 CP 的核心运行过程是：

```text
本地 token shard
  -> 本地生成 Q/K/V
  -> attention 期间跨 CP ranks 交换 K/V 或上下文信息
  -> 每个 rank 得到自己 token shard 的 attention output
```

RoPE、position ids、attention mask 也必须和 CP shard 对齐，否则不同 rank 对同一个 token 的位置理解会错。

和 Sequence Parallel 的区别：

```text
SP:
  主要在 TP group 内切中间 activation，省显存。

CP:
  直接把长上下文 token 分到 CP ranks，attention 内部要跨 rank 交换上下文。
```

不看代码时记住：CP 是“长上下文切片”，重点在 attention 如何跨 rank 拿到足够的 K/V。

---

## 08. Communication Overlap 带读

### 不看代码版：Communication Overlap 运行过程

Communication Overlap 解决的问题是：分布式训练里通信很贵，如果总是“算完再通信”，GPU 会经常等待。

它的思路是：通信尽早发起，真正需要结果时再等待。

```text
普通方式:
  计算 A
  等通信 A
  计算 B
  等通信 B

Overlap:
  计算 A
  发起通信 A，不立刻等
  同时计算 B
  到必须使用 A 的结果时再 wait
```

Megatron 里常见几类 overlap：

```text
DP 梯度 overlap:
  backward 某些梯度 ready 后就开始 reduce。

参数 gather overlap:
  下一层参数提前 all-gather，用到前再 wait。

PP P2P overlap:
  pipeline stage 的 send/recv 先发出，后面再等待。

TP Linear overlap:
  Linear 内部的 TP 通信和 GEMM 尽量重叠。
```

它不改变模型数学结果，只改变通信和计算的时间安排。

不看代码时记住：overlap 的核心是“尽早 start，尽晚 wait”。效果好不好取决于通信能不能被后续计算时间盖住。

---

## 09. MoE / Expert Parallel 带读

### 不看代码版：MoE / Expert Parallel 运行过程

MoE 解决的问题是：想增大模型参数量，但不希望每个 token 都经过所有参数计算。

普通 dense MLP 是每个 token 都走同一个 MLP：

```text
token -> MLP -> output
```

MoE 把一个 MLP 换成多个 expert，并让 router 为每个 token 选择少数几个 expert：

```text
token hidden
  -> router 打分
  -> 选择 top-k experts
  -> token 被发送到对应 expert
  -> expert 计算
  -> 结果按权重合并
```

Expert Parallel 解决的是 expert 太多时怎么放到多张 GPU 上：

```text
rank0: expert 0, 1
rank1: expert 2, 3
rank2: expert 4, 5
rank3: expert 6, 7
```

由于不同 token 会选不同 expert，MoE 中间会有一次明显的 token 重排和通信：

```text
本地 tokens
  -> router 决定每个 token 去哪里
  -> dispatcher 把 token 发到 expert 所在 rank
  -> local experts 计算
  -> combine 把结果送回原 token 顺序
```

MoE 的难点不是单个 expert 的 MLP，而是 router 是否均衡、token dispatcher 怎么通信、expert 参数和普通参数怎么同步。

不看代码时记住：MoE 是“每个 token 只激活少数 expert”；EP 是“expert 列表分布到多张卡”。

---

## 10. Transformer Engine / FP8

### 不看代码版：Transformer Engine / FP8 运行过程

Transformer Engine 解决的问题是：Megatron Core 定义了模型结构和并行语义，但底层 Linear、Attention、LayerNorm 需要更快的高性能实现。

可以这样分工：

```text
Megatron Core:
  决定模型长什么样、并行怎么切、训练流程怎么走。

Transformer Engine:
  提供更快的 Linear / Attention / Norm / Grouped GEMM kernel。
```

所以 TE 不是新模型结构，而是把关键算子换成更高性能的后端：

```text
原来的 GPT layer
  -> Linear / Attention / Norm 换成 TE 实现
  -> 上层结构仍然是 attention + MLP + residual
```

FP8/FP4 是进一步的低精度执行和参数通信机制。它不等于整个模型所有参数都变成 uint8，而是让部分 GEMM 权重、activation 或通信在 TE 管理下用低精度格式。

```text
高精度 tensor
  -> 根据 scale / amax 量化
  -> FP8/FP4 kernel 执行
  -> 必要时反量化或继续低精度流转
```

FP8 有两条线：

```text
执行线:
  fp8 autocast 控制哪些算子低精度执行。

参数线:
  fp8_param / fp8_param_gather 控制部分权重是否低精度存储和通信。
```

不看代码时记住：MCore 画计算图和并行图，TE 负责把图上的关键算子跑快，并在需要时用 FP8/FP4 跑得更省。

---
