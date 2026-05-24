# 03. TransformerLayer 带读

这一篇承接 `02-tensor-parallel-linear.md`。上一轮已经看清楚了两个最常见的并行线性层：

```text
QKV projection / MLP FC1: ColumnParallelLinear
attention projection / MLP FC2: RowParallelLinear
```

这一轮先回到一个完整的 decoder layer，看 Megatron Core 如何把：

```text
LayerNorm
SelfAttention
Bias-Dropout-Add
LayerNorm
MLP
Bias-Dropout-Add
```

串成 `TransformerLayer`。

本篇仍然只读 dense GPT 的单层主线。先跳过 cross-attention、MoE、Transformer Engine、FP8/FP4、inference optimized、CUDA graph、fine-grained activation offload、pipeline parallel 等分支。

主线文件：

- `megatron/core/transformer/transformer_layer.py`
- `megatron/core/fusions/fused_bias_dropout.py`

## A. 不看代码版：TransformerLayer 运行过程

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

## B. 代码带读版：TransformerLayer 实现路径

### B1. TransformerLayer.forward 的主线

入口：

- `megatron/core/transformer/transformer_layer.py`
- 重点类：`TransformerLayer`
- 重点函数：`forward()`、`_forward_attention()`、`_forward_mlp()`、`_forward_post_mlp()`

`TransformerLayer.forward()` 本身很短，主线可以简化成：

```text
hidden_states, context = _forward_attention(...)
output = _forward_mlp(hidden_states, ...)
return output, context
```

对 GPT dense decoder 来说，`context` 通常不用关注，核心是 hidden states 如何穿过 attention 和 MLP 两段。

可以先把一个 layer 理解成：

```text
hidden_states [S, B, H]
  -> input_layernorm
  -> self_attention
  -> self_attn_bda
  -> pre_mlp_layernorm
  -> mlp
  -> mlp_bda
output [S, B, H]
```

注意这里是 pre-norm 结构：每个子层计算前先做 LayerNorm，子层输出再和 residual 做 BDA。

### B2. Attention 段：LayerNorm、SelfAttention、BDA

`_forward_attention()` 的第一步是 input layernorm：

```text
input_layernorm_output = input_layernorm(hidden_states)
residual = hidden_states
```

如果 layernorm 模块返回 tuple，代码会从 tuple 里取出新的 `residual`。普通 dense 主线里可以先记成：

```text
归一化后的张量进入 attention；
归一化前的 hidden_states 保留为 residual。
```

然后进入 self attention：

```text
attention_output_with_bias = self.self_attention(
    input_layernorm_output,
    attention_mask=attention_mask,
    rotary_pos_emb=rotary_pos_emb,
    ...
)
```

上一轮已经读过 attention 内部：

```text
input_layernorm_output [S, B, H]
  -> ColumnParallelLinear QKV
  -> DotProductAttention
  -> RowParallelLinear projection
attention_output_with_bias = (attention_output, bias)
```

attention 返回后，马上进入第一处 BDA：

```text
hidden_states = self.self_attn_bda(
    self.training,
    self.config.bias_dropout_fusion,
)(attention_output_with_bias, residual, self.hidden_dropout)
```

这一步的语义是：

```text
hidden_states = residual + dropout(attention_output + bias)
```

如果 attention 没有 bias，就是：

```text
hidden_states = residual + dropout(attention_output)
```

所以 attention 段的完整主线是：

```text
hidden_states
  -> input_layernorm
normed_hidden
  -> self_attention
attention_output, attention_bias
  -> bias_dropout_add with residual
hidden_states
```

这里的 `hidden_states` 已经包含 attention 子层的 residual 更新，会作为 MLP 段的输入。

### B3. MLP 段：第二个 LayerNorm 和第二个 BDA

`_forward_mlp()` 先调用：

```text
pre_mlp_layernorm_output = _forward_pre_mlp_layernorm(hidden_states)
residual = hidden_states
```

这和 attention 段是同一个模式：

```text
归一化后的张量进入 MLP；
归一化前的 hidden_states 保留为 residual。
```

然后进入 MLP：

```text
mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
```

上一轮的 tensor parallel 视角里，MLP 是：

```text
pre_mlp_layernorm_output [S, B, H]
  -> ColumnParallelLinear FC1
intermediate [S, B, ffn_hidden / tp]
  -> activation / GLU
  -> RowParallelLinear FC2
mlp_output_with_bias = (mlp_output, mlp_bias)
```

MLP 返回后进入 `_forward_post_mlp()`，做第二处 BDA：

```text
hidden_states = self.mlp_bda(
    self.training,
    self.config.bias_dropout_fusion,
)(mlp_output_with_bias, residual, self.hidden_dropout)
```

语义同样是：

```text
hidden_states = residual + dropout(mlp_output + bias)
```

最后代码会用 `make_viewless_tensor(...)` 包一下输出，避免后续 checkpoint / pipeline 逻辑拿到 view tensor 时出现不必要的问题。第一轮阅读只需要知道：它不改变主线语义，输出仍然是 `[S, B, H]`。

### B4. BDA 到底做了什么

入口：

- `megatron/core/fusions/fused_bias_dropout.py`
- 重点函数：`_bias_dropout_add_func()`、`get_bias_dropout_add()`

BDA 是 Megatron 里常见的缩写：

```text
Bias + Dropout + Add residual
```

核心函数先拆开输入：

```text
x, bias = x_with_bias
```

如果有 bias：

```text
x = x + bias
out = dropout(x)
out = residual + out
```

如果没有 bias：

```text
out = dropout(x)
out = residual + out
```

`get_bias_dropout_add(training, fused)` 只是在选择具体实现：

```text
fused=True, training=True   -> bias_dropout_add_fused_train
fused=True, training=False  -> bias_dropout_add_fused_inference
fused=False                 -> bias_dropout_add_unfused(training)
```

这些实现的主线语义一样，差别主要是训练/推理 dropout 语义和是否使用 JIT fusion。

还有一个容易忽略的点：如果开启 `fp32_residual_connection`，residual 可能是 fp32。BDA 里会把 `x` 和 `bias` 转到 residual 的 dtype，保证 residual stream 的精度一致。

### B5. 一个完整 decoder layer 的总图

把前面几节合在一起，一个 dense GPT decoder layer 可以画成：

```text
input hidden_states [S, B, H]
  |
  | residual_1 = hidden_states
  v
input_layernorm
  |
  v
SelfAttention
  ColumnParallelLinear QKV
  DotProductAttention
  RowParallelLinear projection
  |
  v
self_attn_bda
  residual_1 + dropout(attention_output + bias)
  |
  | residual_2 = hidden_states
  v
pre_mlp_layernorm
  |
  v
MLP
  ColumnParallelLinear FC1
  activation / GLU
  RowParallelLinear FC2
  |
  v
mlp_bda
  residual_2 + dropout(mlp_output + bias)
  |
  v
output hidden_states [S, B, H]
```

这也是前几篇内容第一次真正闭环：

```text
SelfAttention 解释 token 如何互相看；
Tensor Parallel 解释 QKV/MLP 的线性层如何切；
TransformerLayer 解释 attention 和 MLP 如何通过 norm/residual/BDA 串成 decoder block。
```

### B6. 第三轮总图

```text
单个 TransformerLayer:

hidden [S, B, H]
  -> LN
  -> SelfAttention
  -> BDA(residual)
  -> LN
  -> MLP
  -> BDA(residual)
  -> hidden [S, B, H]


```

### B7. 建议带着这些问题复读

1. 为什么 `TransformerLayer` 里 attention 和 MLP 前面都要各自做一次 LayerNorm？
2. `self_attn_bda` 和 `mlp_bda` 的 residual 分别来自哪里？
3. BDA 的公式为什么可以写成 `residual + dropout(x + bias)`？
4. 为什么说 dense GPT 主线里 `TransformerLayer` 是 pre-norm block？

### B8. 下一步带读建议

下一段建议读 Pipeline Parallel，重点从“单层怎么计算”切到“多层怎么分布到多个 pipeline ranks 上执行”。

建议下一篇专门读：

- `megatron/core/transformer/transformer_block.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/pipeline_parallel/p2p_communication.py`

目标是把一组 `TransformerLayer` 如何切到不同 PP stage，以及一个 microbatch 如何沿 forward/backward 方向流动读清楚。

