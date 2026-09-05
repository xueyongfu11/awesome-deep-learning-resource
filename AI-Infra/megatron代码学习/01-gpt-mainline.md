# 01. GPT 模型主线：ModuleSpec 设计带读

本篇不再重复 GPT 从 token 到 loss 的基础流程，而是专门看 Megatron Core 里一个关键设计：`ModuleSpec`。

核心问题是：

```text
同样叫 GPTModel，为什么它可以被配置成 LLaMA/Qwen/Mixtral/TE/FP8/MoE/inference optimized 等不同形态？
```

答案是：

```text
GPTModel 负责总结构；
TransformerLayer 负责骨架；
ModuleSpec 负责把具体 Attention、MLP、Norm、Linear 实现注入进去。
```

## A. 不看代码版：ModuleSpec 解决了什么

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

## B. 代码带读版：ModuleSpec 如何接到 GPTModel

### B1. 总入口：`gpt_builder`

入口文件：

- `gpt_builders.py`
- 重点函数：`gpt_builder()`

训练脚本最终会走到：

```text
pretrain_gpt.py
  -> model_provider(...)
  -> gpt_builder(...)
  -> GPTModel(...)
```

`gpt_builder()` 先把命令行参数转成 config：

```python
config = core_transformer_config_from_args(args)
```

然后决定使用哪种 `transformer_layer_spec`：

```python
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
elif args.num_experts:
    transformer_layer_spec = get_gpt_decoder_block_spec(...)
elif args.heterogeneous_layers_config_path is not None:
    transformer_layer_spec = get_gpt_heterogeneous_layer_spec(...)
else:
    transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
```

最后把 spec 交给 `GPTModel`：

```python
model = GPTModel(
    config=config,
    transformer_layer_spec=transformer_layer_spec,
    ...
)
```

这一段是读 GPT-like 模型适配的入口。

### B2. 参数如何影响 spec 选择

几个关键参数会改变 spec 路线：

```text
--transformer-impl transformer_engine
  -> 使用 TE 版 layer spec

--num-experts
  -> 使用 MoE block spec

--spec
  -> 直接导入用户指定 spec

--heterogeneous-layers-config-path
  -> 使用 heterogeneous layer spec

--mtp-num-layers
  -> 额外构造 MTP block spec
```

也就是说，很多“模型类型差异”并不是靠新建一个 `QwenModel` 或 `LlamaModel` 类，而是通过参数选择不同 spec。

典型例子：

```text
LLaMA/Qwen:
  GPTModel + RMSNorm + SwiGLU + RoPE + GQA

Mixtral:
  GPTModel + RMSNorm + SwiGLU + RoPE + GQA + MoE spec

TE/FP8:
  GPTModel + Transformer Engine spec + FP8 config
```

### B3. `ModuleSpec` 本身是什么

入口文件：

- `megatron/core/transformer/spec_utils.py`

`ModuleSpec` 是一个 dataclass：

```python
ModuleSpec(
    module=...,
    params={...},
    submodules=...,
    metainfo={...},
)
```

四个字段的含义：

```text
module:
  要实例化的模块类，或者模块路径。

params:
  初始化这个 module 时额外传入的参数。

submodules:
  这个 module 内部继续需要哪些子模块。

metainfo:
  给构建或 checkpoint 使用的附加信息。
```

真正实例化发生在 `build_module()`：

```text
ModuleSpec
  -> 找到 module 类
  -> 把 params 和 submodules 合并进 kwargs
  -> module(...)
```

所以 `ModuleSpec` 不是模型层本身，而是“如何构建模型层”的描述。

### B4. GPT local spec 长什么样

入口文件：

- `megatron/core/models/gpt/gpt_layer_specs.py`
- 重点函数：`get_gpt_layer_local_spec()`
- 重点函数：`get_gpt_layer_local_submodules()`

local GPT layer spec 最终是：

```python
ModuleSpec(
    module=TransformerLayer,
    submodules=get_gpt_layer_local_submodules(...)
)
```

也就是说：

```text
最外层 module 是 TransformerLayer；
具体 Attention、MLP、Norm、Linear 放在 submodules 里。
```

dense local 路线可以简化成：

```text
TransformerLayerSubmodules(
  input_layernorm = local/Apex/Torch norm,
  self_attention = ModuleSpec(
    module = SelfAttention,
    submodules = SelfAttentionSubmodules(
      linear_qkv = ColumnParallelLinear,
      core_attention = DotProductAttention,
      linear_proj = RowParallelLinear,
      q_layernorm = IdentityOp 或 qk norm,
      k_layernorm = IdentityOp 或 qk norm,
    )
  ),
  mlp = MLP(
    linear_fc1 = ColumnParallelLinear,
    linear_fc2 = RowParallelLinear,
  )
)
```

重点不是每个模块细节，而是这层关系：

```text
TransformerLayer 是骨架；
submodules 决定骨架里装什么零件。
```

### B5. 为什么 `SelfAttention` 本身也用 spec

`self_attention` 不是直接写成一个已经初始化好的对象，而是：

```python
ModuleSpec(
    module=SelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(...)
)
```

这说明 spec 可以嵌套。

外层：

```text
TransformerLayer spec
```

里面包含：

```text
SelfAttention spec
MLP builder
Norm builder
BDA function
```

而 `SelfAttention spec` 内部又包含：

```text
linear_qkv
core_attention
linear_proj
q_layernorm
k_layernorm
```

这就是 Megatron 能替换后端的关键：不只整层能换，层里面的零件也能换。

### B6. BackendSpecProvider：批量换零件

相关文件：

- `megatron/core/models/backends.py`
- `megatron/core/extensions/transformer_engine_spec_provider.py`

`BackendSpecProvider` 可以理解成一个“零件供应商”接口：

```text
column_parallel_linear 用哪个类？
row_parallel_linear 用哪个类？
layer_norm 用哪个类？
core_attention 用哪个类？
grouped_mlp_modules 用哪个类？
是否支持 layernorm + linear fusion？
```

local backend 提供：

```text
ColumnParallelLinear
RowParallelLinear
DotProductAttention
本地/Apex/Torch Norm
```

TE backend 提供：

```text
TEColumnParallelLinear
TERowParallelLinear
TEDotProductAttention
TENorm
TELayerNormColumnParallelLinear
```

所以 spec 构造函数不必到处写复杂分支，只要换 backend provider，就能批量替换子模块。

### B7. TE spec 和 local spec 的差异

`get_gpt_layer_with_transformer_engine_spec()` 和 `get_gpt_layer_local_spec()` 的外层很像：

```text
ModuleSpec(module=TransformerLayer, submodules=...)
```

差别在 `submodules`。

local：

```text
SelfAttention:
  linear_qkv = ColumnParallelLinear
  core_attention = DotProductAttention
  linear_proj = RowParallelLinear

MLP:
  linear_fc1 = ColumnParallelLinear
  linear_fc2 = RowParallelLinear
```

TE：

```text
SelfAttention:
  linear_qkv = TELayerNormColumnParallelLinear 或 TEColumnParallelLinear
  core_attention = TEDotProductAttention
  linear_proj = TERowParallelLinear

MLP:
  linear_fc1 = TEColumnParallelLinear 或 fused LayerNormLinear
  linear_fc2 = TERowParallelLinear
```

结构仍然是 GPT decoder layer，但执行后端换了。

这就是第 10 节 TE/FP8 能无缝接进来的原因。

### B8. MoE 为什么要走 block spec

普通 dense GPT 每层都一样，一个 `TransformerLayer` spec 可以复用。

MoE 不一定每层都是 expert layer。Megatron 需要根据：

```text
num_moe_experts
moe_layer_freq
```

决定每一层是 dense 还是 MoE：

```text
layer 0 -> MoE
layer 1 -> dense
layer 2 -> dense
layer 3 -> MoE
...
```

所以 `args.num_experts` 分支里用的是：

```python
get_gpt_decoder_block_spec(...)
```

它会生成一组 layer specs：

```text
TransformerBlockSubmodules(
  layer_specs = [
    moe_layer_spec,
    dense_layer_spec,
    dense_layer_spec,
    moe_layer_spec,
    ...
  ]
)
```

这说明 `ModuleSpec` 不只可以描述“单层怎么装”，也可以通过 block submodules 描述“每层装法是否不同”。

### B9. 自定义 spec 怎么接入

`gpt_builder()` 里最直接的扩展入口是：

```python
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
```

命令行参数说明里也写了：

```text
--spec <module_location function_name>
```

它允许你提供一个函数或对象，返回自定义 spec。

适用场景：

```text
想替换 attention 实现
想替换 MLP 实现
想做实验性 layer
想接入特殊 backend
```

如果只是适配 LLaMA/Qwen 这种 GPT-like 模型，通常不需要写新 spec，只要调参数：

```text
RMSNorm
SwiGLU
RoPE
GQA
hidden size / heads / layers
```

如果模型结构真的和 Megatron 现有层不同，才需要写自定义 spec。

### B10. 读懂这一节后的主线图

```text
命令行参数 / yaml
  -> core_transformer_config_from_args
  -> gpt_builder
       -> 根据 args 选择 spec
          local spec
          TE spec
          MoE block spec
          heterogeneous spec
          custom --spec
  -> GPTModel(config, transformer_layer_spec)
  -> TransformerBlock(spec)
  -> build_module(spec)
  -> TransformerLayer(submodules=...)
```

再压缩一句：

```text
args 决定 config；
config 和 args 决定 spec；
spec 决定 GPTModel 里每层具体装什么。
```

### B11. 本节应该掌握什么

读完这一节，重点不是记住 GPT forward 细节，而是能回答：

1. `ModuleSpec` 解决了什么工程问题？
2. `module / params / submodules / metainfo` 分别是什么？
3. `GPTModel` 和 `TransformerLayer` 为什么不用硬编码所有后端？
4. local spec 和 TE spec 的区别在哪里？
5. MoE 为什么需要 block-level layer specs？
6. `--spec` 适合什么时候用？
7. 为什么 LLaMA/Qwen/Mixtral 大多可以复用 `GPTModel`？

### B12. 下一步带读建议

下一节适合读 `Tensor Parallel`，因为 GPT 主线里已经反复看到 `ColumnParallelLinear` 和 `RowParallelLinear` 这两个模式：

```text
QKV projection / MLP FC1: ColumnParallelLinear
attention projection / MLP FC2: RowParallelLinear
```

这样第 02 节就能接上本节的 ModuleSpec 设计主线，继续解释具体线性层为什么这样并行切分。
