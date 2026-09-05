# 10. Transformer Engine / FP8

这一节读 Transformer Engine 和 FP8/FP4。

先给结论：

> Transformer Engine 不是新的 Transformer 结构。Megatron Core 仍然负责模型结构、并行语义和训练流程；Transformer Engine 负责把 Linear、LayerNorm、Attention、Grouped GEMM 等算子换成高性能后端，并承接 FP8/FP4 低精度执行。

也就是说，读这一节不要把注意力放在“模型变了没有”。模型主线还是前面读过的：

```text
TransformerLayer
  -> SelfAttention
       -> linear_qkv
       -> core_attention
       -> linear_proj
  -> MLP / MoE
       -> linear_fc1
       -> activation
       -> linear_fc2
```

TE 改的是这些子模块的具体实现。

## A. 不看代码版：Transformer Engine / FP8 运行过程

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

## B. 代码带读版：Transformer Engine / FP8 实现路径

### B1. 这节要解决的问题

你现在已经读过：

- TP：`ColumnParallelLinear` / `RowParallelLinear` 怎么切矩阵。
- SP：TP group 内怎么切 sequence activation。
- CP：长上下文 attention 怎么跨 rank 换 K/V。
- Communication overlap：通信怎么提前发、延后等。
- MoE / EP：expert list 怎么切、token 怎么 dispatch。

TE / FP8 是把这些线再往后端压一层：

```text
MCore 定义：
  模型结构
  TP / SP / CP / EP / DP 的语义
  参数和梯度归属
  checkpoint key
  训练 schedule

TE 提供：
  LayerNorm / RMSNorm kernel
  Linear / LayerNormLinear kernel
  DotProductAttention kernel
  Grouped GEMM / fused MLP
  FP8 / FP4 autocast
  FP8 / FP4 参数存储和 all-gather
  TP communication overlap user buffer
```

一句话：

> MCore 管“这层应该做什么、并行怎么切”；TE 管“这层怎么更快、更省显存地算”。

### B2. 入口：GPT layer spec 怎么切到 TE

先看：

```text
megatron/core/models/gpt/gpt_layer_specs.py
```

最重要的是这组函数：

```python
get_gpt_layer_with_transformer_engine_submodules(...)
get_gpt_layer_with_transformer_engine_spec(...)
get_gpt_decoder_layer_specs(...)
```

`get_gpt_decoder_layer_specs()` 里有一个分叉：

```text
if use_transformer_engine:
    dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(...)
elif config.transformer_impl == "inference_optimized":
    ...
else:
    dense_layer_spec = get_gpt_layer_local_spec(...)
    moe_layer_spec = get_gpt_layer_local_spec(...)
```

所以第一个关键点是：

```text
local spec:
  ColumnParallelLinear
  RowParallelLinear
  DotProductAttention
  Apex/Torch LayerNorm

TE spec:
  TEColumnParallelLinear
  TERowParallelLinear
  TEDotProductAttention
  TENorm
  TELayerNormColumnParallelLinear
```

这些 spec 最后还是塞进同一个 `TransformerLayer`：

```python
ModuleSpec(
    module=TransformerLayer,
    submodules=get_gpt_layer_with_transformer_engine_submodules(...)
)
```

所以读法是：

> `TransformerLayer` 不用重新读一遍。只要看它收到的 submodules 从 local backend 换成了 TE backend。

### B3. BackendSpecProvider：本地实现和 TE 实现的对照表

看：

```text
megatron/core/models/backends.py
megatron/core/extensions/transformer_engine_spec_provider.py
```

`BackendSpecProvider` 是一个协议，问后端几个问题：

```text
column_parallel_linear 用哪个类？
row_parallel_linear 用哪个类？
layer_norm 用哪个类？
core_attention 用哪个类？
grouped_mlp_modules 用哪个类？
activation_func 用哪个类？
是否 fuse layernorm + linear？
```

本地实现是：

```text
LocalSpecProvider
  column_parallel_linear -> ColumnParallelLinear
  row_parallel_linear    -> RowParallelLinear
  core_attention         -> DotProductAttention
  layer_norm             -> Apex/Torch norm
  fuse_layernorm_linear  -> False
```

TE 实现是：

```text
TESpecProvider
  column_parallel_linear -> TEColumnParallelLinear
  row_parallel_linear    -> TERowParallelLinear
  core_attention         -> TEDotProductAttention
  layer_norm             -> TENorm
  fuse_layernorm_linear  -> True
  column_parallel_layer_norm_linear -> TELayerNormColumnParallelLinear
```

这个抽象很重要，因为它让 GPT layer spec 可以少写很多 if/else。

同样一段 spec 构造逻辑，只要换 provider，就能得到不同后端：

```text
LocalSpecProvider -> 纯 MCore/Apex/Torch 实现
TESpecProvider    -> Transformer Engine 实现
InferenceProvider -> inference optimized 实现
```

### B4. TENorm：LayerNorm / RMSNorm 后端替换

看：

```text
megatron/core/extensions/transformer_engine.py
class TENorm
```

`TENorm` 本身不是一个普通 module，而是一个 wrapper。它根据 config 决定实际创建：

```text
config.normalization == "LayerNorm" -> te.pytorch.LayerNorm
config.normalization == "RMSNorm"   -> te.pytorch.RMSNorm
```

并且把 Megatron 的配置传给 TE：

```text
hidden_size
eps
sequence_parallel
zero_centered_gamma
params_dtype
device
```

如果开启 fused residual RMSNorm，还会走：

```text
TEFusedResidualRMSNorm
```

这说明一个阅读重点：

> TE norm 不是只替换一个函数，它还要知道 sequence parallel 和参数初始化设备。

因为 LayerNorm/RMSNorm 的参数和输出都要和 TP/SP 的张量布局匹配。

### B5. TELinear：TE Linear 的总 wrapper

看：

```text
megatron/core/extensions/transformer_engine.py
class TELinear
class TEColumnParallelLinear
class TERowParallelLinear
class TELayerNormColumnParallelLinear
```

`TELinear` 继承自：

```python
te.pytorch.Linear
```

它做了几类适配。

第一类：把 Megatron 的并行语义翻译给 TE。

```text
parallel_mode="column"     -> output dim 切分
parallel_mode="row"        -> input dim 切分
parallel_mode="duplicated" -> 不做 TP 切分
tp_group                   -> TP process group
tp_size                    -> TP world size
sequence_parallel          -> 是否启用 SP
```

这和第 3 节 TP 读过的矩阵切分是一致的：

```text
TEColumnParallelLinear: 类似 ColumnParallelLinear
TERowParallelLinear:    类似 RowParallelLinear
```

第二类：处理 bias 返回约定。

MCore 的 Linear 通常返回：

```python
output, bias
```

TE 有时只返回 output。`TELinear.forward()` 会把它统一成：

```python
return out, None
```

或者如果 `skip_bias_add=True`：

```python
return out, bias
```

这样上层的 `TransformerLayer` 不需要关心 local / TE 差异。

第三类：处理参数和梯度归属。

普通参数：

```text
param.allreduce = True
```

expert 参数：

```text
param.allreduce = not expert_parallel
```

这和第 09 节 MoE 的结论对上了：

> expert 参数不是普通 DP 参数，它的同步 group 和普通 dense 参数不同。

第四类：处理 TP communication overlap。

如果：

```text
config.tp_comm_overlap = True
parallel_mode != "duplicated"
```

TE Linear 会收到 user buffer 相关参数：

```text
ub_overlap_ag
ub_overlap_rs
ub_name
```

`ub_name` 只能是：

```text
qkv
proj
fc1
fc2
```

这对应第 08 节里的 TP Linear overlap：

```text
qkv:  attention 的 QKV projection
proj: attention output projection
fc1:  MLP 第一层
fc2:  MLP 第二层
```

这里可以形成一个连接：

> DP / PP overlap 多在 Megatron 调度层控制；TP Linear overlap 主要由 TE Linear 的 user buffer 机制执行。

### B6. TEColumnParallelLinear / TERowParallelLinear 和本地 TP 的差异

`TEColumnParallelLinear` 限制：

```text
gather_output 不能是 True
```

也就是 TE 这个 wrapper 期望输出继续保持 TP 切分，后续通信由 TE / MCore 的布局规则处理。

`TERowParallelLinear` 限制：

```text
input_is_parallel 必须是 True
```

也就是 row parallel linear 进来时，输入已经按 TP 切好了。

这和本地实现相比更严格，但逻辑还是同一套：

```text
ColumnParallelLinear:
  weight 按 output dim 切
  每个 rank 产生一部分输出

RowParallelLinear:
  weight 按 input dim 切
  每个 rank 消费一部分输入
  输出需要 reduce
```

你可以把 TE 的 TP Linear 理解为：

> 语义仍然是 MCore TP；具体 GEMM、通信融合、FP8 执行交给 TE。

### B7. TELayerNormColumnParallelLinear：为什么 TE 会 fuse LN + Linear

TE backend 的 provider 里：

```python
fuse_layernorm_and_linear() -> True
column_parallel_layer_norm_linear() -> TELayerNormColumnParallelLinear
```

这会让一些结构从：

```text
LayerNorm
ColumnParallelLinear
```

变成：

```text
LayerNormColumnParallelLinear
```

典型位置是 attention 的 QKV projection：

```text
self_attention.linear_qkv
```

以及 MLP 的第一层：

```text
mlp.linear_fc1
```

注意，这不是改变数学表达式，而是把相邻算子交给一个 fused module 做。

阅读时要分清：

```text
结构层面：
  还是 norm 后接 linear

实现层面：
  可能由一个 TE fused module 执行
```

这也是 checkpoint 里需要 `sharded_state_dict_keys_map` 的原因之一：fused 后模块名可能和 local spec 不一样，但 checkpoint key 需要能映射回来。

### B8. TEDotProductAttention：Attention 后端替换

看：

```text
megatron/core/extensions/transformer_engine.py
class TEDotProductAttention
```

它继承：

```python
te.pytorch.DotProductAttention
```

它接收的核心信息包括：

```text
num_attention_heads
kv_channels
attention_dropout
attn_mask_type
sequence_parallel
tp_size
tp_group
layer_number
```

如果启用 Context Parallel，还会传：

```text
cp_group
cp_global_ranks
cp_stream
cp_comm_type
```

这和第 9 节 CP 对上：

> CP 的 attention 通信主要发生在 attention 后端内部。Megatron 负责把 CP group、通信类型和 packed sequence metadata 交给 TE。

`TEDotProductAttention.forward()` 里还会处理：

```text
packed_seq_params
dynamic cp_group
qkv_format
attention_bias
mask type 转换
sliding window attention
num_splits
```

这部分第一次读不用死磕。重点是抓住这条线：

```text
SelfAttention 仍然负责生成 Q/K/V
TEDotProductAttention 负责高性能 attention kernel
CP 相关通信参数也交给它
```

### B9. FP8 不是“整个模型参数都变成 uint8”

这是最容易误解的地方。

`TransformerConfig` 里有：

```text
fp8
fp8_recipe
fp8_param
fp8_wgrad
fp8_dot_product_attention
fp8_multi_head_attention
tp_only_amax_red
first_last_layers_bf16
```

其中：

```text
fp8:
  是否启用 FP8 执行
  可选 e4m3 / hybrid

fp8_recipe:
  scaling / quantization 策略
  delayed / tensorwise / mxfp8 / blockwise / custom

fp8_param:
  是否把部分参数也保持为 FP8，用于节省参数内存和 FP8 all-gather
```

配置说明里明确写了：

> `fp8_param` 不是所有参数都会转成 FP8。bias 等参数不会变，主要影响 GEMM 权重，具体哪些参数由 TE 决定。

所以不要理解成：

```text
模型参数全是 FP8
```

更准确是：

```text
部分 TE 管理的 GEMM 权重可以 FP8 存储/通信；
forward/backward 的部分算子可以在 FP8 autocast context 中执行；
scale / amax 等量化元数据由 TE 维护。
```

### B10. FP8 context：在哪里打开低精度执行

看：

```text
megatron/core/fp8_utils.py
get_fp8_recipe(...)
get_fp8_context(...)
```

`get_fp8_recipe(config)` 会根据：

```text
config.fp8
config.fp8_recipe
config.fp8_dot_product_attention
config.fp8_multi_head_attention
config.fp8_wgrad
```

创建 TE recipe。

主要 recipe：

```text
delayed   -> TEDelayedScaling
tensorwise -> Float8CurrentScaling
blockwise  -> Float8BlockScaling
mxfp8      -> MXFP8BlockScaling
custom     -> CustomRecipe
```

`get_fp8_context(config, layer_no, is_init=False)` 决定是否进入：

```python
transformer_engine.pytorch.fp8_autocast(...)
```

如果是初始化参数，并且 `is_init=True`，则进入：

```python
transformer_engine.pytorch.fp8_model_init(...)
```

它还会处理 first/last layers 保持 BF16：

```text
first_last_layers_bf16
num_layers_at_start_in_bf16
num_layers_at_end_in_bf16
```

所以 FP8 有两条线：

```text
执行线：
  fp8_autocast 控制 forward/backward 算子怎么量化执行

参数线：
  fp8_model_init / fp8_param / fp8_param_gather 控制部分参数是否用 FP8 存储和通信
```

这两条线相关，但不是同一个开关。

### B11. amax / scale：为什么 FP8 需要额外状态

FP8 的数值范围比 BF16/FP16 小很多，所以必须有 scale。

大致逻辑是：

```text
高精度 tensor
  -> 根据 amax 计算 scale
  -> 量化成 FP8
  -> GEMM / attention 执行
  -> 必要时反量化或继续低精度流转
```

`fp8_utils.py` 里会处理：

```text
amax history
scale
scale_inv
QuantizedTensor / Float8Tensor / MXFP8Tensor
```

并且在并行训练里，amax 还涉及 group 内同步：

```python
get_amax_reduction_group(
    with_context_parallel=True,
    tp_only_amax_red=config.tp_only_amax_red
)
```

这里和 CP 有一个连接：

> 开 CP 时，FP8 的 amax reduction group 也要考虑 CP，否则不同 context shard 看到的数值范围可能不一致。

第一遍只需要记：

```text
FP8 tensor = 数据 + scale/amax 元数据
```

不要把它当普通 `torch.uint8` tensor。

### B12. fp8_param_gather：参数通信也可以 FP8

看：

```text
megatron/core/fp8_utils.py
modify_underlying_storage(...)
quantize_param_shard(...)
post_all_gather_processing(...)
correct_amax_history_if_needed(...)
```

这组函数是给：

```text
--fp8-param-gather
```

服务的。

普通 distributed optimizer 的参数流是：

```text
每个 DP rank 持有参数 shard
forward 前 all-gather 出完整参数
```

如果打开 `fp8_param_gather`：

```text
主参数仍可能是 FP32 optimizer state
模型计算参数可以是 FP8 quantized tensor
参数 all-gather 可以用 FP8 形式通信
```

`quantize_param_shard()` 的作用就是：

```text
把 fp32 main param shard cast/quantize 到 fp8 model param shard
并更新 scale / amax / transpose cache 等 TE 需要的状态
```

注意训练参数检查里有约束：

```text
--fp8-param-gather 只支持 distributed optimizer / FSDP / inference mode 等路径
```

这和第 05 节 Distributed Optimizer 对上：

> 普通 DP all-reduce 不是 fp8_param_gather 的主舞台；它主要和参数 shard、all-gather、distributed optimizer 配合。

### B13. FP4：同一条后端思路，但约束更强

配置里有：

```text
fp4
fp4_recipe
fp4_param
fp4_quantizer_factory
```

目前主线是：

```text
fp4_format: e2m1
fp4_recipe: nvfp4
```

代码里要求：

```text
FP4 和 FP8 不能同时启用
FP4 param gather 必须和 FP4 mode 一起用
NVFP4 需要较新的 Transformer Engine
```

第一遍不建议展开 FP4 的底层 packed storage。只要知道：

```text
FP4 是比 FP8 更激进的 TE 量化路径；
它也通过 TE recipe、autocast、param gather 接入；
硬件和 TE 版本约束更强。
```

### B14. MoE 和 TE / FP8 的关系

第 09 节读过 MoE：

```text
router -> token dispatcher -> experts -> combine
```

这里补上 TE 相关点。

`TESpecProvider.grouped_mlp_modules()` 会选择：

```text
TEGroupedMLP
TEColumnParallelGroupedLinear
TERowParallelGroupedLinear
```

如果 TE 版本或 grouped GEMM 不满足，就可能 fallback 到：

```text
SequentialMLP
TEColumnParallelLinear
TERowParallelLinear
```

MoE + FP8/FP4 还有 token padding 问题。

`moe_utils.py` 里有：

```text
get_fp8_align_size(config.fp8_recipe)
get_fp4_align_size(config.fp4_recipe)
```

FP8 GEMM 通常要求 token 数对齐：

```text
delayed / tensorwise / blockwise: 16
mxfp8: 32
```

所以 MoE token dispatcher / grouped GEMM 有时要 pad token，避免 GEMM shape 不满足 FP8/FP4 kernel 要求。

这就是为什么 MoE + FP8 的复杂度很高：

```text
router 造成每个 expert 的 token 数动态变化
FP8/FP4 GEMM 又要求 shape 对齐
dispatcher 必须同时处理 token 排列、通信和 padding
```

第一遍记住这个关系就够了。

### B15. TE 和之前几节的连接

#### 和 TP 的连接

```text
ColumnParallelLinear / RowParallelLinear
  -> TEColumnParallelLinear / TERowParallelLinear
```

矩阵怎么切的语义不变。

TE 增加：

```text
更快 GEMM
FP8/FP4 执行
TP user buffer overlap
fused layernorm + linear
```

#### 和 SP 的连接

TE Linear / Norm 会接收：

```text
sequence_parallel=config.sequence_parallel
```

SP 下的 all-gather / reduce-scatter 仍然要和 Linear 的 input/output layout 对齐。

第 08 节里也看到：

```text
tp_comm_overlap 通常要求 sequence_parallel
```

#### 和 CP 的连接

`TEDotProductAttention` 接收：

```text
cp_group
cp_global_ranks
cp_stream
cp_comm_type
```

CP 的 K/V 交换由 attention 后端实现。

#### 和 Distributed Optimizer 的连接

`fp8_param_gather` 和参数 shard / all-gather 有关：

```text
main param shard -> quantize_param_shard -> FP8 model param shard
forward all-gather -> post_all_gather_processing
```

这和 distributed optimizer 的参数 gather 生命周期对齐。

#### 和 MoE 的连接

MoE expert 可以用：

```text
TEGroupedMLP
TE grouped GEMM
FP8/FP4 token padding
```

但是 expert 参数、token dispatcher、EP group 的语义仍然由 MCore 管。

### B16. 推荐阅读顺序

第一遍按这个顺序读：

1. `gpt_layer_specs.py`

看：

```text
get_gpt_layer_with_transformer_engine_submodules
get_gpt_decoder_layer_specs
```

目标是看懂：

```text
use_transformer_engine=True 时，layer spec 怎么变。
```

2. `transformer_engine_spec_provider.py`

看：

```text
TESpecProvider
```

目标是把 local backend 和 TE backend 对照起来。

3. `transformer_engine.py`

先只看这些类：

```text
TENorm
TELinear
TEColumnParallelLinear
TERowParallelLinear
TELayerNormColumnParallelLinear
TEDotProductAttention
TEDelayedScaling
```

暂时跳过：

```text
TE op fuser 的细节
CUDA graph 相关
quantization config per-module override
checkpoint conversion 细节
```

4. `fp8_utils.py`

看：

```text
get_fp8_recipe
get_fp8_context
quantize_param_shard
post_all_gather_processing
```

目标是分清：

```text
FP8 执行 context
FP8 参数存储 / all-gather
```

5. `transformer_config.py`

只扫 FP8/FP4 字段和 validation。

目标是知道哪些开关互相约束。

### B17. 第一次可以跳过的内容

这些先不用深挖：

- TE 内部 CUDA kernel。
- `QuantizedTensor` 的 raw storage 细节。
- TE 1.x / 2.x 兼容分支。
- per-module quantization config。
- FP4 packed rowwise storage。
- CUDA graph + FP8 transpose cache。
- Kitchen backend。
- inference optimized + MXFP8 的专用路径。
- qk-clip / softmax_type / sliding window attention 的 TE 版本差异。

这些不是不重要，而是会打断主线。

第一遍目标是建立这张图：

```text
GPT layer spec
  -> BackendSpecProvider
      -> Local modules
      -> TE modules
  -> TransformerLayer
      -> TE Linear / Norm / Attention
      -> FP8 autocast context
      -> FP8 param gather
```

### B18. 最小心智模型

最后用三句话收束：

1. TE 是后端替换层，不是模型结构替换。

```text
MCore 定结构和并行，TE 执行高性能 kernel。
```

2. FP8 是执行和参数通信的低精度机制，不等于整个模型参数全变成 FP8。

```text
fp8_autocast 管算子执行；
fp8_param / fp8_param_gather 管部分 GEMM 权重的存储和通信。
```

3. TE 会把前面学过的 TP/SP/CP/MoE/overlap 都接起来。

```text
TP -> TE parallel linear
SP -> sequence_parallel flag
CP -> TEDotProductAttention cp_group
MoE -> TEGroupedMLP / grouped GEMM
overlap -> TE user buffer
```

记忆句：

> MCore 画计算图和并行图，TE 负责把图上的关键算子跑快，并在需要时用 FP8/FP4 跑。

下一节如果继续扩展，建议读 `Distributed Checkpointing`。原因是前面已经读了 optimizer、distributed optimizer、MoE 和 TE，checkpoint 会把“参数如何切、如何命名、如何保存/恢复”统一起来。

