# MCoreAdapter Qwen3.5 加载与训练流程

本文只围绕一个问题展开：当 ROLL 使用 `mcore_adapter` 加载一个 Hugging Face 格式的 Qwen3.5 模型时，代码按什么顺序执行。

## 1. ROLL 决定走 MCoreAdapter

入口通常在：

`roll/models/model_providers.py`

核心判断：

```python
if (
    mca_TrainingArguments is not None
    and training_args is not None
    and isinstance(training_args, mca_TrainingArguments)
):
    model = AutoModel.from_pretrained(model_args.model_name_or_path, training_args)
```

这里的 `AutoModel` 不是 Hugging Face 的 `AutoModel`，而是：

```python
from mcore_adapter.models import AutoModel
```

所以，ROLL 是否进入 Megatron/MCoreAdapter 路径，关键不只是模型是 Qwen3.5，而是当前 strategy 构造出的 `training_args` 是否是 `mcore_adapter.TrainingArguments`。

## 2. AutoModel 识别模型类型

入口：

`mcore_adapter/src/mcore_adapter/models/auto/modeling_auto.py`

调用：

```python
AutoModel.from_pretrained(model_name_or_path, training_args)
```

它会先判断模型目录里有没有 MCoreAdapter 自己的 config。

如果有 MCA config，就从 MCA config 里读取：

```python
hf_model_type
```

如果没有 MCA config，就读取 Hugging Face 的：

```text
config.json
```

然后通过 Hugging Face `AutoConfig` 获取：

```python
hf_config.model_type
```

对 Qwen3.5 来说，期望得到：

```python
model_type = "qwen3_5"
```

接着查模型注册表：

```python
MODEL_MAPPING["qwen3_5"] -> Qwen3_5Model
```

注册位置：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py`

```python
@register_model("qwen3_5")
class Qwen3_5Model(...)
```

## 3. 找到 Qwen3.5 的三类注册信息

Qwen3.5 的注册集中在：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/__init__.py`

主要有三类：

```python
@register_config("qwen3_5")
```

```python
@register_model("qwen3_5")
```

```python
register_template("qwen3_5", ...)
```

含义分别是：

```text
qwen3_5 -> Qwen3_5Config
qwen3_5 -> Qwen3_5Model
qwen3_5 -> Qwen3_5Template
```

也就是说，识别出 `model_type = qwen3_5` 后，MCoreAdapter 知道：

```text
用哪个 config 类
用哪个 model 类
用哪套 HF/MCA 权重转换模板
```

## 4. HF Config 转成 MCA/Megatron Config

入口：

`mcore_adapter/src/mcore_adapter/models/model_factory.py`

关键调用：

```python
config = cls.config_class.from_pretrained(model_name_or_path, args)
```

此时 `cls` 是 `Qwen3_5Model`，所以：

```python
cls.config_class = Qwen3_5Config
```

最终进入：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/config_qwen3_5.py`

这个阶段不是简单复制 HF config，而是把多个来源合并成一个 Megatron-Core 可用的 `Qwen3_5Config`。

## 5. 字段名映射

Qwen3.5 的 config 映射规则在：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/__init__.py`

示例映射：

```text
HF:  hidden_size
MCA: hidden_size

HF:  num_hidden_layers
MCA: num_layers

HF:  num_attention_heads
MCA: num_attention_heads

HF:  num_key_value_heads
MCA: num_query_groups

HF:  intermediate_size
MCA: ffn_hidden_size

HF:  vocab_size
MCA: padded_vocab_size

HF:  max_position_embeddings
MCA: max_sequence_length

HF:  rms_norm_eps
MCA: layernorm_epsilon
```

Qwen3.5 的文本模型字段通常在：

```text
text_config.xxx
```

所以 `Qwen3_5Template.adjust_config_hf_to_mca()` 会给普通文本字段自动补上：

```text
text_config.
```

例如实际读取的是：

```text
text_config.hidden_size
text_config.num_hidden_layers
text_config.num_attention_heads
```

## 6. 注入 Qwen3.5 的 Megatron 常量

Qwen3.5 template 会额外注入一些 MCA/Megatron 侧需要的常量配置。

典型包括：

```python
"swiglu": True
"position_embedding_type": "mrope"
"normalization": "RMSNorm"
"add_bias_linear": False
"hidden_dropout": 0.0
"qk_layernorm": True
"layernorm_zero_centered_gamma": True
"attention_output_gate": True
"experimental_attention_variant": "gated_delta_net"
```

这些字段不一定直接来自 HF config，但构造 Megatron-Core 模型时必须明确。

## 7. 合并训练并行参数

`training_args` 里的并行参数会覆盖或补充 config。

常见字段：

```text
tensor_model_parallel_size
pipeline_model_parallel_size
context_parallel_size
expert_model_parallel_size
sequence_parallel
virtual_pipeline_model_parallel_size
```

所以最终的 `Qwen3_5Config` 实际由四部分组成：

```text
HF 模型结构参数
+ Qwen3.5 template 常量
+ ROLL/Megatron 并行训练参数
+ checkpoint 兼容参数
```

## 8. Qwen3_5Config 后处理

位置：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/config_qwen3_5.py`

`Qwen3_5Config.__post_init__()` 会处理 Qwen3.5 特有字段：

```python
vision_config
rope_scaling
mrope_section
rotary_base
rotary_percent
pixel_values_dim
merge_size
layer_types
```

例如：

```python
self.pixel_values_dim = (
    vision_config_obj.patch_size
    * vision_config_obj.patch_size
    * vision_config_obj.in_channels
    * vision_config_obj.temporal_patch_size
)
```

这会影响后面视觉输入 `pixel_values` 如何送入 vision encoder。

如果 `layer_types` 没有显式提供，会自动生成：

```python
self.layer_types = [
    "linear_attention" if bool((i + 1) % self.linear_attention_freq) else "full_attention"
    for i in range(self.num_layers)
]
```

也就是说，Qwen3.5 可能不是每层都用普通 attention，而是混合：

```text
linear_attention
linear_attention
full_attention
linear_attention
...
```

## 9. 构造 VirtualModels

位置：

`mcore_adapter/src/mcore_adapter/models/model_factory.py`

关键代码：

```python
models = VirtualModels(cls, config=config)
```

此时：

```python
cls = Qwen3_5Model
```

`VirtualModels` 用来支持 virtual pipeline parallel。它可能只构造一个模型，也可能构造多个 model chunk：

```python
for i in range(config.virtual_pipeline_model_parallel_size or 1):
    self.models.append(cls(config, ...))
```

所以逻辑上的一个 Qwen3.5 模型，在 PP/VP 场景下可能被拆成多个 chunk。

## 10. 构造 Megatron-Core GPTModel 主体

模型初始化调用链：

```text
Qwen3_5Model.__init__
    ↓
Qwen3_5McaGPTModel.__init__
    ↓
McaGPTModel.__init__
    ↓
Megatron-Core GPTModel.__init__
```

通用主体在：

`mcore_adapter/src/mcore_adapter/models/model_factory.py`

`McaGPTModel.__init__()` 会构造 Megatron-Core GPTModel：

```python
super().__init__(
    config=config,
    transformer_layer_spec=transformer_layer_spec,
    vocab_size=config.padded_vocab_size,
    max_sequence_length=config.max_sequence_length,
    pre_process=self.pre_process,
    post_process=self.post_process,
    parallel_output=True,
    share_embeddings_and_output_weights=config.tie_embeddings_and_output_weights,
    position_embedding_type=config.position_embedding_type,
    ...
)
```

这里最关键的是：

```python
transformer_layer_spec = self._get_transformer_layer_spec(config)
```

它决定 Megatron-Core 每层 transformer 的具体结构。

## 11. Qwen3.5 自定义 Transformer Layer Spec

位置：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py`

Qwen3.5 重写了：

```python
def _get_transformer_layer_spec(...)
```

并且要求：

```python
assert config.transformer_impl == "transformer_engine"
```

也就是说当前 Qwen3.5 实现只支持 `transformer_engine`。

当 `experimental_attention_variant` 不为空时，会走：

```python
get_transformer_block_with_experimental_attention_variant_spec(...)
```

这使 Qwen3.5 能支持：

```text
full_attention 层
linear_attention / gated_delta_net 层
qk_layernorm
RMSNorm
SwiGLU MLP
attention output gate
```

## 12. 添加 Qwen3.5 特殊模块

`Qwen3_5McaGPTModel.__init__()` 会重建多模态 RoPE：

```python
self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(...)
self.mrope_section = self.config.mrope_section
```

`Qwen3_5Model.__init__()` 还会在 pipeline first stage 上构造视觉模型：

```python
if self.pre_process:
    self.vision_model = Qwen3_5VisionModel._from_config(...)
```

所以模型骨架最终大致是：

```text
embedding
vision_model        只在 pre_process stage
decoder layers      按 TP/PP/VP 切分
final_norm          只在 post_process stage
output_layer        只在 post_process stage
```

## 13. 判断加载 MCA 权重还是 HF 权重

模型骨架构造完成后，参数还只是初始化状态。接下来判断权重来源。

位置：

`mcore_adapter/src/mcore_adapter/models/model_factory.py`

关键逻辑：

```python
mca_ckpt_exist = exists_mca_config(model_name_or_path)

if mca_ckpt_exist:
    old_mca_config = cls.config_class.from_pretrained(model_name_or_path)
    dist_config_match = config.distribute_config_match(old_mca_config)
```

分两种情况。

情况 A：目录里有 MCA checkpoint，并且并行配置匹配：

```python
state_dict = load_state_dict_from_checkpoint(model_name_or_path)
```

这时直接加载 MCA/Megatron 权重。

情况 B：没有 MCA checkpoint，或者并行配置不匹配：

```python
converter = ModelConverter(config)
state_dict[key] = converter.load_mca_state_dict_from_hf(model_name_or_path, vp_stage=i)
```

这时会从 HF 权重即时转换成当前 rank 需要的 MCA/Megatron 权重。

## 14. HF 权重转换成 MCA 权重

转换规则来自：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/__init__.py`

核心是 `Qwen3_5Template`。

第一类转换是改名。

```text
HF:  model.language_model.embed_tokens.weight
MCA: embedding.word_embeddings.weight

HF:  model.language_model.norm.weight
MCA: decoder.final_layernorm.weight

HF:  lm_head.weight
MCA: output_layer.weight
```

第二类转换是合并多个 HF 权重。

MLP 的 gate/up 在 HF 是两个矩阵：

```text
.mlp.gate_proj.weight
.mlp.up_proj.weight
```

Megatron 里合成一个：

```text
.mlp.linear_fc1.weight
```

对应：

```python
StackConverOp(..., dim=0)
```

QKV 也类似：

```text
.self_attn.q_proj.weight
.self_attn.k_proj.weight
.self_attn.v_proj.weight
    ↓
.self_attention.linear_qkv.weight
```

对应：

```python
GatedQKVConverOp(...)
```

第三类转换是 Qwen3.5 的 linear attention。

HF 侧：

```text
.linear_attn.in_proj_qkv.weight
.linear_attn.in_proj_z.weight
.linear_attn.in_proj_b.weight
.linear_attn.in_proj_a.weight
```

MCA 侧：

```text
.self_attention.in_proj.weight
```

这一步由：

```python
Qwen3_5_GDNConverOp
```

处理。

它会拆分、reshape、再组合 `q/k/v/z/b/a` 权重。

第四类转换是 zero-centered RMSNorm。

HF 到 MCA：

```python
weight - 1
```

MCA 到 HF：

```python
weight + 1
```

对应：

```python
ZeroCenteredRMSNormConverOp
```

这是因为 Megatron 的 zero-centered gamma 和 HF 普通 RMSNorm 权重表示方式不同。

## 15. 加载 state_dict

转换完成后回到：

`mcore_adapter/src/mcore_adapter/models/model_factory.py`

执行：

```python
missing_keys, unexpected_keys = models.load_state_dict(state_dict, strict=False)
```

然后检查：

```python
assert unexpected_keys is None or len(unexpected_keys) == 0
assert missing_keys is None or len(missing_keys) == 0
```

这一步保证以下几件事是一致的：

```text
HF config
Qwen3.5 模型结构
Qwen3.5 权重转换模板
当前并行配置
checkpoint 权重内容
```

如果这里报 missing/unexpected，常见原因包括：

```text
model_type 不匹配
transformers 版本里的 Qwen3.5 结构变了
template 权重映射不完整
并行配置和 checkpoint 不兼容
vision/text config 没对齐
```

## 16. ROLL 设置 train/eval、LoRA 和 patch

加载完成后回到：

`roll/models/model_providers.py`

如果是训练：

```python
model.train()
for param in model.parameters():
    param.requires_grad = True
```

如果是推理：

```python
model.eval()
for param in model.parameters():
    param.requires_grad = False
```

如果配置了 LoRA，会继续走：

```python
apply_megatron_lora()
set_linear_is_expert(model[0])
setup_lora_training(...)
```

最后：

```python
patch_model(model, config, use_mcore=True)
```

这一步给 ROLL 的 RL 训练逻辑做模型兼容处理。

## 17. 交给 MegatronStrategy

推理策略入口：

`roll/distributed/strategy/megatron_strategy.py`

训练策略入口也在同一个文件中。

初始化时大致做：

```python
self.megatron_train_args = TrainingArguments(**config_dict)
self.model = model_provider(...)
self.models_unwrapped = self.model.get_models()
self.forward_backward_func = get_forward_backward_func()
```

此时模型已经是：

```text
VirtualModels[Qwen3_5Model]
```

ROLL 后面不是直接用 HF forward 训练，而是通过 Megatron 的：

```python
get_forward_backward_func()
```

处理：

```text
micro batch
gradient accumulation
pipeline parallel
tensor parallel
context parallel
distributed optimizer
```

## 18. Qwen3.5 Forward 流程

位置：

`mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py`

输入可能包括：

```python
input_ids
attention_mask
position_ids
labels
pixel_values
pixel_values_videos
image_grid_thw
video_grid_thw
```

执行顺序：

```text
1. 如果没有 position_ids，调用 get_rope_index() 生成多模态位置编码。
2. 如果开启 context parallel，对 input_ids / attention_mask 做当前 CP rank 切片。
3. 如果当前不是 pipeline first stage，就接收上一个 stage 的 decoder_input。
4. 如果是 pipeline first stage，先做 token embedding。
5. 如果有图片，调用 vision_model(pixel_values, image_grid_thw)。
6. 把图片 embedding 填回 image_token_id 对应的位置。
7. 如果有视频，调用 vision_model(pixel_values_videos, video_grid_thw)。
8. 把视频 embedding 填回 video_token_id 对应的位置。
9. 得到 decoder_input = inputs_embeds。
10. 调用 Megatron-Core GPTModel.forward。
11. 如果有 labels，返回 loss 相关 tensor；否则返回 logits 或模型输出。
```

因此 Qwen3.5 的 forward 不是简单的：

```text
input_ids -> embedding -> transformer
```

而是：

```text
input_ids
+ image/video pixels
+ multimodal position ids
+ CP/TP/PP 切分
+ vision encoder embedding injection
+ Megatron decoder
```

## 19. 训练过程中的执行方式

进入 MegatronStrategy 后，训练通常围绕 Megatron 的 pipeline forward/backward 执行。

大致流程：

```text
batch
    ↓
按 micro batch / dynamic batch / sequence packing 切分
    ↓
根据 TP/CP/PP 分发到不同 rank
    ↓
Qwen3_5Model.forward
    ↓
Megatron forward_backward_func
    ↓
loss / grad
    ↓
Megatron optimizer step
    ↓
日志、评估、checkpoint
```

ROLL 的 RL 训练中，还会在 actor、reference、critic、reward 等角色中复用模型 provider 和 strategy，只是 train/eval、requires_grad、loss 计算方式不同。

## 20. 保存 MCA Checkpoint

如果保存 MCoreAdapter/Megatron 格式：

```python
model.save_pretrained(output_dir)
```

会保存：

```text
mca config
当前并行格式下的 model state_dict
tokenizer
training_args
optimizer/scheduler 状态，取决于保存路径和策略
```

训练恢复时，如果 MCA config 存在且并行配置匹配，会优先从 MCA checkpoint 直接加载，跳过 HF 权重转换。

## 21. 可选导出 HF Checkpoint

如果需要导出 Hugging Face 格式：

```python
model.save_pretrained_as_hf(output_dir)
```

会走反向 converter：

```text
MCA 权重
    ↓
按 TP/PP/VP/EP 收集或合并
    ↓
MCA name 转 HF name
    ↓
拆分 linear_qkv / linear_fc1 / GDN 权重
    ↓
zero-centered RMSNorm 做 weight + 1
    ↓
写出 HF config / safetensors
```

这一步和从 HF 加载时的转换方向相反。

## 22. 总流程速记

```text
ROLL strategy config
    ↓
构造 mcore_adapter.TrainingArguments
    ↓
default_actor_model_provider 判断走 MCoreAdapter
    ↓
mcore_adapter AutoModel.from_pretrained
    ↓
读取 MCA config 或 HF config
    ↓
识别 model_type = qwen3_5
    ↓
找到 Qwen3_5Config / Qwen3_5Model / Qwen3_5Template
    ↓
HF config 字段映射到 MCA config
    ↓
合并并行参数 TP/PP/CP/EP/VP
    ↓
Qwen3_5Config.__post_init__ 处理 vision_config、mrope、layer_types
    ↓
VirtualModels 构造一个或多个 Qwen3_5Model chunk
    ↓
McaGPTModel 构造 Megatron-Core GPTModel 主体
    ↓
Qwen3_5Model 添加 vision_model 和 multimodal rope
    ↓
判断加载 MCA checkpoint 还是 HF checkpoint
    ↓
如果是 HF checkpoint，ModelConverter 使用 Qwen3_5Template 转权重
    ↓
按当前并行 rank 得到本 rank state_dict
    ↓
load_state_dict 注入模型
    ↓
ROLL 设置 train/eval、requires_grad、LoRA、patch_model
    ↓
MegatronStrategy 包装 DDP/optimizer/forward_backward_func
    ↓
训练/推理时走 Qwen3_5Model.forward
    ↓
保存 MCA checkpoint
    ↓
可选反向转换导出 HF checkpoint
```

