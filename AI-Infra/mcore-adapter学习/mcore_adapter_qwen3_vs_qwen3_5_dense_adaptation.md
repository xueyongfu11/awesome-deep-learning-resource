# MCoreAdapter 添加 Qwen3 与 Qwen3.5 Dense 模型的差异

本文基于 `qwen3_vs_qwen3_5_dense_architecture.md` 的架构差异，分析在 `mcore_adapter` 中添加 Qwen3 dense 和 Qwen3.5 dense 时，代码适配工作的不同。

## 结论

在 `mcore_adapter` 里，Qwen3 dense 基本是标准 decoder-only Transformer LLM 适配：注册通用 config、通用模型类和一套标准 HF/Megatron 权重转换模板即可。

Qwen3.5 dense 则不是 Qwen3 dense 的小改版。它需要单独的 config、model、template、分布式 checkpoint 规则、Gated DeltaNet 线性注意力转换、多模态 RoPE、vision tower 和多模态 forward 逻辑。

可以简化为：

```text
Qwen3 dense:
  标准 LLM 模板适配

Qwen3.5 dense:
  Qwen3-Next 混合注意力 + Qwen3-VL 多模态路径 + 特殊权重转换
```

## 1. 注册层面的差异

Qwen3 的注册很薄，集中在：

```text
mcore_adapter/src/mcore_adapter/models/qwen3/__init__.py
```

核心是直接复用通用类：

```python
register_config("qwen3", McaModelConfig)
register_model("qwen3", McaGPTModel)
register_dist_config("qwen3", default_dist_config)
```

也就是说，Qwen3 dense 在 MCoreAdapter 里被视为普通 Megatron GPTModel。

Qwen3.5 dense 则拆成专用实现：

```text
mcore_adapter/src/mcore_adapter/models/qwen3_5/config_qwen3_5.py
mcore_adapter/src/mcore_adapter/models/qwen3_5/modeling_qwen3_5.py
mcore_adapter/src/mcore_adapter/models/qwen3_5/__init__.py
```

它注册的是：

```text
qwen3_5 -> Qwen3_5Config
qwen3_5 -> Qwen3_5Model
qwen3_5 -> Qwen3_5Template
```

这说明 Qwen3.5 dense 需要独立描述模型结构、forward 行为和权重转换规则。

## 2. Config 映射差异

Qwen3 的 config 映射是标准文本 LLM 字段：

```text
max_position_embeddings -> max_sequence_length
hidden_size             -> hidden_size
num_attention_heads     -> num_attention_heads
num_key_value_heads     -> num_query_groups
num_hidden_layers       -> num_layers
rms_norm_eps            -> layernorm_epsilon
vocab_size              -> padded_vocab_size
rope_theta              -> rotary_base
intermediate_size       -> ffn_hidden_size
```

并注入标准 LLM 常量：

```python
"swiglu": True
"position_embedding_type": "rope"
"normalization": "RMSNorm"
"add_bias_linear": False
"hidden_dropout": 0.0
"rotary_percent": 1.0
"qk_layernorm": True
```

Qwen3.5 的 config 要复杂得多。HF config 中很多语言模型字段在 `text_config.xxx` 下，所以 `Qwen3_5Template.adjust_config_hf_to_mca()` 会自动给普通文本字段补上 `text_config.`。

但下面这些字段不属于普通文本字段，需要从顶层读取：

```text
vision_config
vision_start_token_id
vision_end_token_id
vision_token_id
image_token_id
video_token_id
tie_word_embeddings
```

Qwen3.5 还需要新增 hybrid attention 和 mRoPE 相关字段：

```text
rope_parameters          -> rope_scaling
layer_types              -> layer_types
full_attention_interval  -> linear_attention_freq
linear_conv_kernel_dim   -> linear_conv_kernel_dim
linear_key_head_dim      -> linear_key_head_dim
linear_value_head_dim    -> linear_value_head_dim
linear_num_key_heads     -> linear_num_key_heads
linear_num_value_heads   -> linear_num_value_heads
```

因此，Qwen3 是普通 config 映射；Qwen3.5 是文本 config、视觉 config、特殊 token、线性注意力参数和 mRoPE 参数的合并。

## 3. 位置编码差异

Qwen3 使用普通 RoPE：

```python
"position_embedding_type": "rope"
```

Qwen3.5 使用 multimodal RoPE：

```python
"position_embedding_type": "mrope"
```

在 `Qwen3_5Config.__post_init__()` 中，代码会从 `rope_scaling` 提取：

```python
self.mrope_section = self.rope_scaling.get("mrope_section")
self.rotary_base = self.rope_scaling.get("rope_theta")
self.rotary_percent = self.rope_scaling.get("partial_rotary_factor")
```

在 `Qwen3_5McaGPTModel.__init__()` 中，模型会重建 rotary embedding：

```python
self.rotary_pos_emb = Qwen3VLMultimodalRotaryEmbedding(...)
self.mrope_section = self.config.mrope_section
```

这对应 Qwen3.5 dense 的 temporal / height / width 三分量位置编码。Qwen3 没有这个逻辑。

## 4. Attention 结构差异

Qwen3 dense 仍是每层 full attention 的标准 Transformer。MCoreAdapter 里直接走通用 `McaGPTModel` layer spec。

Qwen3.5 dense 使用 Qwen3-Next 风格 hybrid attention stack：

```text
linear_attention
linear_attention
linear_attention
full_attention
```

如果 HF config 没有显式给 `layer_types`，`Qwen3_5Config.__post_init__()` 会按 `linear_attention_freq` 自动生成：

```python
self.layer_types = [
    "linear_attention" if bool((i + 1) % self.linear_attention_freq) else "full_attention"
    for i in range(self.num_layers)
]
```

同时 Qwen3.5 会注入：

```python
"attention_output_gate": True
"experimental_attention_variant": "gated_delta_net"
```

这意味着 Qwen3.5 不能只靠普通 attention module 适配，必须接入 Megatron-Core 的 experimental attention variant。

## 5. Transformer layer spec 差异

Qwen3 不需要重写 layer spec。

Qwen3.5 重写了：

```python
def _get_transformer_layer_spec(...)
```

并要求：

```python
assert config.transformer_impl == "transformer_engine"
```

当 `experimental_attention_variant` 不为空时，会走：

```python
get_transformer_block_with_experimental_attention_variant_spec(...)
```

这部分是 Qwen3.5 支持 Gated DeltaNet、full attention 间隔层、QK-Norm、RMSNorm、SwiGLU 和 attention output gate 的关键。

## 6. 权重转换差异

Qwen3 的权重转换主要是标准 Transformer 权重映射：

```text
lm_head.weight                  -> output_layer.weight
model.embed_tokens.weight       -> embedding.word_embeddings.weight
model.norm.weight               -> decoder.final_layernorm.weight
.mlp.gate_proj/up_proj.weight   -> .mlp.linear_fc1.weight
.mlp.down_proj.weight           -> .mlp.linear_fc2.weight
.self_attn.q/k/v_proj.weight    -> .self_attention.linear_qkv.weight
.self_attn.o_proj.weight        -> .self_attention.linear_proj.weight
.self_attn.q_norm/k_norm.weight -> .self_attention.q/k_layernorm.weight
```

Qwen3.5 除了标准 MLP 和 full attention 转换，还要处理 Gated DeltaNet 线性注意力。

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

这一步由 `Qwen3_5_GDNConverOp` 处理。它会根据：

```text
linear_key_head_dim
linear_value_head_dim
linear_num_key_heads
linear_num_value_heads
```

把 HF 权重拆成 `q/k/v/z/b/a`，再组合成 Megatron-Core 期望的格式。

Qwen3.5 还需要转换 GDN 相关权重：

```text
.linear_attn.conv1d.weight -> .self_attention.conv1d.weight
.linear_attn.dt_bias       -> .self_attention.dt_bias
.linear_attn.A_log         -> .self_attention.A_log
.linear_attn.norm.weight   -> .self_attention.out_norm.weight
.linear_attn.out_proj      -> .self_attention.out_proj
```

此外，Qwen3.5 使用 zero-centered RMSNorm，需要专门的 `ZeroCenteredRMSNormConverOp`：

```text
HF -> MCA: weight - 1
MCA -> HF: weight + 1
```

## 7. 多模态路径差异

Qwen3 dense 是纯文本 LLM 适配，forward 走通用 `McaGPTModel.forward()`。

Qwen3.5 dense 即使叫 dense，也包含原生多模态路径。`Qwen3_5Model.__init__()` 会在 pipeline first stage 构造视觉模型：

```python
if self.pre_process:
    self.vision_model = Qwen3_5VisionModel._from_config(...)
```

forward 时，Qwen3.5 会：

```text
1. 如果没有 position_ids，用 get_rope_index() 生成 multimodal position ids
2. 根据 context parallel / sequence parallel 切分输入范围
3. 先用 embedding 得到文本 inputs_embeds
4. 如果有 pixel_values，把图像送入 vision_model
5. 如果有 pixel_values_videos，把视频送入 vision_model
6. 将视觉 embedding scatter 回 image/video token 对应位置
7. 把融合后的 decoder_input 送入 Megatron decoder
```

所以 Qwen3.5 的 forward 不是普通文本 forward，而是多模态 embedding 构造 + decoder forward。

## 8. 分布式 checkpoint 差异

Qwen3 使用：

```python
register_dist_config("qwen3", default_dist_config)
```

Qwen3.5 使用：

```python
default_dist_config
  + gdn_dist_config
  + vision_model 特殊规则
```

其中 vision tower 权重被标记为：

```python
pre_process_weights=["vision_model.*"]
duplicated_weights=["vision_model.*"]
```

这说明 Qwen3.5 checkpoint 转换不仅要处理 decoder 层，还要处理只存在于 pre-process stage 的视觉模型权重。

## 9. 添加工作量对比

添加 Qwen3 dense 主要需要：

```text
1. 注册 qwen3 model_type
2. 复用 McaModelConfig
3. 复用 McaGPTModel
4. 写标准 HF config 到 MCA config 的字段映射
5. 注入 rope、RMSNorm、SwiGLU、QK-Norm 等常量
6. 写标准 QKV、MLP、Norm、Embedding、LM Head 权重转换
```

添加 Qwen3.5 dense 需要：

```text
1. 新增 Qwen3_5Config
2. 新增 Qwen3_5Model
3. 新增 Qwen3_5Template
4. 处理 text_config 嵌套字段
5. 处理 vision_config 和 image/video token id
6. 处理 rope_parameters、mrope_section、rotary_base、rotary_percent
7. 处理 layer_types 和 full_attention_interval
8. 接入 gated_delta_net experimental attention variant
9. 限制或验证 transformer_engine 支持
10. 实现 GDN linear attention 权重转换
11. 实现 zero-centered RMSNorm 转换
12. 构造 vision_model
13. 实现 image/video embedding 注入
14. 处理 context parallel / sequence parallel 下的输入范围
15. 配置 GDN 和 vision tower 的分布式 checkpoint 规则
```

## 10. 风险点

Qwen3 的风险主要在：

```text
QKV bias 是否存在
QK-Norm 字段是否匹配
RoPE 参数是否正确
HF 权重名是否和 transformers 版本一致
```

Qwen3.5 的风险更多：

```text
transformers 里的 qwen3_5 结构变动
text_config 字段路径不匹配
vision_config 缺失或字段变动
rope_parameters / mrope_section 缺失
layer_types 和 full_attention_interval 不一致
GDN 权重 reshape 维度错误
zero-centered RMSNorm 转换遗漏
transformer_engine 或 experimental_attention_variant 不支持
vision tower 权重在 PP/TP/CP 下分布不正确
image/video token scatter 与 position_ids 不一致
```

## 总结

在 `mcore_adapter` 中，Qwen3 dense 是标准 LLM 适配；Qwen3.5 dense 是复合模型适配。

Qwen3 的重点是把 HF 的标准 Qwen3 Transformer 权重映射到 Megatron-Core 的 GPTModel。

Qwen3.5 的重点是让 Megatron-Core 同时理解：

```text
Qwen3-Next 风格 hybrid attention
Gated DeltaNet 线性注意力权重
multimodal RoPE
Qwen3.5 vision tower
图像/视频 token 到 decoder embedding 的融合
```

因此，不能把 Qwen3.5 dense 当作 Qwen3 dense 的配置扩展来做。它必须作为独立模型类型维护。
