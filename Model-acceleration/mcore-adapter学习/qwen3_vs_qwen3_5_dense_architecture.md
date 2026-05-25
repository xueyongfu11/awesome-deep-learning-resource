# Qwen3 与 Qwen3.5 Dense 模型架构差异

本文只比较 dense 模型的架构差异。这里的 `dense` 指每个 token 都经过同一套参数，不走 MoE 路由；它不等于“纯文本”。Qwen3.5 的 dense 版本仍然可以是原生多模态模型。

## 结论

Qwen3 dense 基本还是标准 decoder-only Transformer 路线：GQA + SwiGLU + RoPE + pre-RMSNorm，并在 Qwen3 中去掉 QKV bias、加入 QK-Norm 来提高训练稳定性。

Qwen3.5 dense 的核心变化更大：文本 backbone 换成 Qwen3-Next 风格的混合注意力 decoder，每 4 层中 3 层是 Gated DeltaNet 线性注意力，1 层是 Gated Attention 全注意力。它还内建多模态输入路径，使用三分量 multimodal RoPE，并接入 Qwen3-VL 风格 vision tower。

所以一句话概括：

```text
Qwen3 dense:   标准 Transformer dense LLM
Qwen3.5 dense: dense FFN + 线性/全注意力混合栈 + 原生多模态位置编码和视觉塔
```

## 主要差异表

| 维度 | Qwen3 dense | Qwen3.5 dense |
| --- | --- | --- |
| 总体定位 | 文本 LLM，支持 thinking / non-thinking 模式 | 原生多模态 foundation model，可处理文本、图像、视频 token |
| 基础结构 | Decoder-only Transformer | Qwen3-Next 风格 hybrid decoder，外加 vision tower |
| 注意力层 | 标准 full attention，使用 GQA | 3:1 混合栈：3 层 Gated DeltaNet 线性注意力 + 1 层 Gated Attention 全注意力 |
| 长上下文成本 | full attention 层随序列长度二次增长，靠 YaRN / DCA 等扩展推理上下文 | 大部分层是线性注意力，长上下文和视觉/视频 token 的计算成本更低 |
| 位置编码 | RoPE；长上下文阶段提高 RoPE base，并可用 YaRN / DCA 扩展 | multimodal RoPE，head 维度切成 temporal / height / width 三段 |
| 归一化与 MLP | pre-RMSNorm + SwiGLU；Qwen3 加 QK-Norm、去 QKV bias | 仍使用 RMSNorm / SwiGLU，但 attention block 变成 Gated DeltaNet / Gated Attention 混合 |
| 多模态模块 | dense LLM 本体不含 vision tower | 配置中包含 `vision_config`、image/video token id、视觉 encoder 输出到文本 hidden size |
| tokenizer / vocab | Qwen tokenizer，技术报告给出 vocab size 151,669 | 配置示例中 vocab size 248,320，并保留 image/video 等特殊 token |
| 代表配置 | Qwen3-32B：64 层，Q/KV heads = 64/8，32K 原生上下文，YaRN 到约 128K | Qwen3.5-27B：64 层，hidden size 5120，`full_attention_interval=4`，`max_position_embeddings=262144` |

## Qwen3 Dense 的架构要点

Qwen3 dense 共有 0.6B、1.7B、4B、8B、14B、32B 等规模。官方技术报告描述它与 Qwen2.5 dense 架构相近，主要组件是：

- Grouped Query Attention，减少 KV cache 和注意力计算开销。
- SwiGLU MLP。
- RoPE 位置编码。
- pre-normalization 的 RMSNorm。
- 相比 Qwen2，去掉 QKV bias。
- 在 attention 中加入 QK-Norm，目标是提升训练稳定性。

Qwen3 的长上下文路线主要来自训练和推理扩展：预训练末期使用 32K 长上下文数据；推理侧结合 RoPE base 调整、YaRN 和 Dual Chunk Attention，将部分模型扩展到约 128K 上下文。

这意味着 Qwen3 dense 的 block 形态仍然很“经典”：

```text
RMSNorm -> GQA full attention -> residual
RMSNorm -> SwiGLU FFN          -> residual
```

## Qwen3.5 Dense 的架构要点

Qwen3.5 dense 的“dense”只说明不是 MoE；它的 attention 架构已经不是 Qwen3 dense 的标准 full-attention Transformer。

Hugging Face Transformers 文档说明，Qwen3.5 dense / Qwen3.6 dense 使用 3:1 hybrid attention stack：

```text
linear_attention
linear_attention
linear_attention
full_attention
```

这个模式循环出现。以 Qwen3.5-27B 配置为例，它有 64 层，所以可理解为 16 组：

```text
16 x (3 x Gated DeltaNet + 1 x Gated Attention)
```

其中：

- Gated DeltaNet 是线性注意力路径，适合降低长上下文和视觉/视频 token 的成本。
- Gated Attention 是 full attention 路径，用来保留全局精确交互能力。
- `layer_types` 明确记录每层是 `linear_attention` 还是 `full_attention`。
- `full_attention_interval=4` 表示每 4 层放一个 full attention 层。
- `mrope_section` 将 RoPE head 维度拆为 temporal / height / width，用于图像和视频 token 的位置对齐。
- `vision_config` 表明模型包含视觉编码器，视觉输出维度会投到语言模型 hidden size。

因此 Qwen3.5 dense 的典型 block 不是“每层 full attention”，而是：

```text
多数层:
RMSNorm -> Gated DeltaNet linear attention -> residual
RMSNorm -> SwiGLU FFN                      -> residual

间隔层:
RMSNorm -> Gated Attention full attention  -> residual
RMSNorm -> SwiGLU FFN                      -> residual
```

## 对推理和适配的影响

1. 不能把 Qwen3.5 dense 当成 Qwen3 dense 的小版本。

   二者的 dense 含义相同，都是非 MoE；但 attention module、position embedding、多模态输入和配置字段都不同。适配框架时，Qwen3.5 需要单独处理 `qwen3_5` / `qwen3_5_text`、`layer_types`、Gated DeltaNet、multimodal RoPE 和 vision tower。

2. Qwen3.5 的长上下文效率来自架构，而不只是 RoPE 扩展。

   Qwen3 主要还是 full attention Transformer，通过 YaRN / DCA 等方式扩展上下文；Qwen3.5 则让多数层走线性注意力，因此对 256K 级上下文、图像 token、视频 token 更友好。

3. Qwen3.5 对运行时 kernel 更敏感。

   Gated DeltaNet 快速实现依赖 `causal_conv1d`、`fla` 等 kernel；缺失时可能回退到更慢、更耗显存的 PyTorch 实现。部署时不能只看参数量，还要看推理框架是否支持 Qwen3.5 的 hybrid attention。

4. 多模态位置编码不能按普通 RoPE 简化。

   Qwen3.5 的 multimodal RoPE 把位置维度拆成 temporal / height / width。如果在 Megatron、MCoreAdapter、vLLM、llama.cpp 等系统里重写 rotary embedding，需要保持这个拆分，否则图像和视频 token 的位置会错位。

## 资料来源

- Qwen3 Technical Report, arXiv:2505.09388: https://arxiv.org/abs/2505.09388
- Qwen3 technical report HTML: https://ar5iv.labs.arxiv.org/html/2505.09388v1
- Qwen3-32B Hugging Face model card: https://huggingface.co/Qwen/Qwen3-32B
- Hugging Face Transformers Qwen3.5 documentation: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Alibaba Group Qwen3.5 release note: https://www.alibabagroup.com/en-US/document-1960233590314762240
- Qwen3.5-27B ModelScope page: https://www.modelscope.cn/models/Qwen/Qwen3.5-27B
