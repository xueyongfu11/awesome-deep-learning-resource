[TOC]



## 研究背景

论文动机很现实：在 **RAG** 和各类 **Agent** 系统里，“检索阶段”的 embedding（召回）与 reranker（精排）几乎决定了最终上层生成/推理的上限。随着大模型进入多语言、长文本、复杂指令的应用场景，传统 encoder-only 方案与以往训练范式很难同时满足：**多语言覆盖、指令对齐、长上下文理解、跨领域泛化** 等需求。([arXiv](https://arxiv.org/pdf/2506.05176))

作者提出 **Qwen3 Embedding** 系列：在 Qwen3 foundation models 上系统性构建 embedding + reranking 模型，并通过“合成数据 + 多阶段训练 + 模型融合”把性能推到 SOTA/强竞争水平，同时开源供社区使用。

## 系统架构总览

这套体系不是一个模型，而是一组“可组合”的检索组件：

- Embedding 模型：Qwen3-Embedding-0.6B / 4B / 8B
- Reranker 模型：Qwen3-Reranker-0.6B / 4B / 8B
- 共同特点：基于 Qwen3 dense 基座、32K 上下文；embedding 支持 MRL（可变维度表示）；两类模型都支持“指令可定制/任务可定制”。

结构上：

- Embedding：把 `{Instruction}+{Query}` 拼接输入，取末尾 token（[EOS]）的隐藏状态作为向量表示。
- Reranker：把相似度判断写成二分类（只能回答 yes/no），用模型对 “yes” 与 “no” 的概率（logits）计算相关性分数。

MRL介绍：

- 它解决的问题是：下游任务到底需要 128/256/512/1024… 维往往不确定，但传统 embedding 一旦训练好维度就固定了。MRL 的思路是让 embedding 的前 k 维本身就能作为一个“更小的 embedding”使用（像套娃一样：小表示嵌在大表示的前缀里），因此你可以把一个高维向量直接截断成更低维，来换取更快检索 / 更少存储，同时尽量少掉点效果。
- 不是所有 embedding 都能随便截断：只有训练时用了 MRL这类模型，截断才通常靠谱；普通模型直接砍维度往往会明显掉效果。
- 一般做法是：取前 N 维后再做一次 (L2) normalize，再算相似度

## 核心方法

### 第一步：模型怎么“做 embedding / 做精排”

- Embedding（向量化）
  - 输入显式包含 instruction，使向量天然具备“任务/指令条件化”的特征。
  - 通过 MRL 支持在同一模型里导出不同维度的向量，方便在“效果 vs 成本/延迟”之间做部署折中。
- Reranker（精排）
  - 把 Query、Document、Instruction 放进同一上下文，让模型做 point-wise 判断；用 `P(yes)` 与 `P(no)` 归一化得到分数。

### 第二步：多阶段训练流水线

作者给出一个很“工程化但有效”的 recipe（embedding 与 reranker 训练逻辑相近）：

1. Stage 1：大规模弱监督/合成数据预训练（synthetic pair data）
2. Stage 2：高质量数据监督微调（高质量合成数据 + 标注数据）
3. Stage 3：模型融合（model merging）：从 Stage 2 的多个 checkpoint 采样并合并，以增强鲁棒性与泛化。

### 第三步：聪明地生成与筛选训练数据

这篇的一个关键点是：不仅用大模型当 backbone，也用它来“造数据”。

- 合成数据覆盖 retrieval、bitext mining、分类、STS 等多种相似度任务，并在 prompt 里显式控制：query 类型、长度、难度、语言等维度，以提升多样性与真实性。
- 规模上：作者提到总计构建约 150M 多任务弱监督 pair；随后用简单余弦相似度过滤（文中举例阈值 > 0.7）保留约 12M 高质量 pair 进入后续监督训练。

------

## 实验结果

- 在 MTEB Multilingual上，Qwen3-Embedding-8B 的 mean(task) 达到 70.58，并在多个子项上占优。
- 在 MTEB English、CMTEB、MTEB Code上：
  - Qwen3-Embedding-8B：MTEB(Eng,v2) mean(task) 75.22，CMTEB mean(task) 73.83，MTEB(Code) 80.68。
- 文中还说明对比对象包含开源 embedding 系列与部分商业 API（例如 OpenAI 的 text-embedding-3-large、Google 的 Gemini Embedding、Cohere 的 embedding API 等）。

- 先用 embedding 召回 top-100，再用不同 reranker 精排。结果显示 Qwen3-Reranker 系列整体优于对比 reranker；不同任务上大小模型各有优势，例如：

  - Qwen3-Reranker-8B 在多项指标上最高（如 MLDR、MTEB-Code 等列）。

  - 但在 FollowIR 这类“复杂指令检索”指标上，4B 的数值反而更高（表中 FollowIR：4B 为 14.84，8B 为 8.05），说明任务/数据分布下并非“越大越统治”。