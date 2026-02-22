[TOC]



# Paper

- [对合成数据的最后一次总结](https://zhuanlan.zhihu.com/p/1965442666754979695)

- GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation
  - 2025.05
  - 研究背景: 常见合成数据流水线会踩四个坑：1) 事实不准（幻觉）；2) 长尾知识覆盖不足；3) 知识结构浅（难支持多跳推理）；4) 输出同质化导致过拟合风险。
  - GraphGen:
    - 把文档切成语义块，用一个更强的Synthesizer LLM抽取实体/关系并跨片段合并，得到 KG。这样既能处理长文、噪声和知识分散，也能降低胡编概率。
    - 用校准思想评估待学习模型的知识盲点：对 KG 中每条边，先由Synthesizer LLM生成多种同义改写及其否定版本，然后让待提升的模型对这些陈述做二分类判断并给出置信度，如果模型的置信度与真实正确率不一致，就说明懂得不稳/有盲点，把这类知识点优先拿来生成训练样本。
    - 用受约束的图遍历抽取子图，QA 的最小生成单元不是单句，而是一个k-hop 子图
    - 把子图翻译成三种 QA 风格：Atomic QA：单知识点（单节点/单边），适合补基础事实；Aggregated QA：把子图内多点知识组织成一段连贯长答案，再反向生成问题，训练综合归纳；Multi-hop QA：强调连接多个知识点、需要多步推理的问题。
- Synthetic Continued Pretraining
  - 2024.09
  - 背景：把一个大模型适配到小而专的私域/冷门语料时，直接继续预训练（CPT）往往学不进去。原因在于——预训练式记知识非常吃表达多样性：同一个事实/概念如果只在小语料里出现一两次，模型很难像在互联网海量语料那样，通过大量不同说法把知识写进参数。
  - 作者提出的核心思路是：既然小语料本身太浓缩、缺少多样表达，那就先用小语料合成出一个更适合学习的大而多样的训练语料，再去做继续预训练。
  - 作者给出的具体合成算法叫 EntiGraph：对每篇文档抽取一组关键实体/概念；在抽到的实体之间挑选子集，让一个强模型根据原文解释实体间关系，相当于把文档内容重表达为实体—关系—文本的形式，然后汇总为合成语料。
- Genie: Achieving Human Parity in Content-Grounded Datasets Generation
  - 2024.01
  - 背景：很多内容依托任务（比如 给定一段文档再回答问题/做摘要/做信息抽取）真正卡在高质量训练数据稀缺。人工标注长文本数据成本高；而很多现成数据集又来自新闻/论坛等噪声来源，天然不够干净、忠实。
  - 把原始网页/文档变成可喂给模型的干净片段，对每个内容片段，Genie 用 few-shot prompt 让大模型生成任务样本。
  - 过滤 Filtering：格式过滤：缺字段、结构不对的直接丢；忠实度过滤（Faithfulness）：把 grounding content 当 premise、生成文本当 hypothesis，用 NLI 模型判定是否蕴含/矛盾；质量过滤（Quality）：用奖励模型当自动质检员，论文使用 OpenAssistant 的 DeBERTa-v3 reward model，并把阈值设为 0.5（低于就丢）。
- Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling
  - 2024.01，WRAP
  - 研究背景：主流大模型预训练依赖海量网页抓取数据，但这些文本往往结构混乱、噪声大、表达糟糕
  - WRAP 的核心不是让 LLM编新知识，而是让一个现成的指令模型对网页做信息保持（information-preserving）的改写：Easy：像给幼儿讲解一样简单；Medium：高质量英语（类似 Wikipedia 的行文）；Hard：更晦涩、简练、抽象；Q/A：对话式问答格式。每个样本限制 最多 300 tokens，避免长段改写导致信息丢失。
  - 只用合成改写会让文本过于干净，模型可能不适应真实网页的错别字、奇怪符号等噪声。于是 WRAP 采用 真实 C4 与改写数据 1:1 混合采样来训练，兼顾表达质量和真实噪声鲁棒性。
  - 结论: 在一些数据（如 ArXiv、HackerNews）上，训练含合成改写数据的模型 perplexity 接近降低到原来的 1/3。跨多个子域平均，perplexity 提升约 50%（更低）
- SELF-QA: Unsupervised Knowledge Guided Language Model Alignment
  - 2023.05
  - 背景：已有的自举式方法（如 Self-Instruct）虽然能减少标注，但仍依赖少量人工 seed 指令，而且对生成数据的领域覆盖和答案正确性控制不足。
  - 因此作者提出 SELF-QA：用海量无监督知识替代人工 seed，让模型像人类自我提问—自我作答一样，从领域知识里自动长出可用于 SFT 的指令数据，从而同时提升领域定制与正确性保障的可能性。
  - 把一段无监督文本/知识塞进提示词里，让模型基于它生成尽可能多样的指令问题。关键约束是：问题必须能在脱离原文的情况下成立（因此要求不要用 this/these等指代词、不要依赖上文）。
  - 接着用同一份背景知识和生成的问题，让模型在提示词约束下产出答案：要求回答尽量充分，但不改变原知识中的关键信息；同时避免based on the above article这类泄露提示词痕迹的表达。
  - 知识来源可以是非结构化：网页、书籍等清洗后直接用；结构化：表格、知识图谱等需要先转成文本。
  - 作者也坦承：即使在提示词里强约束，模型仍会生成违规文本（指代词、固定措辞、格式不可解析等），所以需要后处理的启发式/规则过滤来删掉不合格样本并修正格式。
- LongForm: Effective Instruction Tuning with Reverse Instructions
  - 2023.04
  - 反向生成能引出这段文本的指令，已知一段高质量人类文本（作为输出），倒过来问 LLM：这段话可能是在回答什么指令？
  - 更聪明地挑人类文本，保证多样性与长度覆盖
  - 用 GPT-3（text-davinci-003）零样本生成指令，并加入长度控制

## 数学/代码数据生成

- TREESYNTH: Synthesizing Diverse Data from Scratch via Tree-Guided Subspace Partitioning
  - 2025.03
  - 背景：常见的数据合成方法，依赖 seed data / prompt，视角更局部，规模变大后容易出现分布偏、重复、空间塌缩（space collapse），多样性不够。温度采样（temperature sampling）会更随机，但可能牺牲质量且覆盖有限；Evol-Instruct、Persona Hub 等也可能受模型偏好/初始数据限制，导致覆盖不全面。
  - TreeSynth 的核心思想：把任务数据空间类比成决策树的空间划分——叶子节点之间互斥（保证差异/多样性）且穷尽（保证覆盖/不塌缩）。
  - 对任意一个当前空间节点（用文本描述表示），TreeSynth反复做两步，构建一棵空间划分树。
  - 对每个叶子节点，把从根到叶的路径上累积的属性约束拼成该子空间描述，然后让 LLM 只在该子空间约束下生成样本；最后汇总所有叶子节点样本，得到覆盖全面且更均衡的合成数据集。
  - 实验结果：多样性更高，微调后下游表现更稳、更可扩展。
  - 数据划分举例：TreeSynth 先让模型随便生成几道数学题当探针，再从中总结出一个最能区分差异的划分标准（比如按题型分成购物找零/行程速度/几何）。
    然后它在每个子空间（叶子）里按约束分别生成一批题，最后把各子空间的题合并。
    因为每个子空间都被覆盖，所以数据不会集中在某一种题型上，多样性更高。

## Agent数据合成

- [Agentic数据合成：合成DeepResearch数据格式](https://mp.weixin.qq.com/s/Cp-KKgzC8fa1nWXhRsQ2HQ)

## RAG

- HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
  - 2024.05，NIPS
  - 背景：论文指出：即使有检索增强，LLM 仍然很难把大量新经验（新文档）跨段落整合起来——因为主流做法往往把每个 passage 孤立编码/孤立检索，需要的信息如果分散在多个段落里，就很难在一次检索中被一起召回。而人脑可以依靠联想式记忆更快串起线索。
  - 因此他们提出：借鉴神经科学中的海马体索引理论，为 LLM 构建一种更像长期记忆系统的检索框架，让检索本身具备跨段落路径搜索/联想的能力。
  - 离线建海马体索引——把语料变成开放式知识图谱：对每个 passage 做 OpenIE，抽取三元组，形成无 schema 的知识图谱。额外用检索编码器在 KG 上补同义/相近连接（synonymy edges），把相似但不完全相同的概念连起来，帮助后续从部分线索补全完整记忆。
  - 在线检索时做模式补全——实体对齐 + Personalized PageRank（PPR）图搜索：LLM 从 query 里抽取关键命名实体，用检索编码器把这些实体链接到 KG 节点（变成 query nodes），以 query nodes 为种子，在 KG 上跑 Personalized PageRank (PPR)，让概率质量沿图扩散，得到与 query 相关的一片子图/节点集合；这相当于把多跳推理需要的路径探索压缩到一次图检索里完成。
  - 实验结果：检索更准、QA 更强，而且更便宜更快。

































