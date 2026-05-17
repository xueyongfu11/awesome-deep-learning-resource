# awesome-deep-learning-resource

深度学习、自然语言处理、大模型与多模态相关资料整理。

## 目录

### 大模型与智能体

- [Large-language-model](./Large-language-model/)：大语言模型、开源 LLM、长上下文、数据合成、微调、CoT、垂直领域大模型等
- [Agent](./Agent/)：Agent 评测、对话记录学习等
- [RAG](./RAG/)：检索增强生成、RAG 分类与系统实践
- [RLHF](./RLHF/)：RLHF、Reward Model、PPO、在线/离线 RLHF、推理强化学习等
- [PEFT](./PEFT/)：LoRA、参数高效微调
- [Quantization](./Quantization/)：模型量化、LLM 量化
- [Model-acceleration](./Model-acceleration/)：模型压缩、推理加速、vLLM 等
- [Prompt-learning](./Prompt-learning.md)：Prompt Learning
- [In-context-learning](./In-context-learning.md)：In-context Learning
- [Pretrained-language-model](./Pretrained-language-model.md)：预训练语言模型

### 多模态、具身智能与语音

- [MultiModal](./MultiModal/)：多模态学习、VLM、Qwen Omni、Speech LLM 等
- [VLA](./VLA/)：Vision-Language-Action、机器人与具身智能
- [Document-understanding](./Document-understanding/)：文档理解、版面分析、LLM 文档理解
- [Table-understanding](./Table-understanding.md)：表格理解
- [TTS-And-STT](./TTS-And-STT.md)：语音合成与语音识别

### 训练、工程与基础能力

- [Machine-learning](./Machine-learning.md)：机器学习基础
- [Base-neural-network](./Base-neural-network.md)：神经网络基础
- [Multi-task-learning](./Multi-task-learning.md)：多任务学习
- [Low-rank-decomposition](./Low-rank-decomposition.md)：低秩分解
- [Datasets](./Datasets.md)：数据集
- [CUDA](./CUDA/)：CUDA、Triton、NCCL、Tensor Core、PTX 等高性能计算内容
- [Programming-language](./Programming-language/)：编程语言相关内容
- [Paper-writing](./Paper-writing.md)：论文写作
- [Other](./Other.md)：其他资料

### CV、推荐与知识图谱

- [Computer-vision](./Computer-vision/)：计算机视觉、OCR、CV 分类
- [Recommendation-system](./Recommendation-system/)：推荐系统、CTR
- [KG](./KG/)：知识图谱、多模态知识图谱

### 传统 NLP 与信息处理

- [Some-NLP-Task](./Some-NLP-Task/)：NLP 基础任务、文本摘要、文本生成、文本纠错、文本分类、NLI、MRC 等
- [Information-extraction](./Information-extraction/)：信息抽取、事件抽取、关系抽取、关键词抽取、主题模型
- [NER](./NER/)：命名实体识别
- [Text-retrieval](./Text-retrieval/)：文本检索、向量检索、Embedding、Reranker
- [Dialogue-system](./Dialogue-system/)：对话系统、KBQA、Text2SQL、任务型对话等
- [Sentiment-analysis](./Sentiment-analysis/)：情感分析、细粒度情感分析

## 使用说明

- 每个专题目录下通常包含该方向的综述、论文笔记、工程实践或数据集整理。
- 新增内容时优先放入对应专题目录；如果是大模型、Agent、RAG、RLHF、多模态等内容，优先维护在 README 前半部分对应分类下。
- 传统 NLP 任务保留在最后一组，便于把当前重点的大模型相关内容前置展示。
