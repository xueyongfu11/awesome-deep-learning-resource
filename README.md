# awesome-deep-learning-resource

深度学习、自然语言处理、大模型、智能体、多模态与 AI Infra 相关资料整理，包含学习资源、论文笔记、源码解读和工程实践。

## 目录

### 大模型与智能体

- [Large-language-model](./Large-language-model/)：大语言模型、开源 LLM、长上下文、数据合成、微调、CoT、知识存储与注入、垂直领域大模型等
- [Agent](./Agent/)：Agent 基础、评测集、Skills、Hermes Agent 对话记录学习笔记
- [LLM-Memory](./LLM-Memory/)：Agent 记忆管理、大模型个性化、TencentDB Agent Memory 流程与提示词
- [RAG](./RAG/)：检索增强生成、RAG 分类与常见问题
- [RLHF](./RLHF/)：RLHF、Reward Model、在线/离线 RLHF、推理强化学习、OpenRLHF 的 Ray 资源调度与 PPO 训练流程
- [PEFT](./PEFT/)：LoRA、参数高效微调
- [Quantization](./Quantization/)：模型量化、LLM 量化
- [Prompt-learning](./Prompt-learning.md)：Prompt Learning
- [In-context-learning](./In-context-learning.md)：In-context Learning
- [Pretrained-language-model](./Pretrained-language-model.md)：预训练语言模型

### AI Infra、分布式训练与推理优化

- [AI-Infra](./AI-Infra/)：模型压缩与加速、分布式训练、推理引擎、显存优化及技术分享笔记
- [Megatron Core 学习笔记](./AI-Infra/megatron代码学习/README.md)：GPT 主线、张量/流水线/序列/上下文并行、分布式优化器、通信重叠、MoE 与 Transformer Engine / FP8
- [mcore-adapter 学习](./AI-Infra/mcore-adapter学习/)：Qwen3 与 Qwen3.5 Dense 架构对比、模型适配与执行流程
- [vLLM](./AI-Infra/vllm/)：vLLM、CUDA Graph、模型编译与前缀缓存
- [推理系统与技术分享](./AI-Infra/presentation/)：投机解码、KV Connector、DCP/PCP、MoE RL 训练与训推一致性等分享笔记
- [CUDA](./CUDA/README.md)：CUDA、Triton、NCCL、Tensor Core、PTX、CCCL 与性能分析工具

### 多模态、具身智能与语音

- [MultiModal](./MultiModal/)：多模态学习、VLM、Qwen Omni、Speech LLM 等
- [VLA](./VLA/)：Vision-Language-Action、Diffusion 到 VLA、机器人与具身智能
- [VLA 入门教程](./VLA/入门教程/)：机器人规划与控制、抓取、位姿估计、模仿学习、强化学习及 VLA 课程笔记
- [World-Model](./World-Model.md)：世界模型入门教程与学习资料
- [Document-understanding](./Document-understanding/)：文档理解、版面分析、LLM 文档理解
- [Table-understanding](./Table-understanding.md)：表格理解
- [TTS-And-STT](./TTS-And-STT.md)：语音合成与语音识别

### 机器学习基础与通用资源

- [Machine-learning](./Machine-learning.md)：机器学习基础
- [Reinforce-learning](./Reinforce-learning/)：强化学习课程、开源实现、TRPO 与 PPO
- [Base-neural-network](./Base-neural-network.md)：神经网络基础
- [Multi-task-learning](./Multi-task-learning.md)：多任务学习
- [Low-rank-decomposition](./Low-rank-decomposition.md)：低秩分解
- [Datasets](./Datasets.md)：数据集
- [Programming-language](./Programming-language/)：C / C++ 学习资料
- [Paper-writing](./Paper-writing.md)：论文写作
- [Other](./Other.md)：其他资料

### CV、推荐与知识图谱

- [Computer-vision](./Computer-vision/)：计算机视觉、OCR、CV 分类
- [Recommendation-system](./Recommendation-system/)：推荐系统、CTR
- [KG](./KG/)：知识图谱、多模态知识图谱

### 传统 NLP 与信息处理

- [Some-NLP-Task](./Some-NLP-Task/)：NLP 基础任务、文本摘要、文本生成、文本纠错、文本分类、指代消解、NLI、MRC 等
- [Information-extraction](./Information-extraction/)：信息抽取、事件抽取、关系抽取、关键词抽取、主题模型
- [NER](./NER/)：命名实体识别
- [Text-retrieval](./Text-retrieval/)：文本检索、向量检索、Embedding、Reranker 与数据集
- [Dialogue-system](./Dialogue-system/)：对话系统、KBQA、Text2SQL、任务型对话等
- [Sentiment-analysis](./Sentiment-analysis/)：情感分析、细粒度情感分析

## 使用说明

- 按上方主题进入对应目录或文档；专题内容包括资源链接、综述、论文笔记、源码解读、工程实践或数据集整理。
- 分布式训练与推理优化可从 AI-Infra 进入，Megatron Core 学习笔记提供分阶段阅读顺序；机器人与具身智能课程位于 VLA/入门教程。
- 新增内容时优先放入对应专题目录；新增专题或调整目录名称后，同步更新此处的导航链接和说明。
