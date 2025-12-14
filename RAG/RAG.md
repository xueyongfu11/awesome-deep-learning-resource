[TOC]



# RAG

## Paper

- https://github.com/AkariAsai/self-rag
  - Self-RAG 是按需检索的（例如，可以根据不同的查询多次检索或完全跳过检索）。这意味着 Self-RAG 不像传统的 RAG 那样固定地进行检索操作，而是更具灵活性。当面对各种各样的查询时，它能够根据实际情况决定检索的次数，甚至可以选择不进行检索，从而更好地适应不同的查询需求。
  - SELF-RAG让模型根据任务需求动态检索、生成和反思自身输出。使用特殊的反思标记（Reflection Tokens）控制模型行为。在推理阶段，模型通过生成反思标记来决定是否检索、评估生成质量并优化输出。

- https://github.com/microsoft/graphrag
  - 基线RAG在以下两种情况下表现不佳：
    - 一是当回答问题需要通过信息点的共同属性来整合分散的信息以生成新见解时
    - 二是当需要全面理解大量数据集合或单个大文档中总结的语义概念时

  - 目标用户
    - GraphRAG旨在支持关键的信息发现和分析场景，这些场景中所需的信息往往分布在多个文档中，可能包含噪声，混杂着错误信息和/或虚假信息，或者用户试图回答的问题比基础数据能够直接回答的内容更加抽象或主题化。
    - GraphRAG被设计用于那些用户已经接受过负责任的分析方法培训，并且期望具备批判性思维的环境中。尽管GraphRAG能够在复杂信息主题上提供高度洞察，但仍需要领域专家对其生成的回答进行人工分析，以验证和补充GraphRAG的结果。

- https://github.com/infiniflow/ragflow
  - RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding
  - 支持树结构索引方法RAPTOR
  - 支持graphrag，但是不能对多文档的知识图谱进行链接
  - 支持text2sql
  - 支持Contextual Retrieval, launched by Claude，即为每个chunk添加额外的补充信息
  - 支持分层的知识库索引，主要是基于Page Rank计算得分权重
  - 支持multi-agent
- https://github.com/labring/FastGPT
  - FastGPT is a knowledge-based platform built on the LLMs, offers a comprehensive suite of out-of-the-box capabilities such as data processing, RAG retrieval, and visual AI workflow orchestration, letting you easily develop and deploy complex question-answering systems without the need for extensive setup or configuration
  - 应用编排能力：对话工作流、插件工作流；工具调用；循环调用
  - 知识库能力：多库复用，混用；chunk记录修改和删除
  - 增量更新算法：实现对新数据的快速整合，无需重建整个索引，显著降低计算成本。
- https://github.com/HKUDS/LightRAG
  - 背景：传统方法难以捕获实体之间的复杂关系，导致生成的答案不够连贯；在处理多实体间的相互依赖时，现有方法难以生成全面、连贯的答案。
  - LightRAG 通过引入图结构增强文本索引和检索流程，利用图形表示实体及其关系，提升信息检索的上下文相关性。
  - 双层检索范式：低层检索：专注于具体实体及其直接关系；高层检索：涵盖更广泛的主题和抽象概念。

- https://github.com/1Panel-dev/MaxKB
  - 内置强大的工作流引擎和函数库，支持编排 AI 工作过程，满足复杂业务场景下的需求

- https://github.com/langgenius/dify
  - Dify是一个开源的大型语言模型（LLM）应用开发平台。它直观的界面将代理式人工智能工作流程、检索增强生成（RAG）管道、代理能力、模型管理、可观测性功能等众多功能相结合，让你能够快速地从原型阶段过渡到生产阶段。

- https://github.com/Mintplex-Labs/anything-llm
  - 一个一体化的桌面和Docker人工智能应用程序，内置了RAG（Retrieval-Augmented Generation，检索增强生成）、人工智能代理以及更多功能。
- https://github.com/langflow-ai/langflow
  - Langflow 是一个低代码应用程序构建器，用于构建检索增强型生成（RAG）和多智能体人工智能应用。它是基于 Python 的，并且对任何模型、API 或数据库都具有不可知性。

- ACL2023关于基于检索来增强大模型的讲习班ppt
  - https://acl2023-retrieval-lm.github.io/
  - 陈丹奇组工作

- Lost in the Middle: How Language Models Use Long Contexts
  - [相关blog](https://zhuanlan.zhihu.com/p/643723202)
  - 1. 语言模型在长输入上下文的中间使用信息时会带来性能下降，并且随着输入上下文的增长，性能会进一步恶化，其更偏向于头尾两个位置
    2. 对检索到的文档进行有效的重新排序，将相关信息推向输入上下文的开头或排序列表截断，必要时返回更少的文档，这个是值得深入的方向

- Copy is All You Need
  - ICLR2023  
  - [聊聊我的AI大黄蜂：Copy is All You Need背后的故事](https://zhuanlan.zhihu.com/p/647457020)
  - 方法：通过文档片段检索的方式来进行文本生成

- In-Context Retrieval-Augmented Language Models
  - [相关blog](https://zhuanlan.zhihu.com/p/647112059)
  - 将已经生成的一些token通过检索的方式获取一些文档，并把这些文档作为prompt加在已经生成文本的前面，然后继续生成后续的token 

- [Ziya-Reader：注意力增强的训练方法](https://mp.weixin.qq.com/s/ekAyYT-Fxj5fw8GNk6Rg0g)

## Embedding

- [Later Chunking技术](https://mp.weixin.qq.com/s/V_4Sxkh01Q-hrBXrv61IFw)
  - https://github.com/jina-ai/late-chunking
- https://huggingface.co/sensenova/piccolo-large-zh-v2
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/TencentBAC/Conan-embedding-v1

## Blog

- [The Rise and Evolution of RAG in 2024 A Year in Review](https://ragflow.io/blog/the-rise-and-evolution-of-rag-in-2024-a-year-in-review#agentic-and-memory)

- [RAPTOR 一种基于树的RAG方法，RAG的准确率提高 20%](https://www.cnblogs.com/xiaoqi/p/18060281/RAPTOR)

- https://www.anthropic.com/news/contextual-retrieval

- [RAG实战全解析：一年探索之路](https://zhuanlan.zhihu.com/p/682253496)

- [北京大学发布AIGC的检索增强技术综述](https://mp.weixin.qq.com/s/o8oTN06UsQSlb5BNyJH23w)

- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)

- [基础RAG技术](https://blog.csdn.net/baidu_25854831/article/details/135331625)

- [高级RAG技术](https://blog.csdn.net/baidu_25854831/article/details/135592272)

- [综述-面向大模型的检索增强生成（RAG）](https://mp.weixin.qq.com/s/TbjbLY6a1h7rgvM5IE4vaw)

- [大模型检索增强生成（RAG）有哪些好用的技巧](https://www.zhihu.com/question/625481187/answer/3279041129)

- [万字长文总结检索增强 LLM](https://zhuanlan.zhihu.com/p/655272123)
