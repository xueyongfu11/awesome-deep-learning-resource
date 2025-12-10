[TOC]



# LLM Doc Understanding

- MDocAgent: A Multi-Modal Multi-Agent Framework for  Document Understanding
  - 2025.03
  - 论文认为现有的方法未有效综合文本和视觉信息
  - MDocAgent是一个多模态多agent文档理解方法，包含5个专家agent：
    - 先介绍下数据预处理：对文档中的每一页，使用OCR或者pdf解析等方式获取文本内容，同时将每一页转成image，分别使用ColBERT和ColPali模型检索跟用户query相近的文本或者图片。
    - General Agent：根据检索到的文本和图片，给出初始的问答
    - Critical Agent：根据初始的回答、query、检索的文本 or 图片，生成“回复用户query最重要的信息”
    - Text Agent：根据“回复用户query最重要的信息”、query、检索的文本，生成基于检索文本的回复
    - Image Agent：根据“回复用户query最重要的信息”、query、检索的图像，生成基于检索图像的回复
    - Summarizing Agent：基于两个回复结果生成最终的答案。
- PDF-WuKong : A Large Multimodal Model for Efficient Long PDF Reading  with End-to-End Sparse Sampling
  - 2024.10
  - PDF-WuKong使用了稀疏采样器检索出于query相关的文档片段和图片，然后基于检索结果进行回复。
  - 构建1.1M长文档问答数据集：将plain文档解析，抽取解析结果，使用chatgpt、gemini构建问题和答案。

  - 局限性：适合答案分布比较集中的场景，对于需要多处总结并推理的问题未被验证效果