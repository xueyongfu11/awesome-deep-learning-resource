<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [knowledge distillation](#knowledge-distillation)
- [inference tools](#inference-tools)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# knowledge distillation
- https://github.com/HoyTta0/KnowledgeDistillation
- https://github.com/airaria/TextBrewer
- 


# C++重写
- https://github.com/NVIDIA/FasterTransformer


# inference tools
- tensorRT https://github.com/shouxieai/tensorRT_Pro
- 神经网络可视化工具 https://github.com/lutzroeder/Netron
- https://github.com/htqin/awesome-model-quantization
  - 模型量化paper整理
- https://github.com/airaria/TextPruner
- Asset eXchange models
  - https://developer.ibm.com/articles/introduction-to-the-model-asset-exchange-on-ibm-developer/


# Quantization

### 2023

- QLora
  - [blog](https://zhuanlan.zhihu.com/p/632229856)
  - 使用4位的NormalFloat（Int4）量化和Double Quantization技术。4位的NormalFloat使用分位数量化，通过估计输入张量的分位数来确保每个区间分配的值相等，
  Double Quantization是将额外的量化常数进行量化。
  - 梯度检查点会引起显存波动，从而造成显存不足问题。通过使用Paged Optimizers技术，使得在显存不足的情况下把优化器从GPU转义到CPU中。
  - QLora张量使用时，会把张量 反量化 为BF16，然后在16位计算精度下进行矩阵乘法。
  - https://github.com/artidoro/qlora
  - https://huggingface.co/blog/4bit-transformers-bitsandbytes
    - 介绍及使用

### 2022

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
  - [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
  - 量化的基本原理
  - transformers中的8bit的矩阵乘法