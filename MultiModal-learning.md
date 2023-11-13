<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [cross-modal retrieval](#cross-modal-retrieval)
  - [Information extraction](#information-extraction)
  - [datasets](#datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://gitee.com/zidongtaichu/multi-modal-models
  - 紫东太初多模态大模型
- https://github.com/Eurus-Holmes/Awesome-Multimodal-Research
- https://github.com/thuiar/MMSA
- https://github.com/pliang279/awesome-multimodal-ml
- https://github.com/Paranioar/Cross-modal_Retrieval_Tutorial
  - 跨模态检索paper set
- https://github.com/IDEA-CCNL/Fengshenbang-LM
  - 太乙：多模态预训练语言模型
- https://wukong-dataset.github.io/wukong-dataset/benchmark.html
  - WuKong, benchmark, baseline


# Paper

## multi-modal 

### 2023

- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
  - https://github.com/salesforce/LAVIS/tree/main/projects/blip2

### 2022

- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
  - https://github.com/salesforce/BLIP

### 2021

- Learning Transferable Visual Models From Natural Language Supervision
  - CLIP模型
  - https://github.com/openai/CLIP

## Information extraction

### 2023

- GeoLayoutLM: Geometric Pre-training for Visual Information Extraction
  - https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM

- Unifying Vision, Text, and Layout for Universal Document Processing

### 2020

- Multimodal Joint Attribute Prediction and Value Extraction for E-commerce Product
  - 阅读笔记:  
    1. 从产品描述中提取属性和属性值，通过融合产品图片特征，来提高抽取效果
    2. 使用global visual gate对text和image特征进行融合，本质是根据text和image特征生成一个权重（用来加权token-visual），然后得到融合的多模态特征
    3. 多模态特征，cls特征，tokens加权特征，作为输入，使用多标签分类方法，分类出属性
    4. regional visual gate：根据3识别出的属性和visual特征融合得到，目的是为了关注图像中的特定区域
    5. tokens特征，多模态特征，区域门控加权的视觉特征，进行属性值抽取
    6. 应为属性和属性值存在一致性，加入kl损失，然后使用多任务学习方式优化
    7. 提出多模态实验数据集
  - code: https://github.com/jd-aig/JAVE


# datasets

- https://paperswithcode.com/dataset/flickr30k-cna
  - 中文数据集
- [多模态分析数据集（Multimodal Dataset）整理](https://zhuanlan.zhihu.com/p/189876288)
- [华为诺亚开源首个亿级中文多模态数据集-悟空](https://mp.weixin.qq.com/s/qXwnKCVi01LRjDYpYl9jLw)
