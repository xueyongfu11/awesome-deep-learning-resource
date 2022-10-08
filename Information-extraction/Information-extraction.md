<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Repo
- https://github.com/JHart96/keras_gcn_sequence_labelling
- https://github.com/cdjasonj/CCF_IE
- 


 # Paper

 - Unified Structure Generation for Universal Information Extraction
  - year: 2022
  - 阅读笔记：
    1. 提出一种统一信息抽取模型，encoder-decoder的生成模型
    2. 构建结构化schema指示器作为prompt，与text进行拼接，输出结构化抽取语言的结构
    3. 预训练方法：通过构建标准的结构化schema指示器和text，以及结构化抽取语言等，来训练模型的encoder-decoder生成能力；通过构建仅包含结构化抽取语言，来训练模型的decoder的结构化解码能力；通过构建MLM任务，来训练模型的encoder语义编码能力
    4. 在预训练任务中，对结构化抽取语言中添加spot负样本，对应值设置为NULL，以此来避免错误生成spot的情况下生成错误的span内容


