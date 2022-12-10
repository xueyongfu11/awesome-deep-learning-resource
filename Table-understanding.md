<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Paper](#paper)
  - [table sub-task](#table-sub-task)
- [datasets](#datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Paper

## table sub-task

- Classification of Layout vs Relational Tables on the Web: Machine Learning with Rendered Pages
  - year: 2022  ACM
  - 阅读笔记：
    1. 通过构建表格特征如何行列位置，cell文本长度，高度宽度等特征
    2. 基于构架好的特征进行表格的分类

- TAPEX: TABLE PRE-TRAINING VIA LEARNING A NEURAL SQL EXECUTOR
  - year: 2022 ICLR
  - 阅读笔记：
    1. 提出了一种基于神经网络SQL执行器的表格预训练模型
    2. 预训练：采样不同复杂度级别的sql template，执行获取结果，使用BART模型预训练，以sql+flatten table作为encoder的输入，以sql执行结果作为decoder的输出
    3. 微调时使用question+flatten table作为输入

- Numerical Tuple Extraction from Tables with Pre-training
  - year: 2022 KDD
  - 阅读笔记: 
    1. 提出一种基于预训练的表格中数据元组的提取方法，提取方法是把元组的提取转成多个cell的二元关系分类问题
    2. 特征输入：将每个cell的text用[SEP]分割，每个cell的pos embedding均从0开始编码；cell的起始行，cross row num，起始列，cross col num，模态类型id（text，visual）;使用TaFor模型提取cell的是视觉特征，作为visual token
    3. 预训练任务：获取cell的text embedding、该cell被mask后通过context得到的mask位置的embedding，使用contrastive learning来拉近二者的空间距离；cell-level masked-language-model
  - code: 

- Numerical Formula Recognition from Tables
  - year: 2021 KDD
  - 阅读笔记: 
    1. 提出一种表格中数值公式识别方法。方法是将该任务转化成result cell识别和cell关系分类两个子任务。任务适用于类财务表格数据
    2. encoder模型：将cell的行列header信息以及行的visual信息融合，作为cell的特征。为了融入context信息，将行text feature和visual feature concat之后使用LSTM来建模context特征，将列text feature 用LSTM建模context特征。最后将cell的行列特征concat之后，使用不同的header来建模result cell分类和cell关系分类任务。
  - code: 

- TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
  - 2021 ACL
  - 阅读笔记：
    1. 提出一个混合的表格-文本问答数据集，并提出能够建模表格-文本数据的模型TAGOP
    2. 模型以large-bert为backbone，输入question，以row方向flatten的table，以及与表格相关联的paragraph
    3. 以I/O的方式抽取所有的span；使用cls预测计算操作符，对于divide、diff、change_ratio计算操作符，还需要预测顺序；使用cls，table的avg pooling，paragraph的avg pooling进行单位scale的预测
  - code：


# datasets

- https://nextplusplus.github.io/TAT-QA/
- https://nextplusplus.github.io/TAT-HQA/


