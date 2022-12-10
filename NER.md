<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
  - [地址解析](#%E5%9C%B0%E5%9D%80%E8%A7%A3%E6%9E%90)
- [Paper](#paper)
  - [Chinese datasets](#chinese-datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Repo
- GlobalPointer：用统一的方式处理嵌套和非嵌套NER https://github.com/bojone/GlobalPointer
- https://github.com/lonePatient/BERT-NER-Pytorch
- https://github.com/fastnlp/TENER
- https://github.com/SadeemAlharthi/Arabic_Question-Answering_System_Using_Search_Engine_Techniques
- https://github.com/LeeSureman/Flat-Lattice-Transformer
- https://github.com/FuYanzhe2/Name-Entity-Recognition
- https://github.com/loujie0822/DeepIE
- https://github.com/hecongqing/CCKS2019_EventEntityExtraction_Rank5
- 

## 地址解析
- https://github.com/modood/Administrative-divisions-of-China
  - 中文地址多级标准数据库
- https://github.com/xiangyuecn/AreaCity-JsSpider-StatsGov
  - 中文地址多级标准数据库
- https://huggingface.co/cola/chinese-address-ner
- https://github.com/youzanai/trexpark
  - 有赞开源：收货地址预训练语言模型
- https://www.kaggle.com/competitions/scl-2021-ds/overview/description
  - Kaggle比赛
- https://github.com/leodotnet/neural-chinese-address-parsing
  - This page contains the code used in the work "Neural Chinese Address Parsing" published at NAACL 2019.


# Paper

- NFLAT: Non-Flat-Lattice Transformer for Chinese Named Entity Recognition
  - year: 2022
  - 阅读笔记：
    1. 提出了一种non-flat-lattice transformer结构来建模中文ner，相比flat-lattice模型，更少的计算量、占用更少的显存和支持处理更长的文本
    2. 将char embedding（原始文本）作为Q，word embedding作为K、V，基于一种InterAttention结构做attention计算。
    3. InterAttention结构：Q加上可学习的参数再与K和相对距离矩阵相乘，相对距离矩阵是char位置和word起始位置的相对位置embedding的concat，然后再进行FFN，LN等
    4. 在InterAttention结构基础上stack上几层transformer网络，使得信息进一步融合
  - 

- Hierarchically-Refined Label Attention Network for Sequence Labeling
  - year:2019  EMNLP
  - 阅读笔记：
    1. 相较于CRF，LAN能够捕捉更长期的标签依赖，更快的解码速度
    2. 基于BiLSTM网络，将BILSTM的隐层输出H作为Q，label embedding作为K，V，使用多头的自注意力网络，得到的输出再cancat上input的embedding
    3. 最后直接使用自注意力得分，得到最后的输出


## Chinese datasets

- [中文命名实体识别数据集](https://mp.weixin.qq.com/s/bIRhscHb1VjMAM1axLcUhw)
- [中文医疗信息处理评测基准CBLUE](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.70e81343dFDilp&dataId=95414)

- https://github.com/liucongg/NLPDataSet
  - 包括中文摘要数据集、中文片段抽取式阅读理解数据集（QA）、中文文本相似度数据集和中文NER数据集

- 微博实体识别.
  - https://github.com/hltcoe/golden-horse

- boson数据。
  - 包含6种实体类型。
  - https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/boson

- 人民日报数据集。
  - 人名、地名、组织名三种实体类型 
  - 1998：[https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/renMinRiBao](https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/renMinRiBao) 
  - 2004：https://pan.baidu.com/s/1LDwQjoj7qc-HT9qwhJ3rcA password: 1fa3
  
- MSRA微软亚洲研究院数据集。
  - 5 万多条中文命名实体识别标注数据（包括地点、机构、人物） 
  - https://github.com/InsaneLife/ChineseNLPCorpus/tree/master/NER/MSRA

- SIGHAN Bakeoff 2005：一共有四个数据集，包含繁体中文和简体中文，下面是简体中文分词数据。
  - MSR: <http://sighan.cs.uchicago.edu/bakeoff2005/>
  - PKU ：<http://sighan.cs.uchicago.edu/bakeoff2005/> 