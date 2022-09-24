<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
  - [地址解析](#地址解析)
- [Paper](#paper)
- [datasets](#datasets)

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

- Hierarchically-Refined Label Attention Network for Sequence Labeling
  - year:2019  EMNLP
  - 阅读笔记：
    1. 相较于CRF，LAN能够捕捉更长期的标签依赖，更快的解码速度
    2. 基于BiLSTM网络，将BILSTM的隐层输出H作为Q，label embedding作为K，V，使用多头的自注意力网络，得到的输出再cancat上input的embedding
    3. 最后直接使用自注意力得分，得到最后的输出

# datasets
- [中文医疗信息处理评测基准CBLUE](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.70e81343dFDilp&dataId=95414)
- 