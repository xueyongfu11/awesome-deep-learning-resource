<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo-keyphrase](#repo-keyphrase)
- [Repo-新词发现](#repo-新词发现)
- [keyphrase](#keyphrase)
- [datasets](#datasets)
  - [english](#english)
  - [chinese](#chinese)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## Repo-keyphrase

- https://github.com/MaartenGr/KeyBERT
- https://github.com/JackHCC/Chinese-Keyphrase-Extraction
- https://github.com/bigzhao/Keyword_Extraction
- https://github.com/thunlp/BERT-KPE
- https://github.com/microsoft/OpenKP
- https://github.com/boudinfl/pke
- https://github.com/dongrixinyu/chinese_keyphrase_extractor
- https://github.com/sunyilgdx/SIFRank_zh
- https://github.com/luozhouyang/embedrank
- https://github.com/swisscom/ai-research-keyphrase-extraction
- https://github.com/memray/OpenNMT-kpg-release
- https://github.com/jiacheng-ye/kg_one2set
- https://github.com/xgeric/UCPhrase-exp


## Repo-新词发现

- https://github.com/Changanyue/newwor-discovery


## keyphrase

- MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction
  - 2022
  - 阅读笔记：
    1. 一种无监督的关键短语抽取方法，不同于phrase-document相似度计算方法，提出了一种基于mask的document-document的相似度计算方法
    2. 根据一定规则提取候选关键短语，然后将候选短语在原始document中进行mask，然后使用bert对mask前后的document进行embedding，相似度越高，短语重要性越低
    3. 提出了一种预训练方法。使用无监督算法提取关键短语，作为正样本。仍然使用2中的方法对文档进行embedding。使用triple-loss来建模。
  - code: https://github.com/LinhanZ/mderank

- Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context
  - year: 2021 EMNLP
  - 阅读笔记：
      1. 提出一种联合建模局部和全局上下文的无监督关键短语抽取算法
      2. 全局：使用bert对document和候选的keyphrase embedding，使用曼哈顿距离计算相似度
      3. 局部：将候选短语两两计算dot-product，并使用一种位置感知的中心度计算方法，本质是计算node的总度数，并根据两两短语与document的开头（或者结尾）距离越近，权重越大的规则计算
      4. 最后排序结果参考两者得分相乘后的总分
  - code: 


## datasets

### english
- KP20k：https://github.com/memray/seq2seq-keyphrase
- KPTimes：https://github.com/ygorg/KPTimes

### chinese
- https://www.clue.ai/introduce.html
  - CLUE 论文关键词抽取