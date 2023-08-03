<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [paper](#paper)
- [term weight](#term-weight)
- [ES](#es)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Repo
- 信息检索优质资源汇总 https://github.com/harpribot/awesome-information-retrieval
- query，doc，profile搜索排序模型 https://github.com/linkedin/detext
- https://github.com/terrifyzhao/text_matching
- https://github.com/TharinduDR/Simple-Sentence-Similarity
- https://github.com/NTMC-Community/MatchZoo-py
- https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match
- https://github.com/MachineLP/TextMatch
- https://github.com/pengming617/text_matching
- https://github.com/shenweichen/DeepMatch
- https://github.com/seatgeek/fuzzywuzzy
- https://github.com/DengBoCong/text-similarity
  
  
## paper

- RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering
  - 2021 NAACL
  - 阅读笔记：
    1. 提出了一个双塔的向量检索模型
    2. 提出了三种训练优化方法：跨batch负样本（相对于batch内负样本）；训练一个交互模型，使用交互模型来获取 难负样本；根据训练好的交互模型，获取更多的训练集
  - code: https://github.com/PaddlePaddle/Research/tree/master/NLP/NAACL2021-RocketQA

- VIRT: Improving Representation-based Models for Text Matching through Virtual Interaction
  - year:2021 
  - 阅读笔记：
    1. 提出一种将交互模型的知识蒸馏到双塔模型的方法
    2. 将双塔的隐层做交互来模拟交互模型的交互表示，然后跟交互模型的交互表示计算L2损失
    3. 双塔模型的输出做双向注意力计算，得到u和v，u、v、|u-v|、max(u,v)拼接后接分类层
  - code: 

- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
  - 2020 SIGIR
  - 阅读笔记：
    1. re-rank：基于maxsim进行doc排序，具体是计算query中的每个token的embedding和doc中每个token的embedding的cosine相似度并取最大，然后相加，作为相似度进行文档排序
    2. 用mask将query pad到固定长度，这种数据增强方式有一定的性能提升
    3. end2end：使用faiss，query中的每个token的embedding来检索，然后将检索结果合并，然后将召回结果使用步骤1的方法re-rank
    4. bert模型的输出经过一个线性层进行降维128，模型训练时使用LTR的pairwise loss，最大化相关doc和无关doc的分数差
  - code: https://github.com/stanford-futuredata/ColBERT

- Poly-encoders: architectures and pre-training strategies for fast and accurate multi-sentence scoring
  - year: 2020 ICLR
  - 阅读笔记: 
    1. 提出跟bi-encoder推理速度相当，精度与cross-encoder接近的poly-encoder模型，该模型延续的bi-encoder的双transformer结构
    2. 使用m个codex来表征context的global特征，m是超参数，影响推理速度
    3. candidate的表征extand到m个condex，来计算注意力。然后根据注意力得分加权codex，从而得到context的表征，最后和candidate的表征计算得分
  - code: 非官方 https://github.com/sfzhou5678/PolyEncoder


## term weight
- https://github.com/AdeDZY/DeepCT


## ES
- https://github.com/elastic/elasticsearch
- https://github.com/medcl/elasticsearch-analysis-ik
  - 中文分词器
- https://github.com/medcl/elasticsearch-analysis-pinyin
- https://github.com/NLPchina/elasticsearch-analysis-ansj


