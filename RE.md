<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [joint RE](#joint-re)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Repo
- https://github.com/roomylee/awesome-relation-extraction
  - paper list
- https://github.com/zhoujx4/NLP-Series-relation-extraction
  - 提供了四种关系抽取模型 
- https://github.com/BaderLab/saber/tree/development
  - 生物学领域的信息抽取工具 

- 2021 https://github.com/Receiling/UniRE
- 2020 EMNLP https://github.com/LorrinWWW/two-are-better-than-one
- bert biaffine机制 https://github.com/bowang-lab/joint-ner-and-re
- https://github.com/Receiling/UniRE
- dilated-cnn-ner https://github.com/iesl/dilated-cnn-ner
- https://github.com/yuanxiaosc/Entity-Relation-Extraction
- https://github.com/princeton-nlp/PURE
- https://github.com/thunlp/OpenNRE
- https://github.com/yuanxiaosc/Entity-Relation-Extraction
- https://github.com/yuanxiaosc/Multiple-Relations-Extraction-Only-Look-Once
- https://github.com/yuanxiaosc/Schema-based-Knowledge-Extraction
- https://github.com/xiaofei05/Distant-Supervised-Chinese-Relation-Extraction
- https://github.com/davidsbatista/Snowball
- https://github.com/MichSchli/RelationPrediction
- 2020 bert biaffine机制 https://github.com/bowang-lab/joint-ner-and-re
- 2019 https://github.com/datquocnguyen/jointRE


# Paper

## joint RE

- Two are Better than One: Joint Entity and Relation Extraction with Table-Sequence Encoders
  - year: 2020  EMNLP
  - 阅读笔记：
    1. 使用填表的方式进行关系抽取，使用两种表示table-encoder和seq-encoder
    2. table-encoder使用多源word embedding进行第一个table unit的初始输入，然后使用MD-RNN网络
    3. seq-encoder使用BERT网络模型，但是取消了scaled-dot用2中table作为注意力得分


