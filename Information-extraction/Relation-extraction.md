[TOC]





## Repo

- https://github.com/thunlp/NREPapers
- https://github.com/roomylee/awesome-relation-extraction
  - paper list
- https://github.com/zhoujx4/NLP-Series-relation-extraction
  - 提供了四种关系抽取模型 
- https://github.com/BaderLab/saber/tree/development
  - 生物学领域的信息抽取工具 

- bert biaffine机制 https://github.com/bowang-lab/joint-ner-and-re
- dilated-cnn-ner https://github.com/iesl/dilated-cnn-ner
- https://github.com/yuanxiaosc/Entity-Relation-Extraction
- https://github.com/thunlp/OpenNRE
- https://github.com/yuanxiaosc/Entity-Relation-Extraction
- https://github.com/yuanxiaosc/Multiple-Relations-Extraction-Only-Look-Once
- https://github.com/yuanxiaosc/Schema-based-Knowledge-Extraction
- https://github.com/xiaofei05/Distant-Supervised-Chinese-Relation-Extraction
- https://github.com/davidsbatista/Snowball
- https://github.com/MichSchli/RelationPrediction
- 2020 bert biaffine机制 https://github.com/bowang-lab/joint-ner-and-re
- 2019 https://github.com/datquocnguyen/jointRE


## pipline RE

- Packed Levitated Marker for Entity and Relation Extraction
  - 2022 ACL
  - 阅读笔记：
    1. 提出两种悬浮标记方法，一种是获取所有的span，并为每个span添加悬浮标记，根据span的相对距离对标记进行分组，然后每个组和text拼接后送入encoder，使用悬浮标记表征以及T-cat方法来获取更好的span表征，来达到对span更好的分类（NER）
    2. 另外一种方式是将subject使用固定标记，即在text中插入标记符，object使用悬浮标记方法
  - code: https://github.com/thunlp/PL-Marker

- A Frustratingly Easy Approach for Entity and Relation Extraction
  - 2021 NAACL
  - 阅读笔记：
    1. 使用两个encoder分别对应NER和RE两个任务
    2. NER使用基于span的方法，span的表示使用的特征包含起始token和span长度
    3. 关系抽取使用了两种方法：第一种是对任意两个entity, 在text中找到对应位置后，在头尾插上标记符，输入encoder后将两个span的起始标记特征拼接，作为span pair的特征
    4. 另外一种是基于悬浮标记的方法，将所有的标记对放在text后，标记对可以注意到text，但不能注意到其他标记对
  - code: https://github.com/princeton-nlp/PURE


## joint RE

- Two are Better than One: Joint Entity and Relation Extraction with Table-Sequence Encoders
  - year: 2020  EMNLP
  - 阅读笔记：
    1. 使用填表的方式进行关系抽取，使用两种表示table-encoder和seq-encoder
    2. table-encoder使用多源word embedding进行第一个table unit的初始输入，然后使用MD-RNN网络
    3. seq-encoder使用BERT网络模型，但是取消了scaled-dot用2中table作为注意力得分
  - code: https://github.com/LorrinWWW/two-are-better-than-one

- UNIRE: A Unified Label Space for Entity Relation Extraction
  - 2021 ACL
  - 阅读笔记：
    1. 使用填表的方式进行端到端的实体以及关系抽取
    2. 加入结构限制：实体对称和关系对称；预测出来的实体的概率要大于包含该实体的关系的概率
    3. 解码：将n * n * Y的特征图按照行维度或者列维度压平，然后相邻行之间计算编辑距离，如果大于阈值就作为span的边界；span中的最大概率对应的label作为实体label；关系中最大的概率作为关系的label
  - code: https://github.com/Receiling/UniRE

