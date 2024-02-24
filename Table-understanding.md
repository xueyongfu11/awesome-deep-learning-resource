<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [survey](#survey)
  - [table interpretation](#table-interpretation)
  - [table pre-train](#table-pre-train)
  - [table-qa](#table-qa)
- [resource](#resource)
- [Datasets](#datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!-- TOC -->

- [Repo](#repo)
- [Paper](#paper)
    - [survey](#survey)
    - [table interpretation](#table-interpretation)
    - [table pre-train](#table-pre-train)
    - [table-qa](#table-qa)
- [resource](#resource)
- [Datasets](#datasets)

<!-- /TOC -->


# Repo

- https://github.com/wenhuchen/OTT-QA

- https://github.com/microsoft/TUTA_table_understanding

- https://github.com/google-research/tapas

- https://github.com/microsoft/Table-Pretraining

- https://modelscope.cn/models/damo/nlp_convai_text2sql_pretrain_cn/summary

- https://github.com/submissionTmp/TabularNet

- https://github.com/alibaba/AliceMind/tree/main/SDCUP

- https://github.com/facebookresearch/TaBERT

- https://github.com/majidghgol/TabularCellTypeClassification

- https://github.com/kianasun/table-understanding-system


# Paper

## survey

- From tabular data to knowledge graphs: A survey of semantic table interpretation tasks and methods

- Table Pre-training: A Survey on Model Architectures, Pre-training Objectives, and Downstream Tasks

- Transformers for Tabular Data Representation: A Survey of Models and Applications
  - 2021, 重要

- Table understanding approaches for extracting knowledge from heterogeneous tables
  - 2021，引用内容很旧，质量整体一般

## table interpretation

- [表格解析挑战赛--冠军方案分享](https://zhuanlan.zhihu.com/p/632436613)
  - 该表格解析任务旨在从一张给定表格中判断其表头和数据，并判断表头间的层级关系，从而实现对表格的要素抽取任务

- MATE: Multi-view Attention for Table Transformer Efficiency
  - 提出了一种稀疏attention对表格进行建模，具体就是token的一部分注意力头只能attend所在行的其他token，另外一部分注意力头只能attend所在列的其他token
  - github.com/google-research/tapas

- StruBERT: Structure-aware BERT for Table Search and Matching
  1. 将表格按照行方向和列方向进行线性化，然后使用bert进行encoding，cell的embedding使用cell内所有token的embedding的average
  2. 对行方向encoding结果用vertical attention，列方向encoding结果用horizontal attention
  3. https://github.com/medtray/StruBERT



- Extraction of Product Specifications from the Web - Going Beyond Tables and Lists
  - <details>
    <summary>阅读笔记: </summary>
    1. 网页数据中的商品说明书信息抽取  <br>
    </details>

- Permutation Invariant Strategy Using Transformer Encoders for Table Understanding
  - Findings-NAACL  
  - <details>
    <summary>阅读笔记: </summary>
    1. 通过一种排列不变性策略对table进行encoding。在column分类，关系抽取，实体链接等表格理解任务上取得了不错的效果  <br>
    2. 排列不变性：同一个column的不同value cell的position id，从同一个位置index开始编码  <br>
    <img src="./assets\PI.png" align="middle" />
    </details>

- Classification of Layout vs Relational Tables on the Web: Machine Learning with Rendered Pages
  - ACM  
  - <details>
    <summary>阅读笔记: </summary>
    1. 通过构建表格特征如何行列位置，cell文本长度，高度宽度等特征  <br>
    2. 基于构架好的特征进行表格的分类  <br>
    </details>

- TAPEX: TABLE PRE-TRAINING VIA LEARNING A NEURAL SQL EXECUTOR
  - ICLR  
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一种基于神经网络SQL执行器的表格预训练模型  <br>
    2. 预训练：采样不同复杂度级别的sql template，执行获取结果，使用BART模型预训练，以sql+flatten table作为encoder的输入，以sql执行结果作为decoder的输出  <br>
    3. 微调时使用question+flatten table作为输入  <br>
    </details>

- Numerical Tuple Extraction from Tables with Pre-training
  - KDD 
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出一种基于预训练的表格中数据元组的提取方法，提取方法是把元组的提取转成多个cell的二元关系分类问题  <br>
    2. 特征输入：将每个cell的text用[SEP]分割，每个cell的pos embedding均从0开始编码；cell的起始行，cross row num，起始列，cross col num，模态类型id（text，visual）;使用TaFor模型提取cell的是视觉特征，作为visual token  <br>
    3. 预训练任务：获取cell的text embedding、该cell被mask后通过context得到的mask位置的embedding，使用contrastive learning来拉近二者的空间距离；cell-level masked-language-model  <br>
    </details>


- TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data
  - KDD
  - <details>
    <summary>阅读笔记: </summary>
    1. 建模任务：表格理解（区域检测、cell分类）  <br>
    2. cell-level特征：text（char长度）、text format（是否是数字、是否是文本）、cell format（cell行列信息、字体粗细等）、text embedding  <br>
    3. 使用wordnet构建相似字词，基于wordnet Tree来构建不同cell中的字词的关系，使用GIN网络学习cell之间的关系embedding  <br>
    4. 使用两个BiGRU网络对表格的行列维度进行建模，得到cell embedding <br>
    5. 将两种embedding concat，进行cell分类，对于表格区域检测任务，将同一行或者同一列的cell embedding进行average pooling
    <img src="./assets\tabularNet.png" align="middle" />
    </details>

- Numerical Formula Recognition from Tables
  - KDD  
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出一种表格中数值公式识别方法。方法是将该任务转化成result cell识别和cell关系分类两个子任务。任务适用于类财务表格数据  <br>
    2. encoder模型：将cell的行列header信息以及行的visual信息融合，作为cell的特征。为了融入context信息，将行text feature和visual feature concat之后使用LSTM来建模context特征，将列text feature 用LSTM建模context特征。最后将cell的行列特征concat之后，使用不同的header来建模result cell分类和cell关系分类任务。  <br>
    </details>

- TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
  - ACL 
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出一个混合的表格-文本问答数据集，并提出能够建模表格-文本数据的模型TAGOP  <br>
    2. 模型以large-bert为backbone，输入question，以row方向flatten的table，以及与表格相关联的paragraph  <br>
    3. 以I/O的方式抽取所有的span；使用cls预测计算操作符，对于divide、diff、change_ratio计算操作符，还需要预测顺序；使用cls，table的avg pooling，paragraph的avg pooling进行单位scale的预测  <br>
    </details>

## table pre-train

- UniTabE: Pretraining a Unified Tabular Encoder for Heterogeneous Tabular Data

- TABBIE: Pretrained Representations of Tabular Data
  1. 使用原始bert模型对cell进行embedding并取平均，得到所有cell的embedding
  2. 使用两个transformer模型对行列cell embedding进行建模，并使用corrupt cell detection任务进行预训练
  3. 下游任务直接获取相应的双向embedding进行合并
  4. NAACL2021

- TURL: Table Understanding through Representation Learning
  - Proceedings of the VLDB Endowment  [[code]](https://github.com/sunlab-osu/TURL)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一种表格预训练模型，使用structure-aware transformer对table进行encoding,并创新性地提出被masked实体恢复预训练任务  <br>
    2. structure-aware transformer：table caption可以attend所有的cell，而cell只能attend相同行或者列的其他cell <br>
    3. 主要用来做table interpretation任务
    <img src="" align="middle" />
    </details>

- TAPAS: Weakly Supervised Table Parsing via Pre-training
  - ACL  [[code]](https://github.com/google-research/tapas)
  - <details>
    <summary>阅读笔记: </summary>
    1. 模型的输入：position id、segment id、row id、column id、rank id（数值或者日期的顺序），表示cell是否是先前问答历史中的答案的id  <br>
    2. 预训练时将table和table中涉及的实体描述等信息作为输入，使用了MLM和实体文本描述和table是否匹配等两个任务，第二个任务作用不大  <br>
    3. 微调:对于cell selection，不存在聚合函数，损失是column选择的损失+column中cell选择的损失；对于scalar answer，需要预测聚合函数，这块比较复杂，参考论文  <br>
    <img src="./assets\tapas.png" align="middle" />
    </details>

- Tabular Cell Classification Using Pre-Trained Cell Embeddings
  1. 使用了表格预训练来对cell进行更好的表征：类似CBOW和skipGram的w2v的向量预训练方法，使用target cell来预测surrounding cells和使用surrounding cells来预测target cell
  2. 单元格分类：使用两个lstm分别建模行和列，将单元格的两个不同方向的隐向量拼接起来做分类


## table-qa

- Answering Numerical Reasoning Questions in Table-Text Hybrid Contents with Graph-based Encoder and Tree-based Decoder

- UniRPG: Unified Discrete Reasoning over Table and Text as Program Generation

- TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
  -  [[code]](https://nextplusplus.github.io/TAT-QA/)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出一个混合的表格-文本问答数据集，并提出能够建模表格-文本数据的模型TAGOP  <br>
    2. 模型以large-bert为backbone，输入question，以row方向flatten的table，以及与表格相关联的paragraph  <br>
    3. 以I/O的方式抽取所有的span；使用cls预测计算操作符，对于divide、diff、change_ratio计算操作符，还需要预测顺序；使用cls，table的avg pooling，paragraph的avg pooling进行单位scale的预测  <br>
    <img src="./assets\tatqa.png" align="middle" />
    </details>

- TABERT: Pretraining for Joint Understanding of Textual and Tabular Data
  1. 该模型只适合DB表格，也就是只有列表头，没有行表头
  2. 先基于context获取表格中最相近的一些行，然后使用bert将context和每行的拼接进行 encoding
  3. 使用vertical attention机制，是的同列的cell可以互相注意到，从而建模行之间的关联
  4. ACL2020

# resource

- SemTab challenge：https://www.cs.ox.ac.uk/isg/challenges/sem-tab/

# Datasets
- https://nextplusplus.github.io/TAT-QA/
- https://nextplusplus.github.io/TAT-HQA/


