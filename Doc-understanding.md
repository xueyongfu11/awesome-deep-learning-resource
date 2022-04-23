
## Pretain-multi-modal

- Paper: Towards a Multi-modal, Multi-task Learning based Pre-training Framework for Document Representation Learning
  - year: 2022
  - 阅读笔记:  
    1.多模态预训练文档理解模型，longformer  
    2.预训练的输入特征：text，text position，layout， token image，images，image position  
    3.token image，images使用同一个resnet + FPN网络生成  
    4.预训练任务：masked token预测，文档分类，使用一个特殊token得到输出与LDA得到的主题分布j计算softCE；对images顺序打乱，其他不改变，模型判断image和其他特征是否对应  
    5.应用：特别是文档检索
  - code:

- Paper: Data-Efficient Information Extraction from Form-Like Documents
  - year: 2022
  - 阅读笔记:  
    1.提出文档信息抽取的迁移学习方法：比如同语言的source domain训练之后，再target domain上微调，或者不通语言训练数据之后的迁移学习  
    2.模型pipline：候选实体抽取，候选实体排序，赋值
  - code:

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding
  - 2021
  - 笔记  
    1.多语言的layoutv2  
    2.提出7中语言的数据集
    3.支持下游的KV抽取
  - code:https://github.com/microsoft/unilm

- LAYOUTLMV2: MULTI-MODAL PRE-TRAINING FOR VISUALLY-RICH DOCUMENT UNDERSTANDING
  - 2021
  - 笔记：
    1.三种与训练任务：token掩码，对齐（对图像的部分覆盖，判断是否被覆盖），匹配（判断图像和文字是否匹配）  
    2.加入相对位置信息，文字和图像都加入位置信息
  - code:https://github.com/microsoft/unilm

- Glean: Structured Extractions from Templatic Documents
  - year：2021
  - 阅读笔记:
    1.paper没有提出新模型去建模doc信息抽取  
    2.提出一种训练数据管理方法，这种方法是基于候选生成，候选排序，赋值的模型来说的
  - code:

- LayoutLM: Pre-training of Text and Layout for Document Image Understanding
  - 2020
  - 笔记：
    1.类似bert的预训练，加入了字体的2D位置信息，token的图像信息。  
    2.预训练使用了只对token进行掩码，文档多分类（optional）  
    3.下游任务：实体抽取，key-value pair抽取，文档分类
  - code:https://github.com/microsoft/unilm
  
- Representation Learning for Information Extraction from Form-like Documents
  - 2020 ACL
  - 笔记：
    1.使用NLP工具进行候选实体的高召回  
    2.融入候选的neighbor特征：left，above10%的文本特征，相对候选实体位置的相对位置特征，但是不融入候选实体的文本特征，根据这些特征得到embedding  
    3.将候选实体类型的embedding和2中embedding进行binary cls  
    4.未使用多模态信息
  - code: https://github.com/Praneet9/Representation-Learning-for-Information-Extraction

## Pretrain-cv-modal

- DIT: SELF-SUPERVISED PRE-TRAINING FOR DOCUMENT IMAGE TRANSFORMER
  - 2022
  - 阅读笔记
    1.首先训练一个d-VAR模型：使用开源的文档数据集，目的是为了对Dit模型中的patch块进行很好的embedding  
    2.使用DIT模型对masked的patch输出一个embedding  
    3.计算两个embedding的交叉熵
  - code：https://github.com/microsoft/unilm/tree/master/dit


## Layout

- layout parser https://github.com/Layout-Parser/layout-parser

- layout parser train https://github.com/Layout-Parser/layout-model-training

- 文档理解相关资源 https://github.com/tstanislawek/awesome-document-understanding


## Dataset

- 发票数据集 https://github.com/clovaai/cord


## chinese

- 中文 https://github.com/HCIILAB/EPHOIE

- 中文数据集 https://github.com/beacandler/EATEN

- 中文 https://github.com/RuilinXu/GovDoc-CN


 