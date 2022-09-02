<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

  - [Pretain-multi-modal](#pretain-multi-modal)
  - [Pretrain-cv-modal](#pretrain-cv-modal)
  - [Not pretrained](#not-pretrained)
  - [Layout](#layout)
  - [Multi-modal document QA](#multi-modal-document-qa)
- [Dataset](#dataset)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## Pretain-multi-modal

- LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
  - 2022
  - 阅读笔记：
    1. 相比layoutlmv1、v2基于token，v3是基于segment的多模态预训练语言模型
    2. 预训练任务：MLM，图像掩码（基于ViT的patch masked predition），alignment（预测segment对应的patch是否被mask）
    3. 1D、2D绝对位置编码，self-attention中加入1D和2D的相对位置编码信息（同layoutlmv2）
  - code:https://aka.ms/layoutlmv3

- BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents
  - year: 2022 AAAI
  - 阅读笔记: 
    1. 多模态预训练语言模型：tokens，layout，没有visual feature
    2. 预训练任务：masked token prediction，Area-masked Language Model：随机选择某一个候选框，然后以该候选框为中心按照某一个分布随机抽样扩大候选框，然后对新候选框的进行mask，使模型进行预测，使得模型依赖更长的上下文进行预测
    3. 表现结果上，超过其他未加入visual feature的文档多模态预训练语言模型
    4. 但是仍然低于加入图像特征的模型
  - code: https://github.com/clovaai/bros
  - blog:https://mp.weixin.qq.com/s/plZJUjB590VnmjHJcgvm9g

- StructuralLM: Structural Pre-training for Form Understanding
  - year:2021
  - 阅读笔记：  
    1. 基于cell-level的多模态预训练语言模型
    2. 预训练任务：常见的MLVM；将cell-level 2D的位置信息全换成0，预测所在的patch块（一种分类任务）

- Unified Pretraining Framework for Document Understanding
  - year: 2021 NeurIPS
  - 阅读笔记: 
    1. region sacle的多模态预训练语言模型
    2. 使用层次文档embedding方法，以sentence为mask基础
    3. 使用cnn-based模型进行图像特征的提取，每个sentence的visual feature使用POIAlign进行特征提取，并使用量化模块对visual feature进行离散化，方便学习
    4. 使用门控多模态cross注意力方式，得到的text feature和visual feature，concat之后经过FNN之后计算权重
    5. 预训练任务：mask sentence model，图像对比学习：pos使用visual feature和量化模块输出visual feature，text-visual align
  - code: 

- Towards a Multi-modal, Multi-task Learning based Pre-training Framework for Document Representation Learning
  - year: 2022
  - 阅读笔记:  
    1. 多模态预训练文档理解模型，longformer  
    2. 预训练的输入特征：text，text position，layout， token image，images，image position  
    3. token image，images使用同一个resnet + FPN网络生成  
    4. 预训练任务：masked token预测，文档分类，使用一个特殊token得到输出与LDA得到的主题分布j计算softCE；对images顺序打乱，其他不改变，模型判断image和其他特征是否对应  
    5. 应用：特别是文档检索
  - code:
  
- StrucTexT: Structured Text Understanding with Multi-Modal Transformers
  - year: 2021
  - 阅读笔记:  
    1. 多模态预训练文档理解模型  
    2. 预训练input：seg of tokens，image of seg， seg ids，token pos， image of seq pos， modal type  
    3. 预训练task:masked token prediction, image of seg所对应的tokens的长度预测，两个image of seg的方位关系预测
  - code:

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding
  - 2021
  - 笔记  
    1. 多语言的layoutv2  
    2. 提出7中语言的数据集
    3. 支持下游的KV抽取
  - code:https://github.com/microsoft/unilm

- LAYOUTLMV2: MULTI-MODAL PRE-TRAINING FOR VISUALLY-RICH DOCUMENT UNDERSTANDING
  - 2021
  - 笔记：
    1. 三种与训练任务：token掩码，对齐（对图像的部分覆盖，判断是否被覆盖），匹配（判断图像和文字是否匹配）  
    2. 加入相对位置信息，文字和图像都加入位置信息
  - code:https://github.com/microsoft/unilm

- LayoutLM: Pre-training of Text and Layout for Document Image Understanding
  - 2020
  - 笔记：
    1. 类似bert的预训练，加入了字体的2D位置信息，token的图像信息。  
    2. 预训练使用了只对token进行掩码，文档多分类（optional）  
    3. 下游任务：实体抽取，key-value pair抽取，文档分类
  - code:https://github.com/microsoft/unilm
  

## Pretrain-cv-modal

- DIT: SELF-SUPERVISED PRE-TRAINING FOR DOCUMENT IMAGE TRANSFORMER
  - 2022
  - 阅读笔记
    1. 首先训练一个d-VAR模型：使用开源的文档数据集，目的是为了对Dit模型中的patch块进行很好的embedding  
    2. 使用DIT模型对masked的patch输出一个embedding  
    3. 计算两个embedding的交叉熵
  - code：https://github.com/microsoft/unilm/tree/master/dit


## Not pretrained

- Data-Efficient Information Extraction from Form-Like Documents
  - year: 2022
  - 阅读笔记:  
    1. 提出文档信息抽取的迁移学习方法：比如同语言的source domain训练之后，在target domain上微调，或者不同语言训练数据之后的迁移学习  
    2. 模型pipline：候选实体抽取，候选实体排序，赋值
  - code:

- Glean: Structured Extractions from Templatic Documents
  - year：2021
  - 阅读笔记:
    1. paper没有提出新模型去建模doc信息抽取  
    2. 提出一种训练数据管理方法，这种方法是基于候选生成，候选排序，赋值的模型来说的
  - code:

- Using Neighborhood Context to Improve Information Extraction from Visual Documents Captured on Mobile Phones
  - year: 2021
  - 阅读笔记:  
    1. 非预训练的多模态的文档信息抽取  
    2. 对每个target block，融入neighborhood block信息，具体是用另外一个bert把周围的neighbor block进行embedding，
    concat到target block中的每个token
  - code:
  
- TRIE: End-to-End Text Reading and Information Extraction for Document Understanding
  - year: 2021 ACM MM2020
  - 阅读笔记:  
    1. 一种end2end的文档信息抽取：文本检测，文本识别，信息抽取
    2. 通过ROIAlign方法从文本检测和识别模块中获取visual features
    3. 将文本特征和l文本框即layout信息融合，并通过自注意力进行建模，最后得到text features
    4. 将text features和visual features融合人得到context features，然后再与text features fuse之后送给LSTM网络，进行信息抽取
  - code:
  
- Representation Learning for Information Extraction from Form-like Documents
  - 2020 ACL
  - 笔记：
    1. 使用NLP工具进行候选实体的高召回  
    2. 融入候选的neighbor特征：left，above10%的文本特征，相对候选实体位置的相对位置特征，但是不融入候选实体的文本特征，根据这些特征得到embedding  
    3. 将候选实体类型的embedding和2中embedding进行binary cls  
    4. 未使用多模态信息
  - code: https://github.com/Praneet9/Representation-Learning-for-Information-Extraction


## Layout

- layout parser https://github.com/Layout-Parser/layout-parser

- layout parser train https://github.com/Layout-Parser/layout-model-training

- 文档理解相关资源 https://github.com/tstanislawek/awesome-document-understanding


## Multi-modal document QA

- https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/doc_vqa



 # Dataset

- https://github.com/baidu/DuReader/tree/master/DuReader-vis
  - 中文多模态文档阅读理解数据集
  - Paper: DuReadervis: A Chinese Dataset for Open-domain Document Visual Question Answering

- https://github.com/clovaai/cord
  - 发票数据集, 英文

- https://github.com/HCIILAB/EPHOIE
  - 中文 
  
- https://github.com/beacandler/EATEN
  - 中文数据集 

- https://github.com/RuilinXu/GovDoc-CN
  - 中文 