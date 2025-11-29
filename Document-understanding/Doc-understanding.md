[TOC]



# Repo

- https://github.com/PaddlePaddle/VIMER#structext
- https://github.com/alibaba/AliceMind
- https://github.com/tstanislawek/awesome-document-understanding
  - 文档理解相关资源

# Resource

- https://rrc.cvc.uab.es/
  - ICDAR比赛
- DocVQA榜单：https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1 
- 网页问答榜单WebSRC：https://x-lance.github.io/WebSRC/index.html


# Paper  
## Pretain-multi-modal  
### 2022  
- ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding  
  - EMNLP  [[code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-layout)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了文档阅读顺序的预训练任务，具体是把attention看作token之间是否相邻，GT是一个01矩阵  <br>
    2. 提出了被替换区域预测的预训练任务，具体是选择部分patch块用其他图像的patch块替换，使用cls来判断哪些patch被替换  <br>
    3. 使用了空间感知的解耦注意力  <br>
    <img src="../assets\ernie-layout.png" align="middle" />
    </details>

- LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding
  - ACL  [[code]](https://github.com/jpWang/LiLT)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出一种语言独立的layout transformer结构，未引入图像特征  <br>
    2. 预训练：使用text流和layout流的双流网络结构，双流之间使用BIACM来进行信息的交互  <br>
    3. 预训练的方式：MLM，通过对bbox进行mask，来预测其所在区域；判断token-box是否对齐等三个任务  <br>
    
    </details> 

- LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking
  -   [[code]](https://aka.ms/layoutlmv3)
  - <details>
    <summary>阅读笔记: </summary>
    1. 相比layoutlmv1、v2基于token，v3是基于segment的多模态预训练语言模型  <br>
    2. 预训练任务：MLM，MIM（预测patch的label），alignment（预测segment对应的patch是否被mask）  <br>
    3. 1D、2D绝对位置编码，self-attention中加入1D和2D的相对位置编码信息（同layoutlmv2）  <br>
    4. patch直接flatten后线性输入，未使用CNN或者faster-RCNN进行特征提取
    
    </details>

- BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents
  - AAAI  [[code]](https://github.com/clovaai/bros)
  - [[blog]](https://mp.weixin.qq.com/s/plZJUjB590VnmjHJcgvm9g)
  - <details>
    <summary>阅读笔记: </summary>
    1. 多模态预训练语言模型：tokens，layout，没有visual feature  <br>
    2. 预训练任务：masked token prediction，Area-masked Language Model：随机选择某一个候选框，然后以该候选框为中心按照某一个分布随机抽样扩大候选框，然后对新候选框的进行mask，使模型进行预测，使得模型依赖更长的上下文进行预测  <br>
    3. 表现结果上，超过其他未加入visual feature的文档多模态预训练语言模型 <br>
    4. 但是仍然低于加入图像特征的模型
    
    </details>

- Towards a Multi-modal, Multi-task Learning based Pre-training Framework for Document Representation Learning
  - <details>
    <summary>阅读笔记: </summary>
    1. 多模态预训练文档理解模型，longformer  <br>
    2. 预训练的输入特征：text，text position，layout， token image，images，image position  <br>
    3. token image，images使用同一个resnet + FPN网络生成  <br>
    4. 预训练任务：masked token预测，文档分类，使用一个特殊token得到输出与LDA得到的主题分布j计算softCE；对images顺序打乱，其他不改变，模型判断image和其他特征是否对应 <br>
    5. 应用：特别是文档检索
    
    </details>

### 2021
- StructuralLM: Structural Pre-training for Form Understanding
  - <details>
    <summary>阅读笔记: </summary>
    1. 基于cell-level的多模态预训练语言模型,使用token+layout等特征  <br>
    2. 预训练任务：常见的MLM任务；将一个cell的2D的位置信息全换成0，预测所在的patch块的位置（一种分类任务）  <br>
    3. patch块的位置：把image划分成等分的N个区域，每个cell所在的区域就是2中提到的patch块的位置  <br>
    
    </details>

- Unified Pretraining Framework for Document Understanding
  - NeurIPS  
  - <details>
    <summary>阅读笔记: </summary>
    1. region sacle的多模态预训练语言模型  <br>
    2. 使用层次文档embedding方法，以sentence为mask基础  <br>
    3. 使用cnn-based模型进行图像特征的提取，每个sentence的visual feature使用POIAlign进行特征提取，并使用量化模块对visual feature进行离散化，方便学习  <br>
    4. 使用门控多模态cross注意力方式，得到的text feature和visual feature，concat之后经过FNN之后计算权重  <br>
    5. 预训练任务：mask sentence model，图像对比学习：pos使用visual feature和量化模块输出visual feature，text-visual align
    
    </details>

- StrucTexT: Structured Text Understanding with Multi-Modal Transformers
  - <details>
    <summary>阅读笔记: </summary>
    1. 多模态预训练文档理解模型  <br>
    2. 预训练input：seg of tokens，image of seg， seg ids，token pos， image of seq pos， modal type  <br>
    3. 预训练task:masked token prediction, image of seg所对应的tokens的长度预测，两个image of seg的方位关系预测  <br>
    
    </details>

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding
  -   [[code]](https://github.com/microsoft/unilm)
  - <details>
    <summary>阅读笔记: </summary>
    1. 多语言的layoutv2  <br>
    2. 提出7中语言的数据集  <br>
    3. 支持下游的KV抽取  <br>
    
    </details>

- LAYOUTLMV2: MULTI-MODAL PRE-TRAINING FOR VISUALLY-RICH DOCUMENT UNDERSTANDING
  -   [[code]](https://github.com/microsoft/unilm)
  - <details>
    <summary>阅读笔记: </summary>
    1. 三种与训练任务：token掩码，对齐（对图像的部分覆盖，判断是否被覆盖），匹配（判断图像和文字是否匹配）  <br>
    2. 加入相对位置信息，文字和图像都加入位置信息  <br>
    </details>

### 2020
- LayoutLM: Pre-training of Text and Layout for Document Image Understanding
  -  [[code]](https://github.com/microsoft/unilm)
  - <details>
    <summary>阅读笔记: </summary>
    1. 类似bert的预训练，加入了字体的2D位置信息，token的图像信息。  <br>
    2. 预训练使用了只对token进行掩码，文档多分类（optional）  <br>
    3. 下游任务：实体抽取，key-value pair抽取，文档分类  <br>
    
    </details>

## Pretrain-cv-modal
### 2022 
- DIT: SELF-SUPERVISED PRE-TRAINING FOR DOCUMENT IMAGE TRANSFORMER
  -   [[code]](https://github.com/microsoft/unilm/tree/master/dit)
  - <details>
    <summary>阅读笔记: </summary>
    1. 首先训练一个d-VAR模型：使用开源的文档数据集，目的是为了对Dit模型中的patch块进行很好的embedding  <br>
    2. 使用DIT模型对masked的patch输出一个embedding  <br>
    3. 计算两个embedding的交叉熵  <br>
    
    </details>

## Non-pretrained  
### 2022    
- Data-Efficient Information Extraction from Form-Like Documents  
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出文档信息抽取的迁移学习方法：比如同语言的source domain训练之后，在target domain上微调，或者不同语言训练数据之后的迁移学习  <br>
    2. 模型pipline：候选实体抽取，候选实体排序，赋值  <br>
    
    </details>

### 2021  
- Glean: Structured Extractions from Templatic Documents
  - <details>
    <summary>阅读笔记: </summary>
    1. paper没有提出新模型去建模doc信息抽取  <br>
    2. 提出一种训练数据管理方法，这种方法是基于候选生成，候选排序，赋值的模型来说的  <br>
    
    </details>

- Using Neighborhood Context to Improve Information Extraction from Visual Documents Captured on Mobile Phones
  - <details>
    <summary>阅读笔记: </summary>
    1. 非预训练的多模态的文档信息抽取  <br>
    2. 对每个target block，融入neighborhood block信息，具体是用另外一个bert把周围的neighbor block进行embedding，
       concat到target block中的每个token  <br>
    
    </details>

- TRIE: End-to-End Text Reading and Information Extraction for Document Understanding
  - ACM MM  
  - <details>
    <summary>阅读笔记: </summary>
    1. 一种end2end的文档信息抽取：文本检测，文本识别，信息抽取  <br>
    2. 通过ROIAlign方法从文本检测和识别模块中获取visual features  <br>
    3. 将文本特征和l文本框即layout信息融合，并通过自注意力进行建模，最后得到text features  <br>
    4. 将text features和visual features融合人得到context features，然后再与text features fuse之后送给LSTM网络，进行信息抽取
    
    </details>

### 2020
- PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks
  - ICPR  [[code]](https://github.com/wenwenyu/PICK-pytorch)
  - <details>
    <summary>阅读笔记: </summary>
    1. 对每个文本片段分别进行embedding。使用word2vec作为token的表示，使用transformer进行encoding；使用resnet对文本片段图像进行特征抽取，最后将两种特征相加  <br>
    2. 步骤1中得到的输出，一是直接输入到BiLSTM+CRF网络进行信息抽取，二是接一个polling layer，作为每个node的embedding，node之间关系的embedding基于node直接的距离以及node自身的宽高属性信息来构建  <br>
    <img src="../assets\pick.png" align="middle" />
    </details>

- Representation Learning for Information Extraction from Form-like Documents
  - ACL  [[code]](https://github.com/Praneet9/Representation-Learning-for-Information-Extraction)
  - <details>
    <summary>阅读笔记: </summary>
    1. 使用NLP工具进行候选实体的高召回  <br>
    2. 融入候选的neighbor特征：left，above10%的文本特征，相对候选实体位置的相对位置特征，但是不融入候选实体的文本特征，根据这些特征得到embedding  <br>
    3. 将候选实体类型的embedding和2中embedding进行binary cls  <br>
    4. 未使用多模态信息
    
    </details>

## Multi-modal document QA

- https://github.com/allanj/LayoutLMv3-DocVQA
- https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/doc_vqa


# Dataset

- https://github.com/baidu/DuReader/tree/master/DuReader-vis
  - 中文多模态文档阅读理解数据集
  - DuReadervis: A Chinese Dataset for Open-domain Document Visual Question Answering
- https://github.com/clovaai/cord
  - 发票数据集, 英文
- https://github.com/HCIILAB/EPHOIE
  - 中文
- https://github.com/beacandler/EATEN
  - 中文数据集
- https://github.com/RuilinXu/GovDoc-CN
  - 中文
