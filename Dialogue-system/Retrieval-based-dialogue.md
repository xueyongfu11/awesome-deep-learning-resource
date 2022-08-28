<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Repo

- https://github.com/gunthercox/ChatterBot
  - 基于回复选择的问答系统框架

- https://github.com/alibaba/esim-response-selection

- https://github.com/Nikoletos-K/QA-with-SBERT-for-CORD19

- https://github.com/SadeemAlharthi/Arabic_Question-Answering_System_Using_Search_Engine_Techniques

- baidu AnyQ https://github.com/baidu/AnyQ

- https://colab.research.google.com/drive/1tcsEQ7igx7lw_R0Pfpmsg9Wf3DEXyOvk#scrollTo=ZiXOVpeutyWg

- https://github.com/baidu/Dialogue/tree/master/DAM

- https://github.com/gunthercox/ChatterBot

- https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/question_answering/faq_system/
  - paddle政务问答系统  
  
  
# Paper

- Paper: 检索式聊天机器人技术综述  
  - year: 2021    
  - 阅读笔记:      
    1. 叙述了检索式聊天机器人的四种常见建模方法，以及相关方法的优缺点，并分析了当前检索式聊天机器人面临的问题
    2. 个人：每种方法体系都没有做很深的介绍，只是一个简单的概述     

- Paper: ConveRT: Efficient and Accurate Conversational Representations from Transformers
  - year: 2020
  - 阅读笔记: 
    1. 一种快并且有效的双塔结构且基于transformer的对话回复选择预训练模型架构
    2. 使用sub-word，双位置编码相加，6层共享的transformer，单头注意力，未使用full注意力，不同层由浅到深使用更大的注意力窗，并使用量化感知方式进行参数学习
    3. 模型在尺度，参数量，效率，效果都由很强的优势
  - code: 非官方 https://github.com/jordiclive/Convert-PolyAI-Torch

- Paper: Multi-turn Response Selection using Dialogue Dependency Relations
  - year: 2020 EMNLP
  - 阅读笔记: 
    1. paper认为response依赖于不同的对话线，所有需要先对context分类出不同的对话线
    2. 对不同的对话线分别encoding，表征的结果与candidate的表征计算attention，并得到context的表征，然后与candidate计算score
  - code: https://github.com/JiaQiSJTU/ResponseSelection.

- Paper: Poly-encoders: architectures and pre-training strategies for fast and accurate multi-sentence scoring
  - year: 2020 ICLR
  - 阅读笔记: 
    1. 提出跟bi-encoder推理速度相当，精度与cross-encoder接近的poly-encoder模型，该模型延续的bi-encoder的双transformer结构
    2. 使用m个codex来表征context的global特征，m是超参数，影响推理速度
    3. candidate的表征extand到m个condex，来计算注意力。然后根据注意力得分加权codex，从而得到context的表征，最后和candidate的表征计算得分
  - code: 非官方 https://github.com/sfzhou5678/PolyEncoder
  
- Paper: Semantic Role Labeling Guided Multi-turn Dialogue ReWriter
  - year: 2020 EMNLP
  - 阅读笔记: 
    1. 使用unilm的文本生成方式进行对话重写
    2. 训练出了一个在对话（多句，传统SRL无法使用）数据上的语义角色分析模型，来增强语义信息
    3. 将树形的语义角色分析结果转成三元组，与context拼接。
    4. 使用了角色类型，SRL类型，对话重写内容的segment type id；不同三元组之间不直接计算self-attention，三元组只和context计算attention；每个三元组内进行位置编码
  - code: 
  
- Paper: Improving Multi-turn Dialogue Modelling with Utterance ReWriter
  - year: 2019 ACL
  - 阅读笔记: 
    1. 基于encoder-decoder+point network的transformer模型
    2. encoder和decoder的attention计算：encoder中的context，utterance要分别计算，然后concat
    3. point network：分别计算context，utterance的注意力得分，并用一个0-1的拉姆达加权
  - code: https://github.com/chin-gyou/dialogue-utterance-rewriter