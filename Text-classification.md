<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [document clssification](#document-clssification)
- [多分类](#多分类)
- [few-shot](#few-shot)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## document clssification
- HAN模型，层次注意力模型，word->sentence-document https://zhuanlan.zhihu.com/p/44776747

- https://github.com/ricardorei/lightning-text-classification
- https://github.com/IBM/MAX-Toxic-Comment-Classifier
- https://github.com/brightmart/text_classification
- https://github.com/facebookresearch/fastText
  - fastText模型：ngram+word特征；向量平均；层次softmax

## 多分类
- https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel
- https://github.com/lonePatient/Bert-Multi-Label-Text-Classification
- https://github.com/hellonlp/classifier_multi_label_textcnn
- https://github.com/lancopku/SGM
- https://github.com/brightmart/multi-label_classification
- https://github.com/RandolphVI/Multi-Label-Text-Classification
- https://github.com/nocater/text_gcn
- https://github.com/yao8839836/text_gcn


## few-shot

- https://github.com/ShaneTian/Att-Induction
- https://github.com/xionghhcs/few_shot_learning

- Dynamic Memory Induction Networks for Few-Shot Text Classification
  - year: 2020
  - 阅读笔记: 
    1. two stage：pretrained model on train datasets，同时得到记忆权重W
    2. meta-learning stage：使用动态记忆模块对支持集，记忆权重进行动态信息融合
    3. 将2得到的embedding和query set进行cos计算
  - code: 

- Induction Networks for Few-Shot Text Classification
  - year: 
  - 阅读笔记: 2019
    1. 采样一个episode，支持集（C个类别*K个样本），请求集（C*K）
    2. 多支持集和请求集的样本text都用encoder进行embedding，具体是LSTM，然后使用self-attention加权得到text的句子embedding
    3. 计算类别embedding：使用胶囊网络对每个样本进行嵌入，然后通过动态路由的方法加权类别的所有样本，得到类别embedding
    4. 类别embedding和query集样本的两个embedding计算mse得分。
  - code: https://github.com/wuzhiye7/Induction-Network-on-FewRel

