<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [document clssification](#document-clssification)
- [多分类](#多分类)
- [few-shot](#few-shot)
- [Chinese datasets](#chinese-datasets)

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

- https://github.com/daooshee/Few-Shot-Learning
- https://github.com/ha-lins/MetaLearning4NLP-Papers
- https://github.com/johnnyasd12/awesome-few-shot-meta-learning

- https://github.com/ShaneTian/Att-Induction
  - 未开放paper，测试发现Induction network loss并不收敛，acc始终是20%
- https://github.com/iesl/metanlp


- PROTOTRANSFORMER: A META-LEARNING APPROACH TO PROVIDING STUDENT FEEDBACK
  - year:2020
  - 阅读笔记：
    1. paper主要应用在code相关的meta-learning任务中，并在NLP任务有很好的效果
    2. 使用robert对模型encoding，同时使用label embedding mean embedding作为一个token加入input，以此来融合side information
    3. 使用SMLMT作为self-supervise的训练方式
    4. details: 对于N-way K-shot C-query，采样时对每个类采样K个样本，然后对每个样本采样C个作为query；SMLMT并非是一个直接的分类问题，而是同一个类的support set和query set使用相同token mask；只希望正样本的距离比负样本的距离大就可以，所以推理时support set要包含真实类
  - code: https://github.com/mhw32/prototransformer-public

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
  - code: https://github.com/wuzhiye7/Induction-Network-on-FewRel；https://github.com/zhongyuchen/few-shot-text-classification



## Chinese datasets

- 今日头条中文新闻（短文本）分类数据集 ：https://github.com/fateleak/toutiao-text-classfication-dataset
  - 数据规模：共38万条，分布于15个分类中
  - 采集时间：2018年05月
  - 以0.7 0.15 0.15做分割
  
- 清华新闻分类语料：
  - 根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成。
  - 数据量：74万篇新闻文档（2.19 GB）
  - 小数据实验可以筛选类别：体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
  - http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5
  - rnn和cnn实验：https://github.com/gaussic/text-classification-cnn-rnn
  
- 中科大新闻分类语料库：http://www.nlpir.org/?action-viewnews-itemid-145