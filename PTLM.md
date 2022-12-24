<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
  - [预训练语言模型](#%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)
  - [蒸馏](#%E8%92%B8%E9%A6%8F)
  - [长文本PTLM](#%E9%95%BF%E6%96%87%E6%9C%ACptlm)
  - [tokenizer](#tokenizer)
- [Paper](#paper)
- [词向量](#%E8%AF%8D%E5%90%91%E9%87%8F)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# Repo
- Transformer升级之路：2、博采众长的旋转式位置编码 https://github.com/ZhuiyiTechnology/roformer

## 预训练语言模型
- https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models
  - 收集了一些高质量 中文 预训练语言模型
- https://github.com/dbiir/UER-py
- https://github.com/stanfordnlp/GloVe
- 华为预训练语言模型 https://github.com/huawei-noah/Pretrained-Language-Model
- bert相关 https://github.com/Jiakui/awesome-bert
- https://github.com/brightmart/roberta_zh
- https://github.com/sinovation/ZEN
- https://github.com/brightmart/albert_zh
- https://github.com/zihangdai/xlnet
- https://github.com/bojone/t5_in_bert4keras
- https://github.com/utterworks/fast-bert
- https://github.com/idiap/fast-transformers
- https://github.com/CLUEbenchmark/LightLM
- https://github.com/google-research/electra
- https://github.com/yizhen20133868/BERT-related-papers
- 

## 蒸馏
- https://github.com/kevinmtian/distill-bert
- https://github.com/wangbq18/distillation_model_keras_bert
- https://github.com/qiangsiwei/bert_distill
- https://github.com/PaddlePaddle/ERNIE
- https://github.com/YunwenTechnology/Unilm
- https://github.com/google-research/text-to-text-transfer-transformer
- https://github.com/TsinghuaAI/CPM-1-Generate
- https://github.com/sinovation/ZEN
- https://github.com/dbiir/UER-py
- https://github.com/ymcui/Chinese-ELECTRA
- https://github.com/zhongerqiandan/pretrained-unilm-Chinese
- https://github.com/ymcui/Chinese-BERT-wwm
- https://github.com/ZhuiyiTechnology/WoBERT
- https://github.com/NVIDIA/Megatron-LM

## 长文本PTLM
- https://github.com/SCHENLIU/longformer-chinese
- https://github.com/allenai/longformer
- https://github.com/Sleepychord/CogLTX
- https://github.com/Langboat/Mengzi

## tokenizer
- 基于神经网络的分词模型 https://github.com/google/sentencepiece

# Paper

- TinyBERT: Distilling BERT for Natural Language Understanding
  - year: 2020 EMNLP
  - 阅读笔记：
    1. 提出了一个两阶段包含语训练蒸馏和下游任务蒸馏的语言模型TinyBert
    2. 预训练阶段蒸馏：只进行transformer layer蒸馏，具体包括attention based distillation和hidden state based distillation
    3. 下游任务微调阶段蒸馏：在一个已经微调好的bert的条件下，先使用transfomer layer蒸馏、embedding layer蒸馏，在使用prediction layer蒸馏（注意无需要再次输出数据）
    4. 4-layers的tinybert可以达到9倍的推理速度提升，精度略有下降
  - code: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT

- RoBERTa: A Robustly Optimized BERT Pretraining Approach
  - year: 2019
  - 阅读笔记：
    1. 相比BERT，使用更多的训练数据、训练时间更长、batch更大；去掉了NSP预训练任务；使用了更长的序列；使用了动态mask机制
  - code: https://github.com/facebookresearch/fairseq

- XLNet: Generalized Autoregressive Pretraining for Language Understanding
  - year: 2019 
  - 阅读笔记: 
    1. 一种广义自回归预训练语言模型，通过一种文本随机排列的方法，从而建模双向的信息
    2. 实现时并非对序列进行排列，可以使用attention mask机制来实现。具体是使用content流和query流，content流类似传统的attention计算，query流是以只包含位置信息的作为query，不能包含当前的token信息，从而实现在不同的阶段，token的可见性。
    3. 引用了transformer-xl的片段循环机制和相对位置编码机制
  - code: https://github.com/zihangdai/xlnet

# 词向量
- 100+的中文词向量 https://github.com/Embedding/Chinese-Word-Vectors
- 词向量相关paper，resource，dataset https://github.com/Hironsan/awesome-embedding-models
- ngrams词向量模型 https://github.com/zhezhaoa/ngram2vec
- https://github.com/facebookresearch/fastText
- https://github.com/danielfrg/word2vec.git  
  - Python interface to Google word2vec  
- https://github.com/stanfordnlp/GloVe.git  
  - GloVe model for distributed word representation  
- https://github.com/facebookresearch/fastText.git  
  - Library for fast text representation and classification.  
- https://github.com/zlsdu/Word-Embedding.git  
  - Word2vec, Fasttext, Glove, Elmo, Bert, Flair pre-train Word Embedding  
- https://github.com/Rokid/ELMo-chinese.git  
  - Deep contextualized word representations 中文 汉语  
- https://github.com/Hironsan/awesome-embedding-models.git  
  - A curated list of awesome embedding models tutorials, projects and communities
- https://github.com/kmario23/KenLM-training.git  
  - Training an n-gram based Language Model using KenLM toolkit for Deep Speech 2
- https://github.com/Embedding/Chinese-Word-Vectors.git  
  - 100+ Chinese Word Vectors 上百种预训练中文词向量 
- https://github.com/zhezhaoa/ngram2vec.git  
  - Four word embedding models implemented in Python. Supporting arbitrary context features
- https://github.com/liuhuanyong/ChineseEmbedding.git  
  - Chinese Embedding collection incling token ,postag ,pinyin,dependency,word embedding.中文自然语言处理向量合集,包括字向量,拼音向量,词向量,词性向量,依存关系向量.共5种类型的向量 
- https://github.com/HIT-SCIR/ELMoForManyLangs.git 