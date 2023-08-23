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

- https://github.com/SCHENLIU/longformer-chinese

- T5
  - https://huggingface.co/IDEA-CCNL/Randeng-T5-784M
    - 基于mT5-large，训练了它的中文版
  - https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese
    - 在Randeng-T5-784M的基础上，收集了100个左右的中文数据集，进行Text2Text统一范式的有监督任务预训练
  - https://huggingface.co/IDEA-CCNL/Randeng-T5-77M
    - 中文版的mT5-small
  - https://huggingface.co/IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese
    - 在Randeng-T5-77M的基础上，收集了100个左右的中文数据集，进行Text2Text统一范式的有监督任务预训练
  
  - https://huggingface.co/lemon234071/t5-base-Chinese
    - A mt5-base model that the vocab and word embedding are truncated, only Chinese and English characters are retained

  - https://huggingface.co/IDEA-CCNL/Randeng-T5-Char-700M-Chinese
    - 中文版的T5-large，采用了BertTokenizer和中文字级别词典
  - https://huggingface.co/IDEA-CCNL/Randeng-T5-Char-700M-MultiTask-Chinese
    - 在Randeng-T5-Char-700M的基础上，收集了100个左右的中文数据集，进行Text2Text统一范式的有监督任务预训练
  - https://huggingface.co/Langboat/mengzi-t5-base

- deberta
  - https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E4%BA%8C%E9%83%8E%E7%A5%9E%E7%B3%BB%E5%88%97/index.html

- https://huggingface.co/Langboat/mengzi-bert-base
  - 中文模型，Pretrained model on 300G Chinese corpus. Masked language modeling(MLM), part-of-speech(POS) tagging and sentence order prediction(SOP) are used as training task.


- 哈工大开源预训练语言模型
  - https://huggingface.co/hfl
  - https://github.com/ymcui/Chinese-BERT-wwm

- https://github.com/CLUEbenchmark/CLUEPretrainedModels
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
- [tokenizer：BPE、WordPiece、Unigram LM、SentencePiece相关blog](https://zhuanlan.zhihu.com/p/620508648)


# Paper

### 2023

- DEBERTAV3: IMPROVING DEBERTA USING ELECTRA-STYLE PRE-TRAINING WITH GRADIENTDISENTANGLED EMBEDDING SHARING
  - ICLR  [[code]](https://github.com/microsoft/DeBERTa)
  - <details>
    <summary>阅读笔记: </summary>
    1. 模型延续了deberta，不同的是，deberta-v3使用了ELECTRA的RTD任务  <br>
    2. ELECTRA训练时，生成器和判别器的embedding是共享的，由于不同的任务会引起类似拔河竞争。deberta-v3实验证明相比各自拥有embedding，embedding共享是有效的  <br>
    3. 训练的方法与NES相同，即生成器和判别器分别计算loss和反向传播  <br>
    4. 先生成判别器的输入，然后再用MLM loss更新生成器的embedding权重（由于共享，此时判别器的embedding也相应更新），计算判别器的RTD loss，不在对生成器也就是判别器进行更新，而是更新E-delta，将E-delta+E从而得到判别器的embedding，同时也是生成器的新的embedding
    </details>


### 2021 

- DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION
  - ICLR  [[code]](https://github.com/microsoft/DeBERTa)
  - <details>
    <summary>阅读笔记: </summary>
    1. 相比bert，deberta的每个token使用content向量和position向量来表示，计算attention矩阵时，计算content2position,content2content,position2content,position2position等4个score的加和，因为这里的position是相对位置，实现时去掉了最后一个score  <br>
    2. 使用了增强mask解码器，不同于bert把绝对位置加载输入层，enhanced mask decoder把绝对位置加在模型的最后一层，softmax之前  <br>
    3. 下游finetune时，使用了对抗训练方法，不同传统的直接对word embeddding加上干扰项，deberta是在word embedding进行LN之后的输出加上干扰项，相比大模型来说，这种方法更加稳定  <br>
    </details>

- NEZHA: NEURAL CONTEXTUALIZED REPRESENTATION FOR CHINESE LANGUAGE UNDERSTANDING
  - 2021 
  - 阅读笔记：
    1. 提出了一个中文预训练语言模型Nezha
    2. 使用函数式相对位置编码（与transformer-XL中的相同位置编码的计算方式基本相似）；使用全词mask机制；使用了混合精度训练；使用了LAMB优化器
  - code: https://github.com/huawei-noah/Pretrained-Language-Model

- Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese
  - 2021
  - 阅读笔记：
    1. 提出了一个轻量级并强大的预训练语言模型mengzi
    2. 预训练的setup：探索性的数据清洗，使用roberta来对模型进行初始化，使用LAMB优化器，并使用FP16和deepspeed来加速训练
    3. 预训练：使用了POS NER MLM等预训练任务；使用了SOP预训练任务；使用了动态梯度纠正
    4. 微调: 知识蒸馏、迁移学习、choice smoothing、对抗学习、数据增强等手段
  - code：https://github.com/Langboat/Mengzi

### 2020

- ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS
  - ICLR  [[code]](https://github.com/google-research/ALBERT)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一个轻量级的预训练语言模型（无蒸馏操作）  <br>
    2. embedding矩阵分解，跨层参数共享、句子顺序预测（采样方法同bert，即从同一个文档中采样两个两个的句子片段，将句子顺序颠倒）  <br>
    </details>

- TinyBERT: Distilling BERT for Natural Language Understanding
  - EMNLP  [[code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
  - <details>
    <summary>阅读笔记: </summary>
    1. 提出了一个两阶段包含语训练蒸馏和下游任务蒸馏的语言模型TinyBert  <br>
    2. 预训练阶段蒸馏：只进行transformer layer蒸馏，具体包括attention based distillation和hidden state based distillation  <br>
    3. 下游任务微调阶段蒸馏：在一个已经微调好的bert的条件下，先使用transfomer layer蒸馏、embedding layer蒸馏，在使用prediction layer蒸馏（注意无需要再次输出数据）  <br>
    4. 4-layers的tinybert可以达到9倍的推理速度提升，精度略有下降
    </details>

### 2019 

- RoBERTa: A Robustly Optimized BERT Pretraining Approach
  - EMNLP  [[code]](https://github.com/facebookresearch/fairseq)
  - <details>
    <summary>阅读笔记: </summary>
    1. 相比BERT，使用更多的训练数据、训练时间更长、batch更大；去掉了NSP预训练任务；使用了更长的序列；使用了动态mask机制  <br>
    </details>

- XLNet: Generalized Autoregressive Pretraining for Language Understanding
  - [[code]](https://github.com/zihangdai/xlnet)
  - <details>
    <summary>阅读笔记: </summary>
    1. 一种广义自回归预训练语言模型，通过一种文本随机排列的方法，从而建模双向的信息  <br>
    2. 实现时并非对序列进行排列，可以使用attention mask机制来实现。具体是使用content流和query流，content流类似传统的attention计算，query流是以只包含位置信息的作为query，不能包含当前的token信息，从而实现在不同的阶段，token的可见性。  <br>
    3. 引用了transformer-xl的片段循环机制和相对位置编码机制  <br>
    </details>

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