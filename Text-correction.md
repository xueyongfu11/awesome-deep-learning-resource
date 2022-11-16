<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
  - [chinese spelling correction](#chinese-spelling-correction)
  - [hanzi similar](#hanzi-similar)
- [Paper](#paper)
  - [Grammatical Error Correction](#grammatical-error-correction)
  - [chinese spelling correction](#chinese-spelling-correction-1)
  - [Post-OCR text correction](#post-ocr-text-correction)
  - [Post ASR Error correction](#post-asr-error-correction)
- [Competition](#competition)
- [Datasets](#datasets)
  - [chinese](#chinese)
  - [englist](#englist)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

## other

- https://github.com/wdimmy/Automatic-Corpus-Generation
  - 中文数据合成:通过模糊图像并进行OCR识别来构建混淆集；通过ASR来构建混淆集
- https://github.com/awasthiabhijeet/PIE/tree/master/errorify
  - 英文数据合成
  
- https://github.com/nghuyong/text-correction-papers
- https://github.com/Chunngai/gec-papers
- https://github.com/destwang/CTCResources
  
- https://github.com/shibing624/pycorrector
- https://github.com/iqiyi/FASPell
- https://github.com/gitabtion/BertBasedCorrectionModels
  - 基于bert的文本纠错模型
- https://github.com/yuantiku/fairseq-gec
- https://github.com/1250658183/Chinese-Error-Checking
- https://github.com/HillZhang1999/MuCGEC

- https://github.com/letiantian/Pinyin2Hanzi
  - 拼音转汉字

## chinese spelling correction
- https://github.com/liushulinle/PLOME

## hanzi similar
- https://github.com/houbb/nlp-hanzi-similar
- https://github.com/houbb/word-checker/blob/master/README_ZH.md
- https://github.com/Inigo-numero1/zh-spelling-mistakes-dictionaries
- https://github.com/TmengT/WordSimilarity
  
- 各种词库：https://github.com/fighting41love/funNLP
- THUOCL：清华大学开放中文词库:http://thuocl.thunlp.org/

# Paper

##  Grammatical Error Correction
- GECToR – Grammatical Error Correction: Tag, Not Rewrite
  - 2020
  - 阅读笔记：
    1.使用序列标注的方法，通过一种转化映射标签集将错误文本标注上标签，根据标签可以转换出正确文本
    2.在推理方面，相较于NMT的方法在速度上有很大提升
  - code：https://github.com/grammarly/gector


## chinese spelling correction

- SDCL: Self-Distillation Contrastive Learning for Chinese Spell Checking
  - 2022  AACL
  - 阅读笔记：
    1. 提出一个自蒸馏对比学习的中文拼写纠错模型
    2. 将包含错误token的文本输入bert，将正确的文本输入bert输入另外一个bert，并作为教师模型
    3. 将错误token的hidden_state和相对应的正确的token的hidden_state作为正样本，负样本从同一个batch中获取。目的是通过对比学习来拉近confusion set之间的距离
    4. 学生模型的hidden_state和对应token的word embedding进行点积的计算之后进行softmax，计算和真实token的交叉熵
    5. 教师模型部分的交叉熵损失计算方法没有看懂

- General and Domain Adaptive Chinese Spelling Check with Error Consistent Pretraining
  - 2022
  - 笔记：
    1. 根据错误一致性来构建预训练数据集，输入的特征包含拼音，字形等，预训练任务包含正确字词预测和通过字图预测所对应字标签
    2. 模型使用基于token分类的类ner模型，tag类别使用常见的中文字，增加新的不纠正标签，对不在tag中的token打上非纠错标签
    3. 通过引入领域词典的方式来提供自适应能力，该方法鼓励模型解码时更多的领域词典中的字
  - code；https://github.com/Aopolin-Lv/ECSpell

- MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction
  - 2022 ACL
  - 阅读笔记：
    1. 使用多任务学习（即检测loss和纠正loss）的方式建模中文拼写纠错任务
    2. 使用同一个bert模型对source文本和target文本进行表征，使用binary loss计算检测网络的损失
    3. 将检测网络的最后一层的输出，融合到纠错网络的最后一层输出
    4. 纠错网络最后的投影层（一层全连接）的权重参数使用输入的word embedding进行初始化

- Visual and Phonological Feature Enhanced Siamese BERT for Chinese Spelling Error Correction
  - 2022 
  - 阅读笔记：
    1. 使用一个融合了字形和拼音的bert模型和vanilla bert的双胞胎网络，来分别对形似和音似的字纠错，以及和形似音似无关的字纠错
    2. 字形的embedding使用node2vec模型训练得到，通过对字进行组成分解，包含相同组分的字直接链接起来，计算得到字与字之间的链接权重
    3. FS-BERT和vanilla BERT的输出用一个标量point加权起来，point通过一个sigmoid函数得到

- Correcting Chinese Spelling Errors with Phonetic Pre-training
  - 2021 ACL
  - 阅读笔记：
    1. 中文拼写错误方面，基于融入拼音的预训练和融入拼音的错误检测和错误纠正等方法
    2. 预训练：将传统的[MASK]替换成拼音或者基于拼音的混淆词，以及传统给的mask策略
    3. 错误检测：基于序列标注的方法，将word+pinyin作为输入
    4. 根据错误检测的概率数据将word embedding和pinyin embedding加权，作为错误纠正模块的输入
  - code：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc

- Spelling Error Correction with Soft-Masked BERT
  - 2020
  - 阅读笔记：
    1. 原生的bert的检错能力很差，所有soft-masked bert增加了一个错误检测模块
    2. 根据错误检测模块的数据概率将word embedding和mask embedding加权，作为纠正模块的输入
  - code:

## Post-OCR text correction

- https://github.com/tiantian91091317/OCR-Corrector
- https://github.com/shrutirij/ocr-post-correction
- https://sites.google.com/view/icdar2019-postcorrectionocr
  - ICDAR2019 OCR识别后文本纠错


## Post ASR Error correction
- [ASR文本纠错近期论文汇总](https://zhuanlan.zhihu.com/p/424852619)
  

# Competition
- 第三届中国AI+创新创业大赛-自然语言处理技术创新大赛-中文文本纠错比赛
  - https://github.com/destwang/CTC2021
  - https://mp.weixin.qq.com/s/uASKfgiyhZC4WNMenX60lQ

- CCL 2022 汉语学习者文本纠错评测
  - https://github.com/blcuicall/CCL2022-CLTC
  - https://github.com/HillZhang1999/MuCGEC


# Datasets

## chinese
- https://github.com/HillZhang1999/MuCGEC
  - CCL2022文本纠错任务数据集
- https://github.com/destwang/CTCResources#datasets
- 汉语拆字字典
  - https://github.com/kfcd/chaizi
  - https://link.zhihu.com/?target=https%3A//github.com/kfcd/chaizi 

## englist
- [CoNLL-2014 Shared Task: Grammatical Error Correction](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- https://www.cl.cam.ac.uk/research/nl/bea2019st/#data

