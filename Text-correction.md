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

## englist
- [CoNLL-2014 Shared Task: Grammatical Error Correction](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- https://www.cl.cam.ac.uk/research/nl/bea2019st/#data

