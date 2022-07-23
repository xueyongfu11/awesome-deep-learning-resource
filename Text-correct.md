<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Paper](#paper)
  - [chinese spelling correction](#chinese-spelling-correction)
  - [Post-OCR text correction](#post-ocr-text-correction)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo
- https://github.com/wdimmy/Automatic-Corpus-Generation
  - 数据合成
- https://github.com/awasthiabhijeet/PIE/tree/master/errorify
  - 数据合成
  
- https://github.com/nghuyong/text-correction-papers
- https://github.com/Chunngai/gec-papers
  
- https://github.com/shibing624/pycorrector
- https://github.com/iqiyi/FASPell
- https://github.com/HillZhang1999/MuCGEC
  - CCL2022文本纠错任务数据集
- https://github.com/blcuicall/CCL2022-CLTC
  - CCL 2022 汉语学习者文本纠错评测，官方任务发布网站


# Paper

- GECToR – Grammatical Error Correction: Tag, Not Rewrite
  - 2020
  - 阅读笔记：
    1.使用序列标注的方法，通过一种转化映射标签集将错误文本标注上标签，根据标签可以转换出正确文本
    2.在推理方面，相较于NMT的方法在速度上有很大提升
  - code：https://github.com/grammarly/gector

## chinese spelling correction
- Correcting Chinese Spelling Errors with Phonetic Pre-training
  - 2021 ACL
  - 阅读笔记：
    1.中文拼写错误方面，基于融入拼音的预训练和融入拼音的错误检测和错误纠正等方法
    2.预训练：将传统的[MASK]替换成拼音或者基于拼音的混淆词，以及传统给的mask策略
    3.错误检测：基于序列标注的方法，将word+pinyin作为输入
    4.根据错误检测的概率数据将word embedding和pinyin embedding加权，作为错误纠正模块的输入
  - code：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_correction/ernie-csc


## Post-OCR text correction

- https://github.com/tiantian91091317/OCR-Corrector
- https://github.com/shrutirij/ocr-post-correction
- https://sites.google.com/view/icdar2019-postcorrectionocr
  - ICDAR2019 OCR识别后文本纠错
- https://github.com/skishore/makemeahanzi






