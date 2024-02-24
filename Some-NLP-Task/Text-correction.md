<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
  - [other](#other)
  - [chinese spelling correction](#chinese-spelling-correction)
  - [grammatical error correction](#grammatical-error-correction)
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

## grammatical error correction
- https://modelscope.cn/models/damo/nlp_bart_text-error-correction_chinese/summary

## hanzi similar
- https://github.com/houbb/nlp-hanzi-similar
- https://github.com/houbb/word-checker/blob/master/README_ZH.md
- https://github.com/Inigo-numero1/zh-spelling-mistakes-dictionaries
- https://github.com/TmengT/WordSimilarity
  
- 各种词库：https://github.com/fighting41love/funNLP
- THUOCL：清华大学开放中文词库:http://thuocl.thunlp.org/

# Paper

##  Grammatical Error Correction

- From Spelling to Grammar: A New Framework for Chinese Grammatical Error Correction
  - 2022
  - 阅读笔记：
    1. 提出了一种两阶段的中文语法纠错模型，先进行拼写纠错再进行语法纠错
    2. 拼写纠错：通过频率阈值分割找出可能错误的位置，基于bert模型得到候选，根据混淆集得到高precision的拼写纠错结果
    3. 使用BART模型，并融入词的POS以及从同义词词林种获取到的词的语义信息。
    4. 引入一种POS辅助任务

- FOCUS IS WHAT YOU NEED FOR CHINESE GRAMMATICAL ERROR CORRECTION
  - 2022
  - 阅读笔记:
    1. 探索了multi-reference对模型不能带来正收益
    2. 使用编辑距离、jaccard距离等方法从mult-reference种来构建出较少不确定性的数据

- Mining Error Templates for Grammatical Error Correction
  - 2022
  - 阅读笔记：
    1. 提出一种基于错误模板挖掘的终于语法纠错
    2. 基于一种search pattern（如A...B是语法错误吗）来从互联网获取大量错误模板，如原因是...引起的，大约...左右
    3. 基于获取的模板有三种纠正行为，提出使用基于GPT-2的语言困惑都的方法来得出每个模板的最佳纠错行为
    4. 在learner text上没有效果，但在native text上有4%+的绝对提升
  - code：https://github.com/HillZhang1999/gec_error_template

- Chinese grammatical error correction based on knowledge distillation
  - 2022
  - 阅读笔记;
    1. 提出使用模型蒸馏的中文语法纠错模型
    2. 该paper没什么创新

- Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction
  - 2021 ACL
  - 阅读笔记:
    1. 提出了一种非自回归的端到端的中文语法纠错模型
    2. 输入层：对于len(X)《 len(Y),则在X尾部添加mask，表示字符的插入; 相反则在Y的尾部添加pad，表示字符的删除；长度相等则不做改变。目的是为了使用CRF进行end2end的解码。
    3. 使用bert模型获取hidden state，使用全连接层，输出维度是词表的大小。使用加入了focal loss的NLL loss作为模型参数学习的任务损失，使用CRF loss作为全局特征依赖的任务损失
    4. 由于词表较大，CRF中的转移矩阵使用了矩阵分解方法；在inference时，使用top-k node的维特比解码算法来加速计算
  - code: https://github.com/lipiji/TtT

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
    1. 提出一种基于对比学习的中文拼写纠错模型
    2. 使用两个共享weight的bert模型，teacher model输入correct sentence，另外一个输入corrupted sentence
    3. student model的输出H与word-embedding点积，计算交叉熵损失；corrupted word embedding和correct word embedding作为正例，同一个batch作为负例，计算对比损失；teacher model输入corrupted sentence，与loss1类型，输出H与word-embedding点积，计算交叉熵损失。

- uChecker: Masked Pretrained Language Models as Unsupervised Chinese Spelling Checkers
  - 2022  COLING
  - 阅读笔记：
    1. 提出一种基于MLM的无监督预训练的中文文本纠错模型
    2. 以BERT为基础模型进行fine-training，在整个过程种freeze bert模型参数
    3. 提出：1）无监督的拼写错误检测以及相应的错误纠正2）自监督的拼写错误检测：理论是correct token的embedding和其隐状态输出向量的relation > error token的embedding和其隐状态输出向量的embedding。因为token的embedding和隐状态的向量属于不同的向量空间，因此使用了一个interation model来建模
  - code: 

- A Chinese Spelling Check Framework Based on Reverse Contrastive Learning
  - 2022
  - 阅读笔记：
    1. 提出一种反对比学习的中文拼写纠错模型
  2. 只关注负样本的构建。负样本是：一个batch中的同音异形词作为负样本；
  一个batch中的形近字（在混淆集中）作为负样本
  3. 反对比学习的好处是可以对容易混淆的样本做区分

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

- CRASpell: A Contextual Typo Robust Approach to Improve Chinese Spelling Correction
  - 2022 ACL
  - 阅读笔记：
    1. 提出了一种上下文多错误鲁棒的中文拼写纠错模型
    2. 为了解决过纠问题，再mlm的基础上加入copy概率，来鼓励模型使用对应的输入token，具体是预测一个概率值作为拷贝概率
    3. 为了解决上下文错误对当前token纠错的影响，引入上下文错误并得到当前token的预测分布，使用KL散度来计算引入上下文错误和不引入上下文错误的预测分布差异
  - code: https://github.com/liushulinle/CRASpell

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
- https://github.com/blcuicall/YACLC
  - 智源指数平台：汉语学习者文本多维标注数据集YACLC
- https://github.com/HillZhang1999/MuCGEC
  - CCL2022文本纠错任务数据集
- https://github.com/destwang/CTCResources#datasets
- 汉语拆字字典
  - https://github.com/kfcd/chaizi
  - https://link.zhihu.com/?target=https%3A//github.com/kfcd/chaizi 

## englist
- [CoNLL-2014 Shared Task: Grammatical Error Correction](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
- https://www.cl.cam.ac.uk/research/nl/bea2019st/#data

