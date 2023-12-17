<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [情感分析](#%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90)
- [ABSA](#absa)
  - [综述](#%E7%BB%BC%E8%BF%B0)
  - [extraction ABSA](#extraction-absa)
  - [category ABSA](#category-absa)
  - [Emotion cause extraction](#emotion-cause-extraction)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



# 情感分析

- [基于情感词典的情感分析](https://blog.csdn.net/lom9357bye/article/details/79058946)



# ABSA

## 综述
- [香港中文大学最新《基于Aspect的情感分析》综述论文，涵盖近200篇文献阐述ABSA方法体系](https://mp.weixin.qq.com/s/uKxr4NguT2MmvTArsAIMig)
- 

## extraction ABSA

- [如何在基于Aspect的情感分析中结合BERT和语法信息](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247493810&idx=1&sn=7739e319e64f7895d9161d6e23ae1f6d&chksm=eb53fc21dc24753706a1d2bfd0b315c53061026cdf4155981a5daa7d177fe6aed60ab450e916&mpshare=1&scene=24&srcid=090698uCVENW4IpifG81cBmE&sharer_sharetime=1599397136536&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A9IntLWPEXsq0HNjyUksYz8%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
  - AE:使用roberta模型，并结合了词法信息和句法信息。句法信息：一个token相关的所有关系
  - SC：探索如何更好结合sentence和aspect。不同与CDM、CDW，paper使用两个单词在句法树中的距离。

- [复旦邱锡鹏Lab提出：一个统一的面向基于Aspect的所有情感分析子任务的生成式方法](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650427046&idx=4&sn=c135ad5a45b76dae3a6b8403eaabc4c5&chksm=becdc8fc89ba41ea5d75e54a5b1afd2ecf41215cbbaff66cc1cea476d7ecaae497b447f69803&mpshare=1&scene=24&srcid=1106Y9QVHi1qw1ohlb9AtISQ&sharer_sharetime=1636211308160&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AwkeNCr7h8XmptVhT326XcE%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
  - A Uniﬁed Generative Framework for Aspect-Based Sentiment Analysis
  - 基于BART模型，encoder输出 = embedding + last layer output
  - decoder：使用pointer network，输出index是encoder pos和三个情感极性类别

- [EMNLP21' | 细粒度情感分析新突破 —— 通过有监督对比学习方法学习隐式情感](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247507097&idx=2&sn=144605d4e3964fe915889f099553e2cc&chksm=eb53c80adc24411cb862ac6a933708d0788b6fdeea043cccad7c2842b7560ad064c630154a73&mpshare=1&scene=24&srcid=1119kqzAxB0adw5BcfqOMAbS&sharer_sharetime=1637316336524&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A5MevBiwgD2ot8vt4yljuLk%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
  - 文章提出，隐形情感在开源数据集中大量存在，需要模型学习相应尝试知识，考虑预训练
  - 构建大规模多方面情感分析数据集进行预训练：有监督对比学习，评论重建DAE，方面词预测
  - 微调：将句子级表示（CLS）和方面词（经过average pooling）拼接之后进行极性分类，
  较传统的每个方面词和评论分别拼接判断，提高了效率

- [细粒度情感分析在保险行业的应用](https://zhuanlan.zhihu.com/p/151216832)
  - bert pair模型进行ABSA建模，将aspect转成如'XXX性价比怎么样'
  - backbone使用KBERT，基于知识图谱的bert
  
- [论文浅尝 - ICLR2020 | 知道什么、如何以及为什么：基于方面的情感分析的近乎完整的解决方案](https://mp.weixin.qq.com/s?__biz=MzU2NjAxNDYwMg==&mid=2247488959&idx=1&sn=1a3261c701b0a709e00e2d90f7a7b06b&chksm=fcb3b25acbc43b4c9486b6846d4aed0004652067f9b54fc658ba365e9a01f1f265d1089ddc54&mpshare=1&scene=24&srcid=0802FCQbNy4o3a6pXrSrXsYp&sharer_sharetime=1596383621820&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=Ay5eCQzijmA3%2F1F28TyxT8M%3D&pass_ticket=%2Fiuk0Yfg7CrYxacY%2F347pmZcCE1UxpnHXEwngLMc%2BDJTSlAVtev8q4cY8e9W%2Bxmv&wx_header=0#rd)
  - 两阶段：先抽取提及的所有实体，情感提及词和观点词语
  - 将实体和提及词两两匹配进行分类
  - 
  
## category ABSA

- [【论文阅读】Joint Aspect and Polarity Classification for Aspect-based Sentiment Analysis with End-to-End](https://blog.csdn.net/BeforeEasy/article/details/104184051)
  - 使用cnn进行建模，对一个样本，模型输出维度：类别数*4，pos/neg/neu/none，none表示无该类别
  
## Emotion cause extraction

- [【论文解读】情感-原因关系挖掘 —— ACL2019杰出论文](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247488466&idx=2&sn=7a767374bbb46183053416ee1d25c53d&chksm=eb500741dc278e577595004da2eacead580de53ad789ad1774e06c67c601b5dbe672bf0c917d&scene=0&xtrack=1&exportkey=A9xbtRbhCjX%2FJZ7OWRFrWOM%3D&pass_ticket=H0sgsFf0Diewumyma%2FRYfqkoyYzoismRNGo4T2CNs2J00r2R%2FjAgF5ufzYIdfDws&wx_header=0#rd)
  - 任务内容：提取文档中所有的情感-原因句对
  - step1：使用independent多任务学习方式：使用多层级的LSTM网络，先对句子encoding，然后在对句子序列encoding。使用interactive多任务学习方式，
  两种方法，一种是将情感句子embedding和情感句子识别结果融合，作为原因LSTM网络的输入
  - step2：对所有的情感-原因句对进行分类
  
- [给定情感极性，输出支撑情感的原因-Kaggle Tweet Sentiment Extraction 第七名复盘](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650415097&idx=2&sn=a95233663bae8056458f7c549a51b7e6&chksm=becd99a389ba10b58126bc82770ad8b766a5539dca2524222af8350fd81311abdd89d6885827&mpshare=1&scene=24&srcid=0724iXVGG20WJx5ZYdxHZH6i&sharer_sharetime=1595603342254&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A1XEOJ0Qtp9%2B0Nb8V1Syd5o%3D&pass_ticket=H0sgsFf0Diewumyma%2FRYfqkoyYzoismRNGo4T2CNs2J00r2R%2FjAgF5ufzYIdfDws&wx_header=0#rd)
  - 建模为机器阅读理解任务