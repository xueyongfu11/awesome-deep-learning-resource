<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [english datasts](#english-datasts)
- [chinese datasets](#chinese-datasets)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://github.com/BDBC-KG-NLP/QA-Survey-CN
- https://github.com/sogou/SogouMRCToolkit
- https://github.com/PaddlePaddle/RocketQA

- IDEA UniMRC
  - https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-DeBERTa-v2-1.4B-Chinese


# Dataset

## english datasts

### 抽取式阅读理解
- https://rajpurkar.github.io/SQuAD-explorer/
  - 源自维基百科文章的问答对。在SQuAD中，问题的正确答案可以是 给定文本中标记的任何部分。由于问题和答案是由人通过众包的方式产生的，因此它比其他一些问答数据集更加多样化
  - SQuAD 1.1 包含536篇文章中的 107785 个问答对
  - SQuAD 2.0 是最新版本，在原来基础上增加对抗性问题的同时，也新增了一项任务：判断一个问题能否根据提供的阅读文本作答

- NewsQA
  - https://www.microsoft.com/en-us/research/project/newsqa-dataset/
  - https://huggingface.co/datasets/newsqa
  - https://github.com/Maluuba/newsqa

- BiPaR
  - https://github.com/sharejing/BiPaR

- SubQA
  - https://huggingface.co/datasets/subjqa

### 其他QA数据集

- https://microsoft.github.io/msmarco/
  - 其背后的团队声称这是目前这一类别中最有用的数据集，因为这个数据集是基于匿名的真实数据构建的
  - 和 SQuAD 不一样，SQuAD 所有的问题都是由编辑产生的。MS MARCO 中所有的问题，都是在 Bing 搜索引擎中抽取 用户的查询 和 真实网页文章的片段 组成。一些回答甚至是“生成的”。所以这个数据集可以用在开发 生成式问答系统
  - 不太适合抽取式阅读理解任务

- CNN/Daily Mail
  - 可以看出数据集规模是很大的，这是由于此数据集是半自动化构建的。数据集文章的来源是CNN和Daily Mail（两个英文刊物）中的文章，而每篇文章都有摘要，问题和答案是完形填空的形式——问题是被遮住一些实体后的摘要，答案是基于文章内容对被遮挡住实体的填充
  - 10w和20w

- CoQA
  - [介绍](https://zhuanlan.zhihu.com/p/62475075)
  - 不是完全的抽取式数据集，答案存在整理
  - 多轮对话数据集


## chinese datasets


- https://github.com/InsaneLife/ChineseNLPCorpus
- [NLP机器阅读理解：四大任务及相应数据集](https://mp.weixin.qq.com/s/KXq0d0xXuGVDOzlNds0jgw)
- [中文机器阅读理解（片段抽取）数据集整理](https://mp.weixin.qq.com/s/dYDalqYB4JRiTbMgHDkYKA)

- https://github.com/baidu/DuReader
  - DuReader
    - [paper 链接](https://www.aclweb.org/anthology/W18-2605.pdf) | [数据集链接](https://ai.baidu.com/broad/introduction?dataset=dureader)
    - 30万问题 140万文档 66万答案  多文档   非抽取式，答案由人工生成
  - DuReade_robust
    - [数据集链接](https://github.com/PaddlePaddle/Research/tree/master/NLP/DuReader-Robust-BASELINE)
    - 2.2万问题，单篇章、抽取式阅读理解数据集
- CMRC 2018 
  - 2万问题 篇章片段抽取型阅读理解 哈工大讯飞联合实验室
  - [paper链接](https://www.aclweb.org/anthology/D19-1600.pdf) | [数据集链接](https://github.com/ymcui/cmrc2018)
- 观点型阅读理解数据集  百度
  - [数据集链接](https://aistudio.baidu.com/aistudio/competition/detail/49/?isFromLUGE=TRUE)
- 抽取式数据集 百度
  - [数据集链接](https://aistudio.baidu.com/aistudio/competition/detail/49/?isFromLUGE=TRUE)
- 中文完形填空数据集
  - https://github.com/ymcui/Chinese-RC-Dataset
  - https://link.zhihu.com/?target=https%3A//github.com/ymcui/Chinese-RC-Dataset
