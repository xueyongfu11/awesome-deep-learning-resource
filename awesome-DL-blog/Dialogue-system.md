<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Blog](#blog)
- [Survey](#survey)
- [问答系统方案](#%E9%97%AE%E7%AD%94%E7%B3%BB%E7%BB%9F%E6%96%B9%E6%A1%88)
- [Retrieval-based dialogue system](#retrieval-based-dialogue-system)
- [FAQ](#faq)
- [Open-domain dialogue system](#open-domain-dialogue-system)
- [Dialogue generation](#dialogue-generation)
- [Task-oriented dialogue system](#task-oriented-dialogue-system)
  - [end2end](#end2end)
  - [DST](#dst)
  - [DPL](#dpl)
  - [RASA](#rasa)
- [KBQA](#kbqa)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Blog

- [thu-coai清华AI对话组](https://github.com/thu-coai)
- [DSTC10比赛官网](https://dstc10.dstc.community/home)



# Survey

- [对话系统综述：新进展新前沿](https://zhuanlan.zhihu.com/p/45210996)
- [问答系统和对话系统-KBQA和对话系统综述](https://zhuanlan.zhihu.com/p/93023782)
- [对话系统有哪些最新进展？这17篇EMNLP 2021论文](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247540033&idx=2&sn=f9a9dba9c1e38b5fc58e65541bccc42f&chksm=96ea82c1a19d0bd71aad804f1143e4681c981f4381f60290c9057f3f6de9f685d30fe6bece6e&mpshare=1&scene=24&srcid=1118eA0jEQBFFQQMyaTzf4tB&sharer_sharetime=1637165444494&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AyitK9th8FmGPwxGKZLl%2B%2Bc%3D&pass_ticket=3YSLQZ0%2BFGkSbSLIxeI5ld3daRcSE5x5m%2FqFag47PCWFTeogIXft8nu1uI5rJumG&wx_header=0#rd)

# 问答系统方案

- [小米代文：用户意图复杂多变，如何构造小爱智能问答的关键技术？](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247556727&idx=1&sn=3bc1aa55a9642328586457e1ce99e18c&chksm=fbd7ac1bcca0250dcaaa9ae2e24be20d075c522fc1ecbc652e658508ac5dbf45c83e8c45994a&mpshare=1&scene=24&srcid=1114r9460I3oqevY7egPpJok&sharer_sharetime=1636885599401&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A7xwmOEWC4nE8vencXmyuA0%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)

# Retrieval-based dialogue system

- [基于FAQ的智能问答(一): Elasticsearch的调教](https://zhuanlan.zhihu.com/p/347957917)
- [基于FAQ的智能问答(二): 召回篇](https://zhuanlan.zhihu.com/p/349993294)
- [基于FAQ的智能问答(三): 精排篇](https://zhuanlan.zhihu.com/p/352316559)


- [平安寿险PAI](https://www.zhihu.com/column/PAL-AI)
- [ROCLING 2019 | 基于深度语义匹配，上下文相关的问答系统](https://zhuanlan.zhihu.com/p/111380177)
- [AAAI 2020 | 基于Transformer的对话选择语义匹配模型](https://zhuanlan.zhihu.com/p/259810988)


- [基于预训练模型的检索式对话系统最新研究进展](https://mp.weixin.qq.com/s/vISU6GPHP7q5zmwq3QS01w)
- [检索式对话系统预训练](https://zhuanlan.zhihu.com/p/408272506)
- [多轮检索式对话系统小结](https://zhuanlan.zhihu.com/p/84163773)
- [检索型多轮对话管理](https://zhuanlan.zhihu.com/p/355916328)
  - 百度Multi-view Response Selection for Human-Computer Conversation，
  词粒度+utterance粒度，区别是utterance对每个sentence抽取语义，然后把每个sentence
  的语义用GRU建模utterance完整语义
  - Sequential Matching Network(SMN),response和每个utterance分别匹配，然后
  用CNN，rnn累计匹配信息
  - Deep Attention Matching Network：使用transformer，模型似SMN

# FAQ

- [CCKS 2019 | 融入知识图谱的问答匹配寿险问答系统](https://zhuanlan.zhihu.com/p/89983691)
  - 基于问答库检索式的问答系统的基础上，通过使用知识图谱丰富召回和使用注意力机制方式把知识图谱融入排序模块的方式，大大提升了问答系统的效果
  

# Open-domain dialogue system

- [Facebook刷新开放域问答SOTA：模型训模型！Reader当Teacher！](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650422401&idx=5&sn=844a512d5133bfb049ec1fbcf5a043df&chksm=becdbadb89ba33cd6fea933f41ce00cae977c35a964b6281c6e9bead2b26399c0b7858e0b9b4&mpshare=1&scene=1&srcid=05068cLavSdTgSO1yG4cp3XW&sharer_sharetime=1620287858377&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AwAFmFCrdqNtTs9zQqltafg%3D&pass_ticket=2TdDpB9ddfGOZT98TxfdI0%2BydSrf6vzFEEAdeyMDGI%2FZzpXRDDBwFo%2BQrPLaoqwH&wx_header=0#rd)
- [开放域问答系统 DrQA](https://zhuanlan.zhihu.com/p/77077948)
- [让聊天机器人同你聊得更带劲 - 对话策略学习](https://zhuanlan.zhihu.com/p/29749869)
- [微软小冰-多轮和情感机器人的先行者](https://mp.weixin.qq.com/s?__biz=MzIzMzYwNzY2NQ==&mid=2247485954&idx=1&sn=53c49a5af387cd86ea51a6407818e414&chksm=e882529cdff5db8a6d04891e9e939475409d547798c51f00d9fad6a695fd1d96a080160d0148&scene=21#wechat_redirect)
- [训练双塔检索模型，可以不用query-doc样本了？明星机构联合发文](https://mp.weixin.qq.com/s/8NSEbRKP6tKuV7ERdC7yaQ)

# Dialogue generation

- [对话生成预训练模型速览](https://zhuanlan.zhihu.com/p/428382078)


# Task-oriented dialogue system

- [填槽与多轮对话 | AI产品经理需要了解的AI技术概念](https://coffee.pmcaff.com/article/971158746030208/pmcaff?utm_source=forum&from=related&pmc_param%5Bentry_id%5D=950709304427648)
- [UNIT对话系统初级教学](http://bit.baidu.com/products?id=11)
- [近期任务型对话系统综述以及百度UNIT，理论和实践，我全都要！](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650424818&idx=1&sn=89ddc523600e72beecb318c213f267be&chksm=becdc3a889ba4abe426847e4456cd4a49fed381293dd3b0100bbc8c88f3d8173f59a6853bfd5&mpshare=1&scene=24&srcid=1118BZKVAWr8IhzBeUB6pQjN&sharer_sharetime=1637165459841&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A7W878EXTHkmGU1Jjj3p%2BDw%3D&pass_ticket=3YSLQZ0%2BFGkSbSLIxeI5ld3daRcSE5x5m%2FqFag47PCWFTeogIXft8nu1uI5rJumG&wx_header=0#rd)
- [Latent Tree Models](https://cse.hkust.edu.hk/~lzhang/ltm/index.htm)
- [few shot prompt learning + 任务型对话系统的相关任务](https://zhuanlan.zhihu.com/p/422866442)


## end2end

- [任务型对话系统简述与细节把捉](https://zhuanlan.zhihu.com/p/276323615)
- [任务型对话系统预训练最新研究进展](https://mp.weixin.qq.com/s/b3JSE1o9dr7loafwhEWomA)
- [基于GPT-2的端到端任务型对话系统简述](https://zhuanlan.zhihu.com/p/423021503)
- [端到端的任务型对话](https://zhuanlan.zhihu.com/p/64965964)
- [用于多领域端到端任务型对话系统的动态融合网络](https://mp.weixin.qq.com/s?__biz=Mzg3OTAyMjcyMw==&mid=2247487026&idx=2&sn=94ec0e5588b139958fac8bf535185a7d&chksm=cf0b89def87c00c8fc585a7c7d9d0f418e9f5462f0caa5ff3a135aa48e2b692de326dbd54e72&mpshare=1&scene=24&srcid=0731m0abLZOS6gUxPhv6asaY&sharer_sharetime=1596188226791&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A7617s8pi3kRSz7aN6zWAuo%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [End-to-end LSTM-based dialog control optimized with supervised and reinforcement learning](https://zhuanlan.zhihu.com/p/21654924)

## DST

- [赛尔笔记|基于深度学习方法的对话状态跟踪综述](https://zhuanlan.zhihu.com/p/385533676)
- [【DST系列】DST概述](http://www.360doc.com/content/20/0929/21/7673502_938211385.shtml)
- [AI LIVE | DSTC 8“基于Schema的对话状态追踪”竞赛冠军方案解读](https://zhuanlan.zhihu.com/p/159106327?utm_source=wechat_session&utm_medium=social&utm_oi=615941546193850368&utm_campaign=shareopn)
- [对话状态追踪DST](https://zhuanlan.zhihu.com/p/60190066?utm_source=wechat_session&utm_medium=social&utm_oi=615941546193850368&utm_campaign=shareopn)
- [ACL2020 | 基于槽注意力和信息共享的对话状态追踪](https://zhuanlan.zhihu.com/p/346365003?utm_source=wechat_session&utm_medium=social&utm_oi=615941546193850368&utm_campaign=shareopn)
- [一文看懂任务型对话系统中的状态追踪](https://zhuanlan.zhihu.com/p/51476362?utm_source=wechat_session&utm_medium=social&utm_oi=615941546193850368&utm_campaign=shareopn)
- [对话状态追踪（DST）](https://zhuanlan.zhihu.com/p/345159158?utm_source=wechat_session&utm_medium=social&utm_oi=615941546193850368&utm_campaign=shareopn)
- [任务导向型对话系统——对话管理模型研究最新进展](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247502045&idx=1&sn=3c35232c14f71184a230303aaf533fd6&chksm=96ea175da19d9e4b5dcdd0d33cd0eca73fab2753fb44bc52551d4b85fff9eb66e8bddabc57ff&mpshare=1&scene=24&srcid=1118mjZVf7rkuyh0jPVjzkO1&sharer_sharetime=1637165126500&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A9o43ydTag%2Fnlyi3QNSXCO4%3D&pass_ticket=3YSLQZ0%2BFGkSbSLIxeI5ld3daRcSE5x5m%2FqFag47PCWFTeogIXft8nu1uI5rJumG&wx_header=0#rd)
  
## DPL

- [深度强化学习在对话管理中的应用](https://zhuanlan.zhihu.com/p/352583321)
- [Deep Dyna-Q: 任务型对话策略学习的集成规划](https://zhuanlan.zhihu.com/p/50223176)
- [一文看懂任务型对话中的对话策略学习（DPL）](https://zhuanlan.zhihu.com/p/52692962)
- [深度强化学习在对话管理中的应用](https://zhuanlan.zhihu.com/p/352583321)

## RASA

- [利用Rasa Forms创建上下文对话助手](https://zhuanlan.zhihu.com/p/349170436)
- [RASA实现多domain对话](https://zhuanlan.zhihu.com/p/341412567)
- [rasa填表单激活和结束逻辑代码书写问题 active_loop关键字、写在rules中还是stories中、实际应用](https://blog.csdn.net/weixin_42639575/article/details/119046391)
- [rasa文章导引（用于收藏）](https://zhuanlan.zhihu.com/p/88112269)
- [对话机器人](https://www.zhihu.com/column/c_1154767675480821760)


# KBQA

- [KBQA知识库问答论文分享](https://zhuanlan.zhihu.com/p/126268532)
  - Improved Neural Relation Detection for Knowledge Base Question Answering
    * 2017 ACL
    - 实体链接；基于query，候选实体来对实体排序；基于query，候选关系来对实体排序
    - 两个排序得分融合排序；多跳关系中，取和候选答案相邻的实体c，计算q和c的最大重叠部分的字符特征作为得分，如果得分大于某个阈值，则将其作为约束，取c相邻的候选答案作为最终答案
    - 实体链接使用已有算法；关系检测结合图谱的relation embedding和关系文本
  - Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN  
    - 2018 ACL
    - 实体链接，关系候选，query中实体用特殊符号替换
    - 将关系文本分成两部分，subject类型，关系类型，然后与query计算attention，
    然后再和原始关系embedding融合
    - 将关系文本作为一个整体，和query计算cross attention，然后使用CNN进行特征提取
    - 两个部分的特征拼接    
  - Retrieve and Re-rank: A Simple and Effective IR Approach to Simple Question Answering over Knowledge Graphs  
    - 2018 ACL end2end
    - 使用solr召回三元组
    - 使用正负样本输入孪生网络，正样本由query，pos三元组，query pos三元组等三个部分组成，然后使用卷积网络进行特征提取，max polling，最后拼接三个向量  
  - Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader
    - 2019 ACL
    - 针对基于不完整KB的问答，利用额外的文档知识来补充不完整的KB知识
  



