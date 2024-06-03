[TOC]




# Chinese

## Task-oriented dialogue system

- Medical DS
  - 710个对话 67种症状 4种疾病 | 2018年   | Liu et al. | 复旦大学
  - [paper链接](http://www.sdspeople.fudan.edu.cn/zywei/paper/liu-acl2018.pdf) | [数据集链接](http://www.sdspeople.fudan.edu.cn/zywei/data/acl2018-mds.zip)

- NLPCC2018 Shared Task 4
  - 中文呢真实商用车载语音任务型对话系统的对话日志.  5800对话 2.6万问题 | 2018年
  - [paper链接](http://tcci.ccf.org.cn/conference/2018/papers/EV33.pdf) 
  - [训练开发集](http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata04.zip) [测试集](http://tcci.ccf.org.cn/conference/2018/dldoc/tasktestdata04.zip)
  
- SMP-2020-ECDT
  - 这是一系类数据集，每年都会有新的数据集放出
  - 小样本对话语言理解数据集。论文中叫FewJoint 基准数据集，来自于讯飞AIUI开放平台上真实用户语料和专家构造的语料(比例大概为3：7)，包含59个真实domain，目前domain最多的对话数据集之一，可以避免构造模拟domain，非常适合小样本和元学习方法评测。其中45个训练domain，5个开发domain，9个测试domain。
  - 数据集论文：https://arxiv.org/abs/2009.08138。
  数据集下载地址：https://atmahou.github.io/attachments/FewJoint.zip
  小样本工具平台主页地址：https://github.com/AtmaHou/MetaDialog

- SMP-2019-NLU
  - 包含领域分类、意图识别和语义槽填充三项子任务的数据集
  
- SMP-2017
  - 中文对话意图识别数据集，官方git和数据: [https://github.com/HITlilingzhi/SMP2017ECDT-DATA](https://github.com/HITlilingzhi/SMP2017ECDT-DATA)
  - 论文：[https://arxiv.org/abs/1709.10217  ](https://arxiv.org/abs/1709.10217)
  
- 中文任务型对话系统数据集 https://github.com/thu-coai/CrossWOZ
- 任务型对话数据集 https://github.com/sz128/NLU_datasets_with_task_oriented_dialogue
- 任务型对话数据集 https://github.com/alexa/dialoglue
- end to end dialogue dataset https://github.com/yizhen20133868/Retriever-Dialogue
  

## chitchat dataset

- DuLeMon中文长时记忆对话数据集
  - Chatbot可以基于记住的用户画像信息进行深入聊天，体现长期记忆的能力；用途：提高对话系统的长期记忆能力
  - https://www.luge.ai/#/luge/dataDetail?id=60

- Diamante中文开放域闲聊数据集
  - 机器辅助人工标注的中文闲聊数据集
  - https://www.luge.ai/#/luge/dataDetail?id=52

- CDConv中文对话一致性检测数据集
  - 可用作闲聊数据集
  - https://www.luge.ai/#/luge/dataDetail?id=65

- Tencent中文开放域对话数据集
  - https://www.luge.ai/#/luge/dataDetail?id=36

- LUGE-Dialogue开放域对话数据集合
  - https://www.luge.ai/#/luge/dataDetail?id=26

- LCCC开放域短文本对话数据集
  - https://www.luge.ai/#/luge/dataDetail?id=34

- 豆瓣中文开放域对话数据集
  - https://www.luge.ai/#/luge/dataDetail?id=33

- DuRecDial对话推荐数据集
  - https://www.luge.ai/#/luge/dataDetail?id=31

- https://github.com/rustch3n/dgk_lost_conv 
  - 中文电影对白语料，噪音比较大，许多对白问答关系没有对应好

- https://github.com/kite1988/nus-sms-corpus 
  - 包含中文和英文短信息语料，据说是世界最大公开的短消息语料

- https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data 
  - ChatterBot聊天引擎提供的一点基本中文聊天语料，量很少，但质量比较高

- https://github.com/rustch3n/dgk_lost_conv/tree/master/results 
  - https://github.com/candlewill/Dialog_Corpus
  - 据传这就是小黄鸡的语料

- https://github.com/Marsan-Ma/chat_corpus 
  - chat corpus collection from various open sources
  - 包括：开放字幕、英文电影字幕、中文歌词、英文推文

- chatterbot https://github.com/gunthercox/chatterbot-corpus/tree/master/chatterbot_corpus/data/chinese

- douban（豆瓣多轮） https://github.com/MarkWuNLP/MultiTurnResponseSelection 

- ptt（PTT八卦语料） https://github.com/zake7749/Gossiping-Chinese-Corpus 

- qingyun（青云语料） 

- subtitle（电视剧对白语料）  https://github.com/fateleak/dgk_lost_conv 

- tieba（贴吧论坛回帖语料）  https://pan.baidu.com/s/1mUknfwy1nhSM7XzH8xi7gQ 密码:i4si 

- weibo（微博语料）  61.93.89.94/Noah_NRM_Data/   


## FAQ

- https://github.com/Samurais/insuranceqa-corpus-zh 
  -通过翻译  https://github.com/shuzi/insuranceQA 产生的数据集。train_data含有问题12,889条，
  数据 141779条，正例：负例 = 1:10； test_data含有问题2,000条，数据 22000条，正例：负例 = 1:10；
  valid_data含有问题2,000条，数据 22000条，正例：负例 = 1:10
  
- https://github.com/Samurais/egret-wenda-corpus 
  -由白鹭时代官方论坛问答板块10,000+ 问题中，选择被标注了“最佳答案”的纪录汇总而成。人工review raw data，给每一个问题，
  一个可以接受的答案。目前，语料库只包含2907个问答。 

- 地址：https://github.com/SophonPlus/ChineseNlpCorpus


## CQA(社区问答)

- 社区问答json版(webtext2019zh) ：大规模高质量数据集
  - [google](https://drive.google.com/open?id=1u2yW_XohbYL2YAK6Bzc5XrngHstQTf0v)
  - 含有410万个预先过滤过的、高质量问题和回复。每个问题属于一个【话题】，总共有2.8万个各式话题，话题包罗万象。
    从1400万个原始问答中，筛选出至少获得3个点赞以上的的答案，代表了回复的内容比较不错或有趣，从而获得高质量的数据集。
    除了对每个问题对应一个话题、问题的描述、一个或多个回复外，每个回复还带有点赞数、回复ID、回复者的标签。
    数据集划分：数据去重并分成三个部分。训练集：412万；验证集：6.8万；测试集a：6.8万；测试集b，不提供下载。
  - 构建百科类问答：输入一个问题，构建检索系统得到一个回复或生产一个回复；或根据相关关键词从，社区问答库中筛选出你相关的领域数据  
    训练话题预测模型：输入一个问题(和或描述)，预测属于话题。  
    训练社区问答(cQA)系统：针对一问多答的场景，输入一个问题，找到最相关的问题，在这个基础上基于不同答案回复的质量、  
    问题与答案的相关性，找到最好的答案。  
    做为通用中文语料，做大模型预训练的语料或训练词向量。其中类别信息也比较有用，可以用于做监督训练，从而构建更好句子表示的模型、句子相似性任务等。  
    结合点赞数量这一额外信息，预测回复的受欢迎程度或训练答案评分系统。  


## NL2SQL

- NL2SQL 
  - 单表，简单 [NL2SQL](https://arxiv.org/abs/2006.06434)          
  
- CSpider 
  - 多表 复杂 [CSpider](https://arxiv.org/abs/1909.13293)                  
  
- DuSQL  
  - 多表 复杂 [DuSQL](https://www.aclweb.org/anthology/2020.emnlp-main.562.pdf) 

- SQUALL数据集：https://blog.csdn.net/weixin_43413013/article/details/126859147


- Spider数据集:https://juejin.cn/post/7085557671528660999

## Other

- 千言对话数据集
  - 包含知识对话、推荐对话、画像对话。详细见[官网](https://aistudio.baidu.com/aistudio/competition/detail/48/?isFromLUGE=TRUE)
  
- [CATSLU](https://dl.acm.org/doi/10.1145/3340555.3356098)
  - 之前的一些对话数据集集中于语义理解，而工业界真实情况ASR也会有错误，往往被忽略。而是一个中文语音+NLU文本理解的对话数据集，可以从语音信号到理解端到端进行实验，例如直接从音素建模语言理解（而非word or token）


- 训练中文对话系统的语料库 https://github.com/candlewill/Dialog_Corpus

- 中文聊天语料库 https://github.com/codemayq/chinese_chatbot_corpus

- https://github.com/thu-coai/KdConv
  - 基于知识图谱构建的多轮对话数据集
