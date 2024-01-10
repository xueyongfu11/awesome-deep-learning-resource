<!-- TOC -->

- [中文数据集](#%E4%B8%AD%E6%96%87%E6%95%B0%E6%8D%AE%E9%9B%86)
    - [通用语料](#%E9%80%9A%E7%94%A8%E8%AF%AD%E6%96%99)
- [英文数据集](#%E8%8B%B1%E6%96%87%E6%95%B0%E6%8D%AE%E9%9B%86)
- [数据平台](#%E6%95%B0%E6%8D%AE%E5%B9%B3%E5%8F%B0)

<!-- /TOC -->
## 中文数据集

- https://github.com/InsaneLife/ChineseNLPCorpus
- https://github.com/brightmart/nlp_chinese_corpus
- https://github.com/nonamestreet/weixin_public_corpus
- https://github.com/CLUEbenchmark/CLUEDatasetSearch
  - CLUE：搜索所有中文NLP数据集，附常用英文NLP数据集
- https://github.com/liucongg/NLPDataSet
- https://github.com/SophonPlus/ChineseNlpCorpus
- https://github.com/brightmart/nlp_chinese_corpus
- THUOCL：清华大学开放中文词库:http://thuocl.thunlp.org/
- 中文财报数据网站：http://www.cninfo.com.cn/new/index


### 通用语料

- 超对称：https://bbt.ssymmetry.com/data.html
  - 通用语料、金融语料

- 中文法律数据
  - https://github.com/guoxw/wenshu-  裁判文书数据集
  - https://github.com/pengxiao-song/awesome-chinese-legal-resources
    - 本仓库致力于收集和整理全面的中国法律数据资源，旨在帮助研究人员及从业者展开工作

- 维基百科json版(wiki2019zh) 
  - [google](https://drive.google.com/file/d/1EdHUZIDpgcBoSqbjlfNKJ3b1t0XIUjbt/view?usp=sharing)
  - [baidu](https://pan.baidu.com/s/1uPMlIY3vhusdnhAge318TA)
  - 可以做为通用中文语料，做预训练的语料或构建词向量，也可以用于构建知识问答。

- 新闻语料json版(news2016zh)
  - [google](https://drive.google.com/file/d/1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp/view?usp=sharing)
  - [baidu pw:k265](https://pan.baidu.com/s/1MLLM-CdM6BhJkj8D0u3atA)
  - 可以用于训练【标题生成】模型，或训练【关键词生成】模型（选关键词内容不同于标题的数据）；亦可以通过新闻渠道区分出新闻的类型。
  - 包含了250万篇新闻。新闻来源涵盖了6.3万个媒体，含标题、关键词、描述、正文。( 原始数据9G，压缩文件3.6G；新闻内容跨度：2014-2016年)

- 百科类问答json版(baike2018qa)
  - [google](https://drive.google.com/open?id=1_vgGQZpfSxN_Ng9iTAvE7hM3Z7NVwXP2)
  - [baidu pw:fu45](https://pan.baidu.com/s/12TCEwC_Q3He65HtPKN17cA)
  - 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别，其中频率达到或超过10次的类别有434个。
  - 可以做为通用中文语料，训练词向量或做为预训练的语料；也可以用于构建百科类问答；其中类别信息比较有用，可以用于做监督训练，从而构建更好句子表示的模型、句子相似性任务等。

- 百度百科
  - 只能自己爬，爬取得链接：`https://pan.baidu.com/share/init?surl=i3wvfil` 提取码 neqs 。 

- https://dumps.wikimedia.org/zhwiki/

## 英文数据集

- https://gluebenchmark.com/tasks
  - GLUE


## 数据平台

- [CLUE](https://www.cluebenchmarks.com/index.html)
- [千言中文数据集](https://www.luge.ai/#/)
- [智源指数CUGE](http://cuge.baai.ac.cn/#/)
- [天池数据集](https://tianchi.aliyun.com/dataset)
- [格物钛](https://gas.graviti.cn/open-datasets)
- [超神经](https://hyper.ai/datasets)
- [GLUE](https://gluebenchmark.com/)
- [Huggingface dataset](https://huggingface.co/datasets)
- [Kaggle dataset](https://www.kaggle.com/datasets)
- [Paper With Code 数据集](https://www.paperswithcode.com/datasets)
- [LinDat](https://lindat.mff.cuni.cz/)
- [Google dataset](https://datasetsearch.research.google.com/)
- 搜狗实验室
  - 搜狗实验室提供了一些高质量的中文文本数据集，时间比较早，多为2012年以前的数据。
  - [https://www.sogou.com/labs/resource/list_pingce.php](https://link.zhihu.com/?target=https%3A//www.sogou.com/labs/resource/list_pingce.php)
- 中科大自然语言处理与信息检索共享平台
  - [http://www.nlpir.org/?action-category-catid-28](https://link.zhihu.com/?target=http%3A//www.nlpir.org/%3Faction-category-catid-28)
- 中文语料小数据
  - 包含了中文命名实体识别、中文关系识别、中文阅读理解等一些小量数据。
  - https://github.com/crownpku/Small-Chinese-Corpus
- 维基百科数据集
  - https://dumps.wikimedia.org/
