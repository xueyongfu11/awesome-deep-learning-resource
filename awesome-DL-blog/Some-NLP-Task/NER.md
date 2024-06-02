[TOC]




## NER
### 2022
- [【NLP】命名实体识别——IDCNN-CRF论文阅读与总结](https://blog.csdn.net/meiqi0538/article/details/124641801)
### 2021
- [ACL 2021 | 复旦大学邱锡鹏组：面向不同NER子任务的统一生成框架](https://mp.weixin.qq.com/s/2AePxoar9j4MLQLxMzSf_g)
  - 基于BAET模型，使用指针网络，对于不连续实体，需要添加标签 《dis》
  - 推理：将decoder的hidden_state与encoder的每个位置的hidden_state点积，与标签的embedding点积，然后计算softmax
- [工业界求解NER问题的12条黄金法则](https://blog.csdn.net/xixiaoyaoww/article/details/107096739)
- [融合知识的中文医疗实体识别模型](http://www.scicat.cn/yy/20211208/108868.html)
  - 通过构建好的实体库来对文本进行预打标，将预标注结果作为bert模型的输入
- [中文NER | 江南大学提出NFLAT：FLAT的一种改进方案](https://mp.weixin.qq.com/s/-bpr3ySRaPZqRdJgI21A6w)
- [统一NER模型SOTA—W2NER学习笔记](https://mp.weixin.qq.com/s/9A5HXuvVYjHjYb8cn1CYpg)
- [妙啊！MarkBERT](https://mp.weixin.qq.com/s/GDnpvesnX79OS5mhpkd9LA)
- [PERT：一种基于乱序语言模型的预训练模型](https://mp.weixin.qq.com/s/gx6N5QBZozxdZqSOjMKOKA)
- [文本生成序列之前缀语言模型](https://mp.weixin.qq.com/s/WGRRVKiPGR8lZsOkM5Z4Tw)
- [文本生成系列之transformer结构扩展（二）](https://mp.weixin.qq.com/s/brifykEle1Rd7v5F0YxdSg)
- [知识融入语言模型——ERNIE与ERNIE](https://mp.weixin.qq.com/s/trAwVkbwKqUmC5sUbC_S0w)
- [不拆分单词也可以做NLP，哈工大最新模型在多项任务中打败BERT，还能直接训练中文](https://mp.weixin.qq.com/s/UBoMRmymwnw9Ds3S3OW6Mw)
- [BiLSTM上的CRF，用命名实体识别任务来解释CRF](https://mp.weixin.qq.com/s/2Eq1tSt0Wqxh8MULR27qYA)


## 少样本NER
### 2023
- [COLING 2022 | PCBERT: 用于中文小样本NER任务的BERT模型](https://www.aminer.org/research_report/63db41c37cb68b460f84d4fd)

### 2022
- [中文小样本NER模型方法总结和实战](https://cloud.tencent.com/developer/article/2083673)
- [低资源和跨语言NER任务的新进展：词级别数据增强技术](https://mp.weixin.qq.com/s/9vYd9O7BRd_k_56AF5xT0g)
- [COLING 2022 | 少样本NER：分散分布原型增强的实体级原型网络](https://mp.weixin.qq.com/s/vdNKuZRg2Umst0TSn3p2Qw)
- [ACL 2022 | 基于自描述网络的小样本命名实体识别](https://mp.weixin.qq.com/s/WUjK6qM7qkLs66aMoLYaIA)
- [ACL2022 | 序列标注的小样本NER：融合标签语义的双塔BERT模型](https://mp.weixin.qq.com/s/56OH4d7WDYjuLxWh4kW-1w)

### 2021
- [Template-Based Named Entity Recognition Using BART [笔记]](https://zhuanlan.zhihu.com/p/462088365)
- [论文解读：Example-Based Named Entity Recognition](https://blog.csdn.net/qq_36426650/article/details/125504613)
- [微软、UIUC韩家炜组联合出品：少样本NER最新综述](https://mp.weixin.qq.com/s/tiMoFMVdQketm11rdXjiSQ)


## 地址解析
### 2021
- [天池中文NLP地址要素解析大赛Top方案总结](https://mp.weixin.qq.com/s/bjbcT0Yt-Q-4KjQSg-3mFQ)
- [BERT+Biaffine结构中文NLP地址要素解析](https://mp.weixin.qq.com/s/o5BZ8-l-rjyJmF0V_G1cNg)

