<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [文本检索匹配](#%E6%96%87%E6%9C%AC%E6%A3%80%E7%B4%A2%E5%8C%B9%E9%85%8D)
- [双塔模型](#%E5%8F%8C%E5%A1%94%E6%A8%A1%E5%9E%8B)
- [交互模型](#%E4%BA%A4%E4%BA%92%E6%A8%A1%E5%9E%8B)
- [交互和双塔结合](#%E4%BA%A4%E4%BA%92%E5%92%8C%E5%8F%8C%E5%A1%94%E7%BB%93%E5%90%88)
- [Vector retrieval](#vector-retrieval)
- [评估指标](#%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## 文本检索匹配
- [21个经典深度学习句间关系模型｜代码&技巧](https://mp.weixin.qq.com/s?__biz=MzAxMTk4NDkwNw==&mid=2247486128&idx=1&sn=3c77c96c6891a94de629677911b42553&chksm=9bb983d4acce0ac298ad04543676d4b0568977d010f3945d51edc0680c785b0b97827aee9028&token=1200978700&lang=zh_CN&scene=21#wechat_redirect)


## 双塔模型
- [谈谈文本匹配和多轮检索](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247488346&idx=3&sn=5ba89654a742af8bdaf17c94546f7a4e&chksm=eb5007c9dc278edfc0d4256fd14dfab3584eece5545a0464029ae3ae67d47be7e92fbc6034b3&scene=0&xtrack=1&exportkey=A2quZRdG6ZUylyxhp59oMVA%3D&pass_ticket=peaJqRABUyiyXUkxShtHPoJ7onMoJTA4OFYeMuNaXmdNKq47G0x8XJEm7afGdVcX#rd)
- [NLP语义匹配](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247507074&idx=3&sn=42be588256e0fc6eee1b646b57c114d2&chksm=eb53c811dc2441076f9ab07f876f5c00819a103d39f8494af6a6aceab3b912267e7dd17d0b30&mpshare=1&scene=24&srcid=1117Io9vftzLt8q6xRYByepS&sharer_sharetime=1637156450934&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A78%2BrTGzFzSs5hWhqtynFeo%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
- [2020深度文本匹配最新进展：精度、速度我都要！](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247534555&idx=3&sn=9ac3ecdd66f3bc9be8cfe27135050718&chksm=90944e48a7e3c75e0c08d0b0a21a51d147dc5fde4a697caec775ceb4ac800682a0ccf1e1d32f&mpshare=1&scene=24&srcid=0916XmZ74y1eLSVwSazTGATm&sharer_sharetime=1600268113155&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A5ZY0n4H%2BzW%2F9Z%2BLBsIwWqg%3D&pass_ticket=X1hVh%2FzYha2Fa9G%2FZWK0bpCofPY07lt8BPBNyjf1xUWYljT%2Bk%2F9q5rZ%2F%2B4bWWFme&wx_header=0#rd)
- [竞赛】天池-新冠疫情相似句对判定大赛top6方案及源码](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490106&idx=1&sn=4f2fa8a4df430cb3aceb1094f5ca791a&source=41#wechat_redirect)
- [深度文本检索模型：DPR, PolyEncoders, DCBERT, ColBERT](https://zhuanlan.zhihu.com/p/523879656)
- [ERNIE-Search：向交互式学习的表征式语义匹配代表作](https://mp.weixin.qq.com/s/5Benqgq1utHIL097XR5FWA)
- [Facebook全新电商搜索系统Que2Search](https://mp.weixin.qq.com/s/S18T913SeyrtVQadMmLPlA)
- [SIGIR20最佳论文：通往公平、公正的Learning to Rank！](https://mp.weixin.qq.com/s?__biz=MzIzMzYwNzY2NQ==&mid=2247485912&idx=1&sn=4f360828048866bca8138846351a80e6&chksm=e8825146dff5d8505f06c6598d04e9f0c3dab0d8dd6fe4c5eac1d42490b06e233c31d1ea2ef9&scene=21#wechat_redirect)
- [前沿重器2 美团搜索理解和召回](https://mp.weixin.qq.com/s?__biz=MzIzMzYwNzY2NQ==&mid=2247486004&idx=1&sn=2725794c67a9350cb3f9feabd4ee1736&chksm=e88252aadff5dbbcc41e48223e550469aee1a37dcaf2ee7d29b52fa174cfb0d7223dc5b4e5b3&scene=21#wechat_redirect)
- [搜索中的深度匹配模型](https://zhuanlan.zhihu.com/p/113244063)
- [WSDM Cup 2020检索排序评测任务第一名经验总结](https://zhuanlan.zhihu.com/p/116013450)
- [京东：个性化语义搜索在电商搜索中的应用](https://mp.weixin.qq.com/s/S9cw-pLIJSa4F9YvqE9uhw)
- [淘宝搜索中基于embedding的召回](https://mp.weixin.qq.com/s/775qZLQaH9IolmqvPz3Sjw)
- [Transformer 在美团搜索排序中的实践](https://mp.weixin.qq.com/s?__biz=MjM5NjQ5MTI5OA==&mid=2651751586&idx=1&sn=a61c9da125e9b7e68473b32e0278b0ea&chksm=bd125def8a65d4f9d20b682345365d5001e9c863d5046acf683da6116b265d168c0340754fc9&scene=21#wechat_redirect)
- [Embedding-based Retrieval in Facebook Search](https://zhuanlan.zhihu.com/p/152570715)
- [SIGIR 2020之MarkedBERT模型：加入传统检索线索的Rerank模型](https://zhuanlan.zhihu.com/p/175981489)
- [双塔模型的最强出装，谷歌又开始玩起“老古董”了？](https://mp.weixin.qq.com/s/crEfe6Zb7q1AoTGrOAmcCQ)

## 交互模型
- [Dual-Cross-Encoder：面向稠密向量检索的Query深度交互的文档多视角表征](https://mp.weixin.qq.com/s/vbtyqWchdfd3loqUsjQJ7w)
- [如何引入外部知识增强短文本匹配？](https://mp.weixin.qq.com/s/mdNGA97bypX6fK3c5ti3kg)

## 交互和双塔结合
- [交互模型你快跑，双塔要卷过来了](https://mp.weixin.qq.com/s/UF0cI7M1-tHBo45BujmQCg)
  - 交互模型慢，精度高，双塔模型与之相反。二者结合：将交互模型的知识蒸馏到双塔模型上

## Vector retrieval
- [笔记︱几款多模态向量检索引擎：Faiss 、milvus、Proxima、vearch、Jina等](https://mp.weixin.qq.com/s/BbCVTOZ_sEyY9_7iWW1dNg)
  - 工具介绍，向量检索方法
- [向量检索使用场景和关键技术](https://mp.weixin.qq.com/s?__biz=MzkxMjM2MDIyNQ==&mid=2247504260&idx=1&sn=0e2ed82e21878373e8e93e67f470dfcb&source=41#wechat_redirect)
  - 向量搜索的应用领域：对话系统，以图搜图，以字搜图
  - 向量索引方法：LSH，近邻图，乘积量化
- [Elasticsearch 向量搜索的工程化实战](https://mp.weixin.qq.com/s/DtT5NhLOInIPgqbETzNOdg)
- [向量检索研究系列算法](https://mp.weixin.qq.com/s/hf7W8gpUAstNEBEnS9s7zQ)
- [Faiss中的IVF,PQ,IVFPQ原理](https://zhuanlan.zhihu.com/p/356373517)

## 评估指标
- [搜索推荐评价指标Precision@k、Recall@k、F1@k、NDCG@k](https://blog.csdn.net/guolindonggld/article/details/121114309)

