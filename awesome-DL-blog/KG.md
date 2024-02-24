<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [综述](#%E7%BB%BC%E8%BF%B0)
- [传统的知识表示](#%E4%BC%A0%E7%BB%9F%E7%9A%84%E7%9F%A5%E8%AF%86%E8%A1%A8%E7%A4%BA)
  - [知识推理](#%E7%9F%A5%E8%AF%86%E6%8E%A8%E7%90%86)
- [表示学习](#%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0)
  - [传统的知识图谱嵌入](#%E4%BC%A0%E7%BB%9F%E7%9A%84%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E5%B5%8C%E5%85%A5)
    - [匹配模型](#%E5%8C%B9%E9%85%8D%E6%A8%A1%E5%9E%8B)
  - [规则增强的表示学习](#%E8%A7%84%E5%88%99%E5%A2%9E%E5%BC%BA%E7%9A%84%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0)
  - [GNN-based KGEs](#gnn-based-kges)
- [实体链接](#%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8E%A5)
- [知识图谱构建](#%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E6%9E%84%E5%BB%BA)
- [事理图谱](#%E4%BA%8B%E7%90%86%E5%9B%BE%E8%B0%B1)
- [多模态知识图谱](#%E5%A4%9A%E6%A8%A1%E6%80%81%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# 综述
- [2022年最新知识图谱综述文章合集](https://zhuanlan.zhihu.com/p/467077971)
- [藏经阁](https://kg.alibaba.com/index.html)
- [历史最全、最细、近一年最新 知识图谱相关经典论文分享](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247486236&idx=1&sn=01df5139be1b14217086605915114597&chksm=97a0c0c8a0d749debf3887a4f5df6736229e2ada4cc4047ec52595a4739c778c8ac3bcbd1fa7&scene=0&xtrack=1&pass_ticket=5l2GTJoNs3UnPjzRsDzXqTZBP6%2Btylp4BwIFxk3aFUwONC5l8MJz3gdjYHCbXS%2FH#rd)
- 

# 传统的知识表示
- [知识图谱-浅谈RDF、OWL、SPARQL](https://www.jianshu.com/p/9e2bfa9a5a06)
- [知识图谱（二）——知识表示](https://blog.csdn.net/u012736685/article/details/96160990)
- [Protégé本体构建入门（知识图谱构建）](https://blog.csdn.net/weixin_42159586/article/details/104007716)
- [网络本体语言](http://www.wendangku.net/doc/cb9166035.html)
- [OWL本体语言中OWL Lite、OWL DL、OWL Full理解](https://blog.csdn.net/qq_38842357/article/details/80706872)
- [语义网络，语义网，链接数据和知识图谱](https://blog.csdn.net/ZJRN1027/article/details/80486306)

## 知识推理
- [知识图谱入门 (七) 知识推理](https://blog.csdn.net/pelhans/article/details/80091322)
  


# 表示学习

- [知识图谱(KG)表示学习](https://mp.weixin.qq.com/s/CyUIlZ9UYQJ_cbJUM7tKrA)
- [知识图谱表示学习综述 | 近30篇优秀论文串讲](https://mp.weixin.qq.com/s/GZ40cpoO_JLvyoa_TEbQzQ)
- [知识图谱表示学习](https://zhuanlan.zhihu.com/p/459340212)



## 传统的知识图谱嵌入

- [传统的知识图谱表示](http://neuralkg.zjukg.org/CN/uncategorized/c-kges/)
- [【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://blog.csdn.net/u012151283/article/details/87255951)

### 匹配模型
- [RESCAL、LFM、DistMult](https://www.cnblogs.com/fengwenying/p/15033263.html)
  

## 规则增强的表示学习 

- [双线性模型（四）（HolE、ComplEx、NAM）](http://www.manongjc.com/detail/25-lecemoptpmmmyrr.html)
- [规则增强的表示学习](http://neuralkg.zjukg.org/CN/uncategorized/%e8%a7%84%e5%88%99%e5%a2%9e%e5%bc%ba%e7%9a%84%e8%a1%a8%e7%a4%ba%e5%ad%a6%e4%b9%a0/)

## GNN-based KGEs
- [关于 GCN，我有三种写法](https://mp.weixin.qq.com/s?__biz=MzA4ODUxNjUzMQ==&mid=2247486162&idx=1&sn=d5871ce9b01bc2b2f83b4364942ea31c&chksm=9029b80ea75e31181b53940155c9a5f3bf84fa15105a1d822eb25b8214cbfdf0357005d76e5d&mpshare=1&scene=24&srcid=&sharer_sharetime=1590198792672&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2BNA8x%2FoupbOsmVMZxM%2B3qg%3D&pass_ticket=LlL6Ad5uohnLAlqJrzan%2BA5dDM3m9%2Bnl4L%2FaTWpnfTNnifRhbExGygOrgXBzVB7b&wx_header=0#rd)
- [基于知识图谱和图卷积神经网络的应用和开发](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650414067&idx=4&sn=999b7fa16964984aa5da055aec0fe370&chksm=becd9da989ba14bf2aaf556a6c1289c1a4402d72a5b6c66be0b5870d5d3fa1d9c76540485683&mpshare=1&scene=24&srcid=&sharer_sharetime=1591924745158&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A7%2Fc6ebupiLJ6582fz26P%2FQ%3D&pass_ticket=LlL6Ad5uohnLAlqJrzan%2BA5dDM3m9%2Bnl4L%2FaTWpnfTNnifRhbExGygOrgXBzVB7b&wx_header=0#rd)
- [Graph neural network](https://www.sciengine.com/SSM/doi/10.1360/N012019-00133)
- [使用时空图神经网络检验对新冠病毒肺炎的预测 | 网络科学论文速递35篇](https://mp.weixin.qq.com/s?__biz=MzIzMjQyNzQ5MA==&mid=2247511382&idx=2&sn=003e40541158b0a36fbdcf90b9b50508&chksm=e897f7dbdfe07ecdf930ab3eef89d2f85ec6ac11e3c3b93c2f90cd4e09b673a5e020d4848e3a&mpshare=1&scene=24&srcid=0731ZJxbb1lgM04YQg6cREB9&sharer_sharetime=1596190515295&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=Axhj0iwSyswvbzMAceXUAmk%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [注意力图神经网络的小样本学习](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247532328&idx=7&sn=bfc11d2674da6e3712fee49fa242b95b&chksm=fc869e3bcbf1172d72261fb86e690935e93caeae196cf6458d040131f20c22458443960c2510&mpshare=1&scene=24&srcid=0731eI4AWBTI3VhQpb5fqb6Z&sharer_sharetime=1596191021680&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A%2Fm2WmK%2BvpDTJyc8P6HxjiM%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [基于 GNN 的图表示学习](https://www.infoq.cn/article/sWzLpqkg70TQhlwQcDhI)
- [从图网络表示到图神经网络](https://www.toutiao.com/article/6797981792621036035/)
- [图神经网络三剑客：GCN、GAT与GraphSAGE](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247503692&idx=1&sn=d4b21689389baa4724fd642fb8f17a0b&chksm=96ea10cca19d99dade6c497373f583e9951128064331df2fb31ffd43be2609b205bb15761f24&scene=0&xtrack=1&exportkey=A2BHkaUS95MGf8o5gHyFpl0%3D&pass_ticket=Df0Hp50Eu2KPStu3JpQZ9Z6bR9SWbRuBOlf%2B3ZYlN45nWCSRWViMRhGlg2I6f9uz#rd)
- [ICLR 2020 开源论文 | 多关系图神经网络CompGCN](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247503628&idx=2&sn=9eca256b5d39ade2e12192e2f566fe03&chksm=96ea108ca19d999af6f693c4bc024f376541f880738ddc333f2f6a33df29c4eb783b4345374b&scene=0&xtrack=1&exportkey=A7Z%2Bmt57Z9kWTBoZmWJWsIQ%3D&pass_ticket=Df0Hp50Eu2KPStu3JpQZ9Z6bR9SWbRuBOlf%2B3ZYlN45nWCSRWViMRhGlg2I6f9uz#rd)
- [论文分享 ： Modeling Relational Data with GCN](https://zhuanlan.zhihu.com/p/61834680)
- [Graph-GCN](https://www.cnblogs.com/ChenKe-cheng/p/11874273.html)
- [论文笔记 | 使用GCN建模关系数据](https://www.jianshu.com/p/9804b81a4151)
- [GCN的概念与应用](https://zhuanlan.zhihu.com/p/72546603?utm_source=wechat_session)
- [拉普拉斯矩阵与拉普拉斯算子的关系](https://zhuanlan.zhihu.com/p/85287578)
- [如何理解 Graph Convolutional Network（GCN）？](https://www.zhihu.com/question/54504471/answer/332657604)
- [一文读懂图卷积GCN](https://zhuanlan.zhihu.com/p/89503068)
- [华人博士发 127 页万字长文：自然语言处理中图神经网络从入门到精通](https://mp.weixin.qq.com/s/brIIukHx06daYw-M3nN8PQ)
- 

# 实体链接
- [【比赛获奖方案开源】中文短文本实体链指比赛技术创新奖方案开源](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247515188&idx=2&sn=cd456ad91d95b6c187b8d795f3bdc6e0&chksm=fc865d27cbf1d43130f3e2bc90842ae95faee5e908bc10e27606ff8cd68a882ce265e1b7903d&scene=0&xtrack=1&pass_ticket=5l2GTJoNs3UnPjzRsDzXqTZBP6%2Btylp4BwIFxk3aFUwONC5l8MJz3gdjYHCbXS%2FH#rd)
- [百度实体链接比赛后记：行为建模和实体链接（含代码分享）](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247499381&idx=1&sn=16f58c043a47ce4d5082e63e496c4d6b&chksm=96ea21f5a19da8e3faac75a67265625f09ae08569588ebdf5898512e521d9262261764379df2&scene=0&xtrack=1&pass_ticket=5l2GTJoNs3UnPjzRsDzXqTZBP6%2Btylp4BwIFxk3aFUwONC5l8MJz3gdjYHCbXS%2FH#rd)
- 


# 知识图谱构建
- [他山之石 | 丁香园 医疗领域图谱的构建与应用](https://mp.weixin.qq.com/s/dTAoI3pfCCT0CjNveanaxA)


# 事理图谱
- [图谱实战 | 基于金融场景的事理图谱构建与应用](https://mp.weixin.qq.com/s/S4S0qTtWKuWPKIsBkRnxhw)
  - 只抽取显式的事理关系，不对隐式事理关系进行抽取，只抽取因果关系，其他关系不重要
  - 事理关系抽取：先抽取因果关系词，再因事件和果事件，方法同事件模型
  - 事例对齐：基于文本和子图的方法对齐
- [百度事件图谱技术与应用](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247513234&idx=1&sn=e5ae4715f396f4cc8d63919a4f01cd86&chksm=fbd706fecca08fe81c24da702b22f42ad74e02eac42beafb77b66370e690cf4440ef40c477a6&mpshare=1&scene=24&srcid=1125MFEHHk5B28uV7t3go9Et&sharer_sharetime=1606319309592&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A7B18abxjRagXO1eHOEcZbA%3D&pass_ticket=l%2FwLTQLEiN41ZMPE4ecjUee90P1pxZN0%2BL5MASpFjA%2Bgl02XuicToi%2FXGttULYXk&wx_header=0#rd)
- [基于事理图谱的文本推理](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650799466&idx=1&sn=0a055ea5a682108684d63e335608b055&chksm=8f476881b830e197860b9139e19704b20d0a495ef4d05a1974d1981e0d35aec42f123c401c09&mpshare=1&scene=1&srcid=0927vDop6mqRUqvi53iJg7l4&sharer_sharetime=1601178608128&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A6ex04XvSLxFKCEzZw1A3mE%3D&pass_ticket=dHTQ1tEcNnJx3zi33GpdvKYRIx04Yw%2Fx1Th6gfOGi9LT46IvshrrywPlXoxF6Cji&wx_header=0#rd)
- [哈工大赛尔 | 事理图谱：事件演化的规律和模式](http://blog.openkg.cn/%e5%93%88%e5%b7%a5%e5%a4%a7%e8%b5%9b%e5%b0%94-%e4%ba%8b%e7%90%86%e5%9b%be%e8%b0%b1%ef%bc%9a%e4%ba%8b%e4%bb%b6%e6%bc%94%e5%8c%96%e7%9a%84%e8%a7%84%e5%be%8b%e5%92%8c%e6%a8%a1%e5%bc%8f/#more-633)
- [赛尔原创 | 抽象因果事理图谱的构建和应用](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650791483&idx=1&sn=e3238c78669cf136ab05b546816e50d5&chksm=8f474bd0b830c2c6423a0ec28c645152587f5b96ffae3348a84acb62fd7979f1d81c3474be68&scene=21#wechat_redirect)
- [哈工大SCIR正式对外发布金融事理图谱Demo V1.0](https://www.jiqizhixin.com/articles/2018-09-10-7)
- [刘挺 | 从知识图谱到事理图谱](https://blog.csdn.net/tgqdt3ggamdkhaslzv/article/details/78557548)
- [从工业应用角度解析事理图谱](https://zhuanlan.zhihu.com/p/53699796  )


# 多模态知识图谱
- [【NLPCC2020】多模态知识图谱构建、推理与挑战，东南大学王萌博士](https://mp.weixin.qq.com/s/0gQSTKdyhBi9Ag-UcVV9mQ)
- [深度学习论文和代码的多模态知识图谱](https://mp.weixin.qq.com/s/_MRljJhamlo9M46sriiF8A)
- [OpenKG开源系列｜首个多模态开放知识图谱OpenRichpedia (东南大学)](https://mp.weixin.qq.com/s/PAHa0VfebDPNOQu0lQbPEQ)
- [阿里小蜜多模态知识图谱的构建及应用](https://mp.weixin.qq.com/s/bTqr5EEQD5_rModP8NR99g)
- [【综述专栏】多模态知识图谱前沿进展](https://mp.weixin.qq.com/s/VNMkSOs0aSLaBHj7Rij2gw)
- [复旦大学：多模态知识图谱最新综述](https://mp.weixin.qq.com/s/5BzvtF-Ua2ty07iAjiRxcA)
- 