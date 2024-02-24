<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [文本分类](#%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)
- [少样本](#%E5%B0%91%E6%A0%B7%E6%9C%AC)
- [多标签分类](#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB)
- [层次标签分类](#%E5%B1%82%E6%AC%A1%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## 文本分类  
  
- [基于Prompt Learning、检索思路实现文本分类，开源数据增强、可信增强技术](https://mp.weixin.qq.com/s/tas7yM8vapxwtlJt-MRZdg)
- [6个你应该用用看的用于文本分类的最新开源预训练模型](https://zhuanlan.zhihu.com/p/130792659)
- [从理论到实践解决文本分类中的样本不均衡问题](https://mp.weixin.qq.com/s?__biz=MzU1NjYyODQ4NA==&mid=2247484661&idx=1&sn=8a91b910e941c87fba79a6154c819d66&chksm=fbc36b9eccb4e288c0e37d811b8511c1b093bb664f6442ec9d9e2189d1ea201613da37a3cd33&mpshare=1&scene=24&srcid=0703DjmDK4AfPSuBKxT8gEMH&sharer_sharetime=1625253571562&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A17XMeW6otqsYpmcg5f1gW0%3D&pass_ticket=ahSCjZBnxTVe3IcKWMxBQVeAXXap9Se8HXejNWF3PIlQHiDsRH5Yr1%2FzLdG%2FTkZA&wx_header=0#rd)
  - 数据的采样方法层面，相似样本生成（simbert），loss层面
- [ICML 2020 | 基于类别描述的文本分类模型](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/106754400)
- [弱监督文本分类](https://mp.weixin.qq.com/s/rgI7R1Z1elCzbSC93O6IhQ)


## 少样本
- [论文分享 | AAAI 2022 探索小样本学习在解决分类任务上过拟合问题的方法](https://mp.weixin.qq.com/s/DymXP95X77mfs2CGJZEyPQ)
- [文本分类大杀器：PET范式](https://mp.weixin.qq.com/s/qW0iwqWhkj12euEXRenZDg)
  - 使用prompt learning方式进行文本分类
  - 不通pattern对效果有很大影响
  - 构造soft label训练数据+无标签MLM任务
- [少样本文本分类 InductionNet](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247502646&idx=4&sn=d1cccfd219b485bf2e5b412d3f08056c&chksm=eb53dfa5dc2456b37657207d86c33758b10884f958909c982ef5926eacd5f08b220fa51536ec&mpshare=1&scene=24&srcid=0527W5ME2Zj4D1bpqs1JhWuZ&sharer_sharetime=1622127690299&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A8UlYy8n%2F03u15ioi5Fk7t8%3D&pass_ticket=ahSCjZBnxTVe3IcKWMxBQVeAXXap9Se8HXejNWF3PIlQHiDsRH5Yr1%2FzLdG%2FTkZA&wx_header=0#rd)
- [只有少量标注样本，如何做好文本分类任务？](https://mp.weixin.qq.com/s/IwdEisDXjbyPLTjjcixqGg)
  - 主要思想是对文本进行更好的向量表示，然后以向量为特征使用SVM算法进行分类
- [ACL2022 | KPT: 文本分类中融入知识的Prompt Verbalizer](https://mp.weixin.qq.com/s/20nwLbn1hezf_BICgdhRzg)
- [小样本学习--笔记整理](https://blog.csdn.net/u010138055/article/details/90690606)


## 多标签分类

- [多标签分类(multi-label classification)综述](https://www.cnblogs.com/cxf-zzj/p/10049613.html)
  - 问题转化方法：分类器链，基于生成的方法，多标签转成单标签
  - 直接建模
- [EMNLP2021论文：元学习大规模多标签文本分类](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650427439&idx=4&sn=3cb2d0edac444a20d61174d877e661d3&chksm=becdd67589ba5f632533c05ac744d8dffb9d825e99e46fb45b6c9e1f69b0ead11819a3384f06&mpshare=1&scene=24&srcid=1127vW4uT3jw6Xmn9tyxtOeN&sharer_sharetime=1637963151380&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=A2REQqlB2R5ChUzrLQzPhuU%3D&pass_ticket=ahSCjZBnxTVe3IcKWMxBQVeAXXap9Se8HXejNWF3PIlQHiDsRH5Yr1%2FzLdG%2FTkZA&wx_header=0#rd)
  - 提出了一种多标签数据的采样策略
- [nn.BCEWithLogitsLoss和nn.MultiLabelSoftMarginLoss有啥区别](https://blog.csdn.net/jiangpeng59/article/details/92016262?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.compare)
- [标签感知的文档表示用于多标签文本分类（EMNLP 2019）](https://zhuanlan.zhihu.com/p/207221522)
  - 对label的文本表示信息进行embedding
  - 文档self-attention,label-document attention
  - attention fusion, 计算权重后加权
- [关于多目标任务有趣的融合方式](https://mp.weixin.qq.com/s/bJpkMbreO9FvRl3-A5pIKA)
- [ACL'22 | 使用对比学习增强多标签文本分类中的k近邻机制](https://mp.weixin.qq.com/s/xtc9MNlZIN5Xbg0nsBThPA)
- 


## 层次标签分类
- [层次文本分类-Seq2Tree：用seq2seq方式解决HTC问题](https://mp.weixin.qq.com/s/eSJF8SAtuYq-DH_VfY-Ixg)
- [label embedding与attention机制在层次文本分类中的简单架构-HAF](https://mp.weixin.qq.com/s/XxFBm5p-srxVkFvt4cnxIA)
