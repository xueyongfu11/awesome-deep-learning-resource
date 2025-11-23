[TOC]




## 文本纠错

- [针对OCR的NLP纠错：从原理到实践](https://zhuanlan.zhihu.com/p/179957371)
- [AI LIVE | 文本纠错技术探索和实践](https://zhuanlan.zhihu.com/p/159101860)
- [NLP上层应用的关键一环——中文纠错技术简述](https://zhuanlan.zhihu.com/p/82807092)
- [小爱同学-基于BERT的ASR纠错](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247503412&idx=1&sn=75ef312902713d3766a43a6c71e1024e&chksm=fbd77c58cca0f54e3a9ffbe9ec075a144e8b16a171287367173d4a0d69f511106335c7b05298&scene=27#wechat_redirect)
  - 基于模糊拼音的数据生成：划分成5个模糊拼音等级
  - 预训练一个基于拼音的的语言模型，将拼音的embedding和token的embedding拼接作为输入，
  然后经过一个softmax预测可能的正确的token
  - 预测时对此表增加限制；句子中存在多个错误是，可以递归的进行预测
- [策略算法工程师之路-Query纠错算法](https://zhuanlan.zhihu.com/p/145198390)
- [垂直电商搜索纠错](https://zhuanlan.zhihu.com/p/161946260)
  - 检错模型：基于bert模型，类似序列标注，0001001
  - 召回模型：基于bert模型的MLM机制
  - 排序模型：根据纠错前后词频特征，文本、拼音编辑距离特特征，语言模型ppl特征等，使用LR或者树模型进行排序
- [百度中文纠错技术](https://www.infoq.cn/article/25fsqE_h9ltDkfLJlcti)
  - 错误检测：使用序列表示+CRF方式：使用特征包括token，句法，词法信息，字词多粒度特征
  - 候选召回：包括离线候选挖掘（构建训练数据），在线的候选召回
  - 接错排序：基于受限词表的语言模型，关联知识（原始query召回部分句子）
- [医疗健康领域的短文本解析探索（三) ----文本纠错](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650415123&idx=2&sn=f3e17a19b9bb96627d24d96ee8167dc5&chksm=becda64989ba2f5fef5eac58743e17d4fc97440b11df1658d8ef50c171fb118f71f57ee58c03&mpshare=1&scene=24&srcid=0731WF3sx2ekxIjHQuttwsFS&sharer_sharetime=1596188559215&sharer_shareid=9d627645afe156ff11b0a8519d982bcd&exportkey=AxZACC%2FGGHzrwIQGtUNf4mY%3D&pass_ticket=IL%2BeHRprAt5yAlLjjC250jaLkeHDOYyDyV4vRbYX%2F0r7c3KJ%2FwPqrBhOiTesV9Z9&wx_header=0#rd)
- [中文语法错误诊断CGED2021评测TOP1队伍总结](https://zhuanlan.zhihu.com/p/444117518)
- 


## 中文拼写纠错
- [中文拼写检测（Chinese Spelling Checking）相关方法、评测任务、榜单](https://blog.csdn.net/qq_36426650/article/details/122807019)
- [腾讯中文纠错模型 PLOME 论文详解！](https://juejin.cn/post/7039144348021162021)

## 语法错误
- [EMNLP'22 | 苏大LAGroup提出：利用定制句法信息的语法纠错](https://mp.weixin.qq.com/s/Y1iMxfhJ9xMKiOE9e8P7Rw)
- [文本语法纠错不完全调研：学术界 v.s. 工业界最新研究进展](https://mp.weixin.qq.com/s/Dj8KIe6LbVGonV-Kk9mO2Q)
- [语法纠错的研究现状](https://mp.weixin.qq.com/s/0_qp1WsrEsjnj8ST4zQyTQ)
  - 伪标注数据生成方法
  - 语法纠错模型介绍
- [NLPCC2018中文语法错误修正任务分享](http://techblog.youdao.com/?p=1281)
- [文本纠错 | 怎样改善模型对 multi-typo 的纠正效果？](https://mp.weixin.qq.com/s/GIOQo3iguAJ5MgW7MjMZww)

## ASR纠错
- [拼写、常识、语法、推理错误都能纠正，云从提出基于BART的语义纠错方法](https://mp.weixin.qq.com/s/uGeQgyPUfz5VRlpp1pnCbA)
- [基于BERT的ASR纠错](https://mp.weixin.qq.com/s/Y1iMxfhJ9xMKiOE9e8P7Rw)