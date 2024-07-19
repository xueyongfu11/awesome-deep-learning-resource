[TOC]



## 大模型对齐

- [Alignment 最新进展](https://zhuanlan.zhihu.com/p/650788172)
- [OpenAI前对齐团队「遗作」：RLHF不够用了！用GPT-4训练GPT-4](https://zhuanlan.zhihu.com/p/705989405)
- [有关DPO训练时，为什么chosen和rejected的reward一起下降的猜想](https://zhuanlan.zhihu.com/p/694381064)
- [深度解析DPO及其变体在多种任务上的表现如何，该如何选择](https://mp.weixin.qq.com/s/DwBpfMiSbGJ8N07e6zN4eg)
- [剑桥提出RLHF平替方案：在SFT以外，我们还能拿SFT数据做什么？](https://mp.weixin.qq.com/s/Sbu1-EA6gCKsyUdGpRTuRg)
- [阿里提出大语言模型对齐框架Reward Learning on Policy (RLP)](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/137845870)
- [一些RLHF的平替汇总](https://blog.csdn.net/m0_37310036/article/details/134453906)
- [偏好学习算法并不学习偏好排序](https://zhuanlan.zhihu.com/p/701126178)
- [Self-Play的对齐算法介绍](https://zhuanlan.zhihu.com/p/699292524) 

- RLHF-PPO
  - [ChatGPT 背后的“功臣”——RLHF 技术详解](https://huggingface.co/blog/zh/rlhf)
  - [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://mp.weixin.qq.com/s/J8c7rEmkQH4lBj1pWntv9w)
    - PPO原理解读，非常详细
- [Secrets of RLHF I：大模型RLHF的trick](https://zhuanlan.zhihu.com/p/646385336)
- [RLCD：无需人类反馈即可对齐！田渊栋团队新作RLCD：无害型、有益性、大纲写作全面超越基线模型](https://mp.weixin.qq.com/s/sQolnpmBdCufVVR8q6GG8w)
  - 论文提出基于positive prompt和negative prompt来生成对比性强、质量好的偏好对，然后训练reward模型，接下来的PPO训练部分与常见方案相同
  - 不同于RLAIF，仅使用同一个prompt生成两个回复并打分，RLCD是使用两个对比prompt生成回复。

- [LLM RLHF 2024论文（三）RSO](https://zhuanlan.zhihu.com/p/690198669)
  - DPO算法所使用的数据是SFT或者其他算法采样出来的，而不是最优策略采样出来的
  - 想要估计某一个分布，需要用这个分布下采样出来的数据才能很好地对分布进行估计
  - DPO使用其他策略采样出的数据计算MLE去估计最优策略，会导致数据和最优策略之间的不匹配

- [探索最优POST-TRAINING方案](https://zhuanlan.zhihu.com/p/661323551)
  - 对齐LLMs的对比后训练包括SLic 、DPO 、RLHF 、RLAIF以及结合课程学习不同组合方法的收益对比以及优缺点分析


## 基于Rank的对齐方法

- [RAFT：玩不起RLHF？港科大开源高效对齐算法RAFT「木筏」，GPT扩散模型都能用 - 知乎](https://zhuanlan.zhihu.com/p/623069114)
  - RAFT
- [大语言模型之RRHF](https://zhuanlan.zhihu.com/p/622198781)
  - RRHF	

## DPO及类似工作

- [全面超越DPO：陈丹琦团队提出简单偏好优化SimPO，还炼出最强8B开源模型](https://www.jiqizhixin.com/articles/2024-05-27-8)

- [为什么我们应该做online RLHF/DPO？](https://mp.weixin.qq.com/s/f68yoZkByWlPvckoFK9qCg)

- [仅靠开源数据复刻出LLaMA3指令学习效果，在线迭代RLHF全流程解决方案来了](https://www.jiqizhixin.com/articles/2024-05-18)

- [大模型偏好对齐-ODPO](https://mp.weixin.qq.com/s/FT4XUDDKO4e_aEiq0aqgzA)
  - 将accept和reject的reward差值考虑了进去
- [大模型的PPO、DPO偏好优化算法玩不起？那建议你看一下ORPO](https://zhuanlan.zhihu.com/p/688583797)
  - ORPO
  - 不需要参考模型，在SFT损失的基础上添加了一个不受欢迎生成的惩罚项，也叫赔率比
- [使用KTO进行更好、更便宜、更快速的LLM对齐](https://mp.weixin.qq.com/s/vFrcW43jhraZT8ZaDBxl7A)
  - KTO


## 免微调对齐

- [大模型免微调解锁对话能力，RLHF没必要了！节省大量成本和时间，一作上交大校友](https://zhuanlan.zhihu.com/p/670682075)
  - URIAL, base model的免微调方法

- [BPO：灵活的 Prompt 对齐优化技术](https://zhuanlan.zhihu.com/p/667767805)
  - BPO，主要是优化prompt
- [OPO:无需训练实现价值观实时动态对齐：上交开源价值观对齐方法，闭源与开源大模型均适用](https://mp.weixin.qq.com/s/_CB0LBQVI_2NBiX63pyYSA)
  - OPO，收集相关法律或者道德准则，使用RAG检索与query相关的准则，基于检索结果来生成

## 面试trick

- [在LLM中选择像传统RL中value network和policy network共享底座会有问题吗？如果有解释一下为什么？](https://zhuanlan.zhihu.com/p/699827201)
- [RLHF中，为什么计算了两次actor_prob？](https://www.zhihu.com/question/654282515/answer/3481039875)