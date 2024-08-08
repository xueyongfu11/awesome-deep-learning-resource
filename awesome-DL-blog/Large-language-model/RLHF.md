[TOC]



## 大模型对齐

- [有关DPO训练时，为什么chosen和rejected的reward一起下降的猜想](https://zhuanlan.zhihu.com/p/694381064)
- [深度解析DPO及其变体在多种任务上的表现如何，该如何选择](https://mp.weixin.qq.com/s/DwBpfMiSbGJ8N07e6zN4eg)
- [剑桥提出RLHF平替方案：在SFT以外，我们还能拿SFT数据做什么？](https://mp.weixin.qq.com/s/Sbu1-EA6gCKsyUdGpRTuRg)
- [Self-Play的对齐算法介绍](https://zhuanlan.zhihu.com/p/699292524) 
- RLHF方法-PPO详解
  - [ChatGPT 背后的“功臣”——RLHF 技术详解](https://huggingface.co/blog/zh/rlhf)
  - [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://mp.weixin.qq.com/s/J8c7rEmkQH4lBj1pWntv9w)
    - PPO原理解读，非常详细
  - [详解大模型RLHF过程（配代码解读）](https://blog.csdn.net/qq_27590277/article/details/132614226)
- [如何完成一次成功的对齐(1)：SFT篇](https://zhuanlan.zhihu.com/p/687926037)

## offline-rlhf

- [为什么我们应该做online RLHF/DPO？](https://mp.weixin.qq.com/s/f68yoZkByWlPvckoFK9qCg)
- [仅靠开源数据复刻出LLaMA3指令学习效果，在线迭代RLHF全流程解决方案来了](https://www.jiqizhixin.com/articles/2024-05-18)


## 免微调对齐

- [大模型免微调解锁对话能力，RLHF没必要了！节省大量成本和时间，一作上交大校友](https://zhuanlan.zhihu.com/p/670682075)
  - URIAL, base model的免微调方法
- [OPO:无需训练实现价值观实时动态对齐：上交开源价值观对齐方法，闭源与开源大模型均适用](https://mp.weixin.qq.com/s/_CB0LBQVI_2NBiX63pyYSA)
  - OPO，收集相关法律或者道德准则，使用RAG检索与query相关的准则，基于检索结果来生成

## 面试trick

- [在LLM中选择像传统RL中value network和policy network共享底座会有问题吗？如果有解释一下为什么？](https://zhuanlan.zhihu.com/p/699827201)
- [RLHF中，为什么计算了两次actor_prob？](https://www.zhihu.com/question/654282515/answer/3481039875)