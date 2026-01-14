[TOC]



## resource

- https://github.com/RLHFlow/Online-RLHF
- https://github.com/RLHFlow/RLHF-Reward-Modeling
- https://github.com/PKU-Alignment/safe-rlhf
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
- https://github.com/huggingface/trl
- https://github.com/huggingface/alignment-handbook
  - DPO，IPO，KTO，excluding PPO and reward model
- https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
  - reward model realted codes from trlx
- https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat
  - 提及两种reward loss计算方法
- https://github.com/OpenLMLab/MOSS-RLHF
- https://github.com/SupritYoung/Zhongjing
  - 医疗领域，使用了rlhf
- https://huggingface.co/blog/trl-peft
- finetune Mistral-7b with DPO 
  - [Fine-tune a Mistral-7b model with Direct Preference Optimization](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac)
- https://github.com/CarperAI/trlx
- https://huggingface.co/blog/pref-tuning
  - DPO/IPO/KTO等模型的对比实验
  - 经过bug修复的IPO实现的最好的效果。其次是KTO相比DPO/IPO-src相关更好，当时在一定的超参设置下，DPO/IPO也能取得相比KTO更好的效果
  - 经过实验验证，DPO仍然是鲁棒性比较强的模型，KTO不需要成对的偏好数据，因此也具有一定的竞争优势
  


## self-improving

- Bootstrapping Language Models with DPO Implicit Rewards
  - 2024.06
  - 方法与我的想法不谋而合，是对iterative DPO的改进，reward信号使用DPO的隐式奖励来对采样的数据进行标注
  - 为了减少响应的长度，添加了长度的正则loss
- self-Play Fine-Tuning Converts Weak Language Models  to Strong Language Models
  - 2024.01
  - 提出了一种自我博弈的方法提升弱模型到强模型，具体思路是main player的目标是最大化human  response与生成response的差值，而opponent player的目标是减小生成回复和human  response的差值，然后以一种对抗的方式进行提升，注意main player想比opponent player多一个iteration
  - 这种方法可以形式化为类DPO的公式描述，policy model相比reference model多一个iteration
  - 实验证明，SPIN相比DPO，不需要偏好数据，仅需要SFT数据，并且效果更好
- Self-Rewarding Language Models
  - 2024.01
  - 通过大模型生成回复，并用大模型自身对生成的回复进行打分
  - 基于打分结果筛选得分最高和最低的回复作为偏好数据对，然后使用DPO进行训练，相比直接用最高分数据微调的模型效果要好
  - 以上训练过程会经过多次迭代，每次迭代会用到之前创建的数据
  - [Meta发布自我奖励机制，Llama在3轮训练后超越GPT-4](https://zhuanlan.zhihu.com/p/680274984)
  - code: https://github.com/lucidrains/self-rewarding-lm-pytorch
- Adversarial Preference Optimization: Enhancing Your Alignment via RM-LLM Game
  - 2023.11, ACL findings2024
  - LLM模型需要不断提高回复质量，使得自己的回复和金标数据之间的得分差距减小，而RM模型需要不断将LLM回复和金标回复的得分差距拉大
  - 同时两个KL正则项会约束RM和LLM不要对抗得过于离谱。通过这种博弈，RM可以跟随LLM的变化而迭代，模型分布偏移的问题也就得到缓解了
  - [APO｜利用GAN的思想训练RLHF中的RM](https://zhuanlan.zhihu.com/p/674776494)
  - 想法：当前的很多模型的表现与gpt-4不相上下，当把gpt-4作为gold label时，可能会影响模型的效果？

## Blog

- [强化微调RFT技术解析](https://blog.csdn.net/AIBigModel/article/details/144350849)
- [比 GRPO 更稳定更高效：GSPO 算法如何颠覆大模型 RL 训练](https://zhuanlan.zhihu.com/p/1932770229693450218)
  - GSPO的advantage的计算与GRPO相同，都是使用组内相对优势
  - GSPO和GRPO的概率比（重要性比）的计算方法不同，GRPO是每个 token 都会计算一个概率比，而GSPO则将整个序列看作一个整体，计算序列级别的概率比。
- [从 PPO、DPO 到 GRPO：万字长文详解大模型训练中的三大关键算法](https://mp.weixin.qq.com/s/OMpD6ITqNi4jX95nSRC2Ig)
- [大模型面试题：PPO算法中Critic和Reward的区别是啥？](https://mp.weixin.qq.com/s/dnPbaMZJilMVhObHMZ7KKw)
- [面试题题：PPO算法到底是on-policy还是off-policy](https://mp.weixin.qq.com/s/N-miDef7gQG7Ev1VYzUfag)
- [面试题：DPO训练过程中，training positive和 negative的概率同时下降的原因？](https://mp.weixin.qq.com/s/KsWsKmXFCNmStfzxrT3aXg)
- [Iterative Length-Regularized DPO: 7B模型也可以打败GPT4](https://zhuanlan.zhihu.com/p/706172882)
  - 提出了一种加入了长度正则项的迭代式的DPO算法
- [LD-DPO：基于DPO的长度脱敏偏好优化算法](https://zhuanlan.zhihu.com/p/5748074631)
  - 通过计算DPO损失相对两个预测变量（$y_w$、$y_l$）的梯度，发现二者梯度之比取决于actor模型的概率预测，而概率预测是逐token相乘的结果。
  - 基于该发现提出了长度脱敏算法LD-DPO

### 大模型对齐

- [深度解析DPO及其变体在多种任务上的表现如何，该如何选择](https://mp.weixin.qq.com/s/DwBpfMiSbGJ8N07e6zN4eg)
- [剑桥提出RLHF平替方案：在SFT以外，我们还能拿SFT数据做什么？](https://mp.weixin.qq.com/s/Sbu1-EA6gCKsyUdGpRTuRg)
- [Self-Play的对齐算法介绍](https://zhuanlan.zhihu.com/p/699292524) 
- [如何完成一次成功的对齐(1)：SFT篇](https://zhuanlan.zhihu.com/p/687926037)
- [在LLM中选择像传统RL中value network和policy network共享底座会有问题吗？如果有解释一下为什么？](https://zhuanlan.zhihu.com/p/699827201)

### offline-rlhf

- [为什么我们应该做online RLHF/DPO？](https://mp.weixin.qq.com/s/f68yoZkByWlPvckoFK9qCg)
- [仅靠开源数据复刻出LLaMA3指令学习效果，在线迭代RLHF全流程解决方案来了](https://www.jiqizhixin.com/articles/2024-05-18)


### 免微调对齐

- [大模型免微调解锁对话能力，RLHF没必要了！节省大量成本和时间，一作上交大校友](https://zhuanlan.zhihu.com/p/670682075)
  - URIAL, base model的免微调方法
- [OPO:无需训练实现价值观实时动态对齐：上交开源价值观对齐方法，闭源与开源大模型均适用](https://mp.weixin.qq.com/s/_CB0LBQVI_2NBiX63pyYSA)
  - OPO，收集相关法律或者道德准则，使用RAG检索与query相关的准则，基于检索结果来生成

## 基础知识

### GRPO如何为每个token分配advantage

- 参考 [blog](https://zhuanlan.zhihu.com/p/20812786520) ，通过组内优势估计计算每个 response 的相对优势 advantage，然后将 advantage 作为每个 token 的advantage。
- 而PPO算法的逐token的advantage是用逐token的及时奖励、值函数，通过GAE算法从后往前计算来的。GRPO没有值函数，所以无法使用GAE。











