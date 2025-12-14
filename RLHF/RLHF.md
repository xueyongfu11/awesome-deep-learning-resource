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
  

## on-policy RLHF（在线强化学习）

- Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models

  - 2025.06，RLSC方法
  - 这是一种无需人类标注、外部奖励模型或手动奖励设计的强化学习框架，其核心是利用语言模型自身的置信度作为奖励信号，通过 “模态锐化” 优化模型输出分布以提升置信度。
  - 模态锐化（Mode Sharpening）：多数投票本质是选择输出分布的众数，隐含优化目标是提升众数的概率质量，RLSC 将此目标转化为可微分的自监督目标，直接利用模型自身置信度作为奖励信号，无需外部监督。
  - RLSC 的优化目标是最大化 两个独立样本$y_1$, $y_2$相同的概率期望： $$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y_1 \sim p_\theta(y|x), y_2 \sim p_\theta(y|x)} \left[ \mathbb{I}(y_1 = y_2) \right]$$ ，根据概率的基本性质，“两个独立样本相同的概率期望”可展开为：  $$\mathbb{E}_{x,y_1,y_2} \left[ \mathbb{I}(y_1=y_2) \right] = \mathbb{E}_x \left[ \sum_y p_\theta(y|x)^2 \right]$$，这个展开式的关键价值是：去掉了不可微分的指示函数，替换为可微分的概率平方和。

- TTRL: Test-Time Reinforcement Learning

  - 2025.04，TTRL方法
  - LLM先处理提示x，从策略πθ(y|x)中采样生成N个候选输出{y₁, y₂, ..., yₙ}。通过多数投票确定共识答案y*，再按与y*的匹配情况（匹配得1分、不匹配得0分）为各候选输出计算奖励，最后用PPO或GRPO等RL算法更新模型参数θ，以最大化预期奖励。
  
- Training language models to follow instructions with human feedback
  - 2022.03，OpenAI
  - 使用人工编写的prompt数据，基于GPT3模型进行再训练，得到一个增强的预训练语言模型
  
  - 基于1中训练好的预训练语言模型以及构建好的prompt集，使用beam-search等生成prompt的多个回复结果，然后人工对生成的多个结果排序，然后基于该数据训练了一个6B的打分模型，使用的是pair-wise的ranking loss
  
  - 基于PPO算法训练强化学习模型，模型使用1中预训练好的模型进行初始化，给定prompt生成输出，然后用2中打分模型计算得分，然后优化强化学习模型。然后再使用强化学习模型生成，再打分。优化目标函数中添加了自回归语言模型的loss
  
- PPO in RLHF

  - [RLHF中PPO原理与源码解读](https://mp.weixin.qq.com/s/J8c7rEmkQH4lBj1pWntv9w)

  - [PPO原理解读和重要性采样](https://www.cnblogs.com/xingzheai/p/15931681.html)

  - 逐token的奖励值如何计算？

    - 假设EOS终止步长为T，当token的pos小于T时，奖励值为0，当大于T时，奖励值为奖励模型计算的序列级的奖励。

    - 很多论文并未使用上述逐token的奖励值，而是使用带KL penalty的逐token奖励值。即在上述奖励值的基础上，添加一项token级别的KL penalty项。
      $$
      r_t =
      \begin{cases}
      -\beta \left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t) \right), 
      & t < T, \\[6pt]
      R_{\mathrm{RM}}(x, y)
      -\beta \left( \log \pi_\theta(a_T \mid s_T) - \log \pi_{\mathrm{ref}}(a_T \mid s_T) \right),
      & t = T.
      \end{cases}
      $$

  - 逐token 的advantage如何计算

    - 逐token的advantage是用逐token的及时奖励、值函数，通过GAE算法从后往前计算来的


## off-policy RLHF（离线强化学习）

- Reasons to Reject? Aligning Language Models with Judgments

  - 2024

  - 该研究首次将 “拒绝理由监督” 融入对比训练框架：当模型生成低质量输出时，不仅提供优质回答作为正样本，还附加具体的批评性评语（如“逻辑矛盾”“信息缺失”）作为负样本标注。这种设计使模型同时学习“应该生成什么”和“应该避免什么”，解决了传统SFT仅优化正样本导致的“好坏不分”问题。

  - 论文提出的双向对比损失函数：通过传统SFT损失最大化优质回答的生成概率

    正样本赔率：$\text{odds}(y_w|x) = \frac{P(y_w|x)}{1-P(y_w|x)}$，负样本赔率：$\text{odds}(y_l|x) = \frac{P(y_l|x)}{1-P(y_l|x)}$ ，最大化两者的差值$\log \text{odds}(y_w|x) - \log \text{odds}(y_l|x)$ 

    $$L_{OR} = -\log \sigma\left( \log \text{odds}(y_w|x) - \log \text{odds}(y_l|x) \right)$$ 

    总损失：$L_{SFT} + \lambda \cdot L_{OR}$

- ORPO: Monolithic Preference Optimization without Reference Model
  - 2024.03, ORPO, 无需参考模型
  - ORPO的核心方法在于它不需要参考模型，并且可以在单一步骤中通过赋予不希望生成的风格一个小的惩罚，来高效地进行偏好对齐的监督式微调
  - ORPO通过在传统的负对数似然损失函数中加入一个基于赔率比（Odds Ratio）的惩罚项来区分优选和非优选的生成风格
  
- Contrastive Instruction Tuning

  - 2024.02

- Reasons to Reject? Aligning Language Models with Judgments

  - 2024

  - 该研究首次将 “拒绝理由监督” 融入对比训练框架：当模型生成低质量输出时，不仅提供优质回答作为正样本，还附加具体的批评性评语（如“逻辑矛盾”“信息缺失”）作为负样本标注。这种设计使模型同时学习“应该生成什么”和“应该避免什么”，解决了传统SFT仅优化正样本导致的“好坏不分”问题。

  - 论文提出的双向对比损失函数：通过传统SFT损失最大化优质回答的生成概率

    正样本赔率：$\text{odds}(y_w|x) = \frac{P(y_w|x)}{1-P(y_w|x)}$，负样本赔率：$\text{odds}(y_l|x) = \frac{P(y_l|x)}{1-P(y_l|x)}$ ，最大化两者的差值$\log \text{odds}(y_w|x) - \log \text{odds}(y_l|x)$ 

    $$L_{OR} = -\log \sigma\left( \log \text{odds}(y_w|x) - \log \text{odds}(y_l|x) \right)$$ 

    总损失：$L_{SFT} + \lambda \cdot L_{OR}$

- Direct Preference Optimization with an Offset
  
  - 2024.02，DPO改进
  
  - 将accept和reject的reward差值考虑了进去，相当于在DPO损失公式中添加一个margin
  
  - code：https://github.com/rycolab/odpo
  
- KTO: Model Alignment as Prospect Theoretic Optimization
  - 2024.02, 无需偏好数据

  - 相比DPO，KTO不需要成对的偏好数据，而是直接使用point wise的数据微调

  - 在一个batch中需要同时包含accept样本和reject样本

  - KTO之所以有效，是因为如果模型提高了accept样本奖励，那么KL惩罚也会上升，而在损失上就不会取得进展。这迫使模型学习究竟是什么使得输出变得理想，以便在保持KL项不变（甚至减少它）的同时增加奖励

  - <img src="../assets/KTO.png" style="zoom: 67%;" />
  
- Aligning Large Language Models with Human Preferences through Representation Engineering
  - 2023.12，RAFT
  - RAHF通过识别和操作LLMs内部与高级人类偏好相关的表示和活动模式，来实现对模型行为的精确控制
  - 与RLHF相比，RAHF方法计算成本更低，因为它不需要额外训练奖励模型和价值网络
  - RAHF包含两种方法来控制表示和提取活动模式：单一LLM方法和双LLM方法。单一LLM方法通过对比指令来微调单一模型，而双LLM方法则分别对偏好和不偏好的响应进行监督训练
  
- Some things are more CRINGE than others:  Iterative Preference Optimization with the Pairwise Cringe Loss
  - 2023.12
  - 基于Cringe Loss改进对偏好数据进行训练，是一种DPO方法平替
  - Binary Cringe Loss：对chosen样本计算似然损失，对与rejected样本计算token-wise的对比loss，具体是基于LLM的top-k token与rejected样本的token计算对比损失
  - Pair Cringe Loss：是对2的改进，增加了一项基于门控的margin loss

- Unveiling the Implicit Toxicity in Large Language Models
  - 2023.11, EMNLP2024
  - 研究者们提出了一种基于强化学习的攻击方法，旨在进一步诱导LLMs生成隐性有毒的文本
  
- Black-Box Prompt Optimization: Aligning Large Language Models without Model Training
  - 2023.11, BPO, ACL2024，免微调对齐
  - BPO 的关键思想是通过优化用户输入的提示，使其更适合 LLMs 的输入理解，从而在不更新 LLMs 参数的情况下实现用户意图的最佳表达
  - 与使用人类反馈进行强化学习的方法相比，BPO 提供了更好的可解释性，因为它通过优化提示来改善模型的响应，而不是直接修改模型参数
  - BPO 使用一个自动的提示优化器，通过比较人类偏好的响应对来学习，并指导 LLMs 重写输入提示，使其更加明确地包含将响应从不受欢迎转变为受欢迎的特征
  
- Statistical Rejection Sampling Improves Preference Optimization
  - 2023.09，ICML2024
  - DPO算法所使用的数据是SFT或者其他算法采样出来的，而不是最优策略采样出来的
  - 想要估计某一个分布，需要用这个分布下采样出来的数据才能很好地对分布进行估计
  - DPO使用其他策略采样出的数据计算MLE去估计最优策略，会导致数据和最优策略之间的不匹配
  - [LLM RLHF 2024论文（三）RSO](https://zhuanlan.zhihu.com/p/690198669)
  
- LARGE LANGUAGE MODELS AS OPTIMIZERS
  - 2023.09, ICML2024, OPRO, 免微调对齐
  - 本文提出了一种名为OPRO的方法，它利用大型语言模型作为优化器来解决各种优化问题，特别是那些缺乏梯度信息的优化问题
  - 在OPRO框架中，优化任务通过自然语言描述，LLM基于此描述和先前生成的解决方案来迭代生成新的解决方案，并通过评估和反馈进行优化
  - 论文在线性回归、旅行商问题和提示优化等案例研究中展示了OPRO的有效性，证明了LLMs能够通过自然语言提示有效优化解决方案，甚至在某些情况下超过了手工设计的启发式算法
  
- Preference Ranking Optimization for Human Alignment
  - 2023.06，AAAI 2024，基于排序的方法
  - 出了一种偏好排序方法，即采样不同源的回复并使用reward打分，并基于打分结果排序，然后计算首个最佳回复与其余回复的InfoNEC loss，然后drop掉最佳回复，然后使用第二位最佳回复与其余回复的InfoNCE loss，然后drop第二位最佳回复，重复过程，知道drop掉所有回复。将所有的infoNCE loss相加，作为loss1。
  - 选择首个最佳回复计算SFT loss，作为loss2。总loss未loss1和loss2的和。
  
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
  - 2023.05, DPO
  
  - DPO通过一个简单的分类损失函数直接优化策略，而不是先拟合一个奖励模型，然后使用强化学习来最大化这个奖励。与RLHF相比，DPO算法更稳定、性能更好，且计算成本更低。它不需要在微调期间从LM中采样，也不需要进行大量的超参数调整
  
  - DPO方法背后的理论基础是，存在一个从奖励函数到最优策略的解析映射，这使得研究者能够将基于奖励的损失函数转换为直接针对策略的损失函数
  
    - 通过RLHF的目标函数，推到出奖励函数和最优策略与参考策略之对数比的数学关系。
  
    - 为了利用偏好数据训练模型，希望生成高质量的回复概率大于低质量回复的概率。Bradley-Terry模型可以用来计算两个对象，其中一个对象比另外一个对象强的概率。因此可以计算高质量回复的奖励值强于低质量回复的奖励值的概率。
  
    - 将强度概率公式中的奖励值使用“最优策略与参考策略之对数比”进行替代，便得到DPO的目标公式。
      $$
      \mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}})
      = -\mathbb{E}_{(x, y_w, y_l) \sim D} \bigg[
          \log \sigma\!\left(
              \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
              \;-\;
              \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
          \right)
      \bigg]
      $$
      其中$\beta$用来调节模型对好坏答案的区分强度，该参数与RLHF中的KL散度的$\beta$等价
  
  - Blog：
  
    - [大语言模型对齐: 直接偏好优化(DPO)](https://syhya.github.io/zh/posts/2025-02-08-dpo/)
  
    - [有关DPO训练时，为什么chosen和rejected的reward一起下降的猜想](https://zhuanlan.zhihu.com/p/694381064)
  
- RRHF: Rank Responses to Align Language Models with Human Feedback without tears
  - 2023.04， NeurIPS 2023，基于排序的方法
  - 采样不同源的回复，并用reward打分。使用ranking loss + best response sft loss最为total loss
  
- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment
  - 2023.04，TMLT
  - 使用reward模型过滤出高质量样本，然后使用高质量的样本微调模型
  
- A General Language Assistant as a Laboratory for Alignment
  - 2021.12,  from Anthropic
  - 偏好模型是在序列的最后一个token上加value head，value head负责预测一个标量值来表示得分；模仿学习是只用good example来微调模型
  - 排序偏好模型相比二进制偏好模型有更好的效果
  - context distillation: prompt会减少输入的长度等缺点，使用了一种基于KL的loss来对prompt微调
  - 偏好模型预训练的第二个阶段，使用二进制判别的预训练方法相比排序偏好方法有更好的收益


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











