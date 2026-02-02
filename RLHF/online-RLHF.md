[TOC]



# Online RLHF

- Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models

  - 2025.06，RLSC方法
  - 这是一种无需人类标注、外部奖励模型或手动奖励设计的强化学习框架，其核心是利用语言模型自身的置信度作为奖励信号，通过 “模态锐化” 优化模型输出分布以提升置信度。
  - 模态锐化（Mode Sharpening）：多数投票本质是选择输出分布的众数，隐含优化目标是提升众数的概率质量，RLSC 将此目标转化为可微分的自监督目标，直接利用模型自身置信度作为奖励信号，无需外部监督。
  - RLSC 的优化目标是最大化 两个独立样本$y_1$, $y_2$相同的概率期望： $$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y_1 \sim p_\theta(y|x), y_2 \sim p_\theta(y|x)} \left[ \mathbb{I}(y_1 = y_2) \right]$$ ，根据概率的基本性质，“两个独立样本相同的概率期望”可展开为：  $$\mathbb{E}_{x,y_1,y_2} \left[ \mathbb{I}(y_1=y_2) \right] = \mathbb{E}_x \left[ \sum_y p_\theta(y|x)^2 \right]$$，这个展开式的关键价值是：去掉了不可微分的指示函数，替换为可微分的概率平方和。

- VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks

  - 2025.04

  - 不同GRPO、DAPO，VAPO是一种基于值模型的方法，并超越了无值模型的方法

  - 作者提出值模型的预训练，认为先前方法效果不好的原因是值模型权重从奖励模型初始化，但是二者目标并不匹配。作者从初始策略模型中采样数据，使用蒙特卡洛回报更新值模型。

  - 论文使用解耦的GAE方法，policy model和value model使用不同的时间衰减系数。在policy model中，advantage作为概率比的加权值，在value model中，advantage与老的值网络的和作为target，与新的值网络之间计算MSE损失。VC-PPO任务将时间衰减系数设置为1，是一种无偏的梯度估计。因此VAPO的policy model的advantage计算采用的时间衰减系数为1，而value model的advantage计算采用的时间衰减系统是0.9

  - 使用长度自适应的GAE。固定的时间衰减系数下，当长度很长时，奖励几乎为0，论文提出的长度自适应的GAE可以根据长度来调整时间衰减系数，使得优势函数中累积的学习信号随着生成长度线性增长。使用了DAPO中token级别的策略梯度损失，进一步缓解异常的序列长度问题。

  - 使用更高的上裁剪值；增加LM损失；组采样时，尽可能采样出有区分度的正负样本。

- TTRL: Test-Time Reinforcement Learning

  - 2025.04，TTRL方法
  - LLM先处理提示x，从策略πθ(y|x)中采样生成N个候选输出{y₁, y₂, ..., yₙ}。通过多数投票确定共识答案y，再按与y的匹配情况（匹配得1分、不匹配得0分）为各候选输出计算奖励，最后用PPO或GRPO等RL算法更新模型参数θ，以最大化预期奖励。

- DAPO: an open-source LLM reinforcement learning system at scale

  - 2025.03

  - 作者研究发现，使用GRPO的默认参数设置训练出的模型效果远低于DS-R1论文中报告的结果。作者发现在训练过程中，存在熵坍塌、奖励噪声、训练不稳定等问题。

  - 将上裁剪值和下裁剪值解耦，使用更高的上裁剪值，帮助模型进行更好的探索，提升多样性，避免熵坍塌

  - 使用动态采样方法提升训练效率和稳定性，摒弃组内采样奖励值都为0或者1的数据，因为这样的数据计算的advantage都为0

  - 为避免样本级 GRPO 在 Long-CoT 场景下对长序列 token 学习信号的过度稀释及对冗余模式惩罚不足，作者将策略梯度损失由 sample-level 重构为 token-level，从而实现对推理关键 token 的精细化优化与对无意义生成的有效抑制。

  - 对于过长的样本一般采用截断处理，这种方法会引入奖励噪声和损害训练过程。论文提出长度过滤方法，通过定义过长的长度区间，在该区间中，模型回复越长惩罚越大。

- REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models

  - 2025.01，
  - 介绍reinforce++算法之前先对比常见的几个value-free方法
    - RLOO/GRPO算法对一个prompt采样多条回复并打分，GRPO使用同prompt的RM Scores计算均值和方差，对每个prompt-response pair的得分进行归一化，RLOO则使用了leave-one-out，归一化之除以自己以外的样本的均值，避免了一些偏差和耦合，相比GRPO，RLOO未除以std。
    - ReMax对每个prompt做一次greedy解码，使用greedy解码的结果作为baseline
  - 论文认为以上方法中，baseline按prompt单独结算，会让优化倾向于把每个训练prompt都推向最高的reward，对更简单更短的prompt来说容易过拟合；同prompt多采样会降低batch内样本多样性，加速reward hacking
  - reinforce++对于每个prompt只采样一个回复，然后计算token-level的奖励值，具体是计算每个token的预期未来奖励（将奖励模型输出的last token的奖励值折现到每个时间步），减去每个token的KL penalty。
  - refinforce++不进行组内归一化，而是进行batch内归一化，计算batch内容所有的token-level奖励值的均值和标准差，最后直接归一化。

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

  - PPO中的clip操作是对概率比进行裁剪，防止学习的policy和采样数据的policy偏离太远

  - 逐token 的advantage如何计算：逐token的advantage是用逐token的及时奖励、值函数，通过GAE算法从后往前计算来的
  
  - critic model的loss：old critic model的状态值 + advantage 与 当前 critic model的状态值的MSE作为训练的损失。advantage可以理解为真实回报相比baseline（old critic model）状态值的增量。