[TOC]



## reward model

### resource

- https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark
  - Reward Model Benchmark

- https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/
  - reward model的训练是通过构建多个aspect的回归loss
- reward model benchmark
  - https://huggingface.co/spaces/allenai/reward-bench
- [Reward Modeling for RLHF](https://efficient-unicorn-451.notion.site/Reward-Modeling-for-RLHF-abe03f9afdac42b9a5bee746844518d0)
  - 收集了几乎所有的开源偏好数据进行reward model的训练
  - 模型在 AlpacaEval 榜单上排名第二
- [Llama2的reward model](https://zhuanlan.zhihu.com/p/679012951)
  - reward model的损失函数中加入了margin
  - reward model的推理结果进行了WHITEN，即归一化操作，减去均值，除标准差
- https://github.com/OpenLMLab/MOSS-RLHF
- Towards Understanding the Influence of Reward Margin on Preference Model Performance

### generative reward model

- Beyond Scalar Reward Model: Learning Generative Judge from Preference Data
  - 2024.10
  - Con-J是一种生成性评判模型，通过让LLM生成带有理由的正负评判，利用这些对比性评判对进行DPO训练，提高了模型的可解释性和对数据偏见的鲁棒性
- Generative Verifiers: Reward Modeling as Next-Token Prediction
  - 2024.10
  - 以SFT和CoT-SFT的方式，在prompt+response基础上添加如“Is the answer correct (Yes/No)?”的问题片段，然后以next-token prediction的推理方式，计算Yes/No的概率值，作为奖励值
- Generative Reward Models
  - 2024.10
  - 论文提出了GenRM和CoT-GenRM。GenRM使用”下一个token预测“的方式，计算偏好选项的概率。相比传统添加value header的方法，GenRM未改变模型结构，而是以大模型自生成的方式。
  2. GenRM以DPO方法进行训练。CoT-GenRM是在推理出偏好选项之前，先生成偏好对比的思维链依据。
- Direct Judgement Preference Optimization
  - 2024.9
  - 通过结合正面和负面的偏好数据优化生成型评估模型的评估能力
  2. 三种训练任务：Chain-of-Thought Critique：通过逐步推理来生成详细的语言反馈和最终判断；Standard Judgement：仅生成最终判断，剔除语言反馈，以提供更直接的监督信号；Response Deduction：通过模型的评估推断原始的模型输出，以加深对高质量和低质量响应的理解
  3. 综合采用DPO和SFT损失函数，在构造数据时，利用强大的教师模型生成高质量的偏好数据，并使用弱模型生成负例
- Critique-out-Loud Reward Models
  - 2024.8
  - 提出了CLoud，首先训练模型生成回复的评论信息，然后将prompt，response，回复的评论信息作为输入，输出reward value

### paper-reward model

- HAF-RM: A Hybrid Alignment Framework for Reward Model Training
  - 2024.07

  - 提出了一种reward model的混合对齐框架，通过共享backbone，header由value header（奖励模型的header）和prob header组成（生成的header）

  - 具体实现是由DPO的loss和奖励模型的loss组成
- DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging
  - 2024.07
  - 提出了DogeRM，将通用的reward model与领域模型进行权重加权，从而使得通用reward model表现出领域reward model的效果

- Boosting Reward Model with Preference-Conditional Multi-Aspect Synthetic Data Generation
  - 2024.07，
  - 方法基于RLCD的改进，探索了基于条件生成偏好数据的方法
  - 具体是模型先生成回复，然后基于该回复生成另外一条回复
- Exploring Domain Robust Lightweight Reward Models based on Router Mechanism
  - 2024.07
  - 核心方法是探索基于路由器机制的领域鲁棒轻量级奖励模型
- Learning Goal-Conditioned Representations for Language Reward Models
  - 2024.07

  - 对比学习目标条件化：通过增加未来状态沿着采样的偏好轨迹的表示相似度，并减少沿着随机采样的不受欢迎轨迹的相似度，来训练奖励模型
- Interpretable Preferences via Multi-Objective Reward Modeling  and Mixture-of-Experts
  - 2024.06
  - 为了解决reward model不可解释性的问题，提出了ArmoRM模型，具体是在last token接一个多目标回归的header层，多目标对应偏好判断的不同方面
  - 为了将多方面的偏好值加权，提出了一种门控网络，该门控网络的输入是prompt的last token的hidden  state，输出是经过softmax的加和为1的权重值，与多方面的偏好值相乘之后得到总奖励值。门控网络的训练是冻结除了门控网络的其他所有权重，使用bradley-terry目标函数进行简介训练
- Preference Learning Algorithms Do Not Learn Preference Rankings
  - 2024.05，google，
  - 现有的模型很难实现高的ranking accuracy，一般低于60%
  - 现有的模型的ranking accuracy低于理想的ranking accuracy，19%-51%的gap
  - DPO这种偏好学习方法很少会纠正数据中的标签，更多是增大偏好回复与非偏好回复的log-prob
  - 在policy model和reference model未偏离太多的条件下，ranking accuracy和win rate两种评价指标是接近的
- Secrets of RLHF in Large Language Models Part II: Reward Modeling
  - 2024.01, 复旦大学
  - 提出了一种基于多个奖励模型投票机制的方法来衡量数据中偏好的强度。这有助于区分数据集中的错误、模糊和正常偏好，并据此对错误的偏好标签进行纠正以及label smoothing
  - 引入了对比学习来增强奖励模型区分被选择和被拒绝响应的能力，从而提高模型的泛化能力
  - 采用元学习来使奖励模型保持对分布外样本的微妙差异的区分能力，这可以用于迭代的RLHF优化
  - [深挖RLHF潜力，复旦语言和视觉团队创新奖励模型优化，让大模型更对齐](https://mp.weixin.qq.com/s/BSaGLikARlvM8yitYtlA3w)
- SALMON: Self-Alignment with Instructable Reward Models
  - 2023.10, ICLR2024
  - 提出了指令性reward model，可以基于任意的人类准则来生成相应的奖励得分
- RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
  - 2023.09
  - 方法旨在解决传统通过人类反馈进行强化学习中的一个关键瓶颈问题：获取高质量的人类偏好标签
  - 偏好标记：使用现成的LLM为一对候选摘要打上偏好标签。然后，使用对比损失训练一个奖励模型，最后使用RM提供的奖励进行策略模型的强化学习微调
- RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment
  - 2023.07，ICML2024，RLCD
  - 论文提出基于positive prompt和negative prompt来生成对比性强、质量好的偏好对，然后训练reward模型，接下来的PPO训练部分与常见方案相同
  - 不同于RLAIF，仅使用同一个prompt生成两个回复并打分，RLCD是使用两个对比prompt生成回复。

### paper-reward hacking

- Spontaneous Reward Hacking in Iterative Self-Refinement
  - 2024.07

  - 本文主要研究了基于大模型的生成器和评估器的自我迭代的框架中，由于基于大模型的评估器并不能代表人类真实的判断意图，造成存在一定的reward hacking问题

  - 当生成器和评估器共享同一个大模型时，这种reward hacking问题会变得更加严重
  - 通过一篇论文编辑任务，展示了迭代自我完善如何导致评估者和人类判断之间出现自发的偏差。研究了奖励黑客攻击发生的条件，并观察了影响其严重性的两个因素：模型大小和生成器与评估者之间的上下文共享

- SCALABLE ENSEMBLING FOR MITIGATING REWARD  OVEROPTIMISATION
  - 2024.06, ICLR2024

  - 提出了一种高效的reward model ensemble方法，即使用共享的encoder，组合中的每个reward model拥有自己的linear head来计算reward value

- Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms
  - 2024.06

  - 该论文主要研究了直接偏好算法如DPO、IPO的reward model overoptimization问题，不同于PPO中的reward model overoptimization问题

  - 实验发现，直接偏好算法同传统的RLHF，也存在reward model overoptimization问题
  - 实验发现直接对齐算法不仅在正常的KL范围内性能会恶化，而且往往在完成数据集的哪怕一个训练周期之前就已经出现性能下降。
  - 论文展示了直接偏好算法中的奖励建模目标是严重欠约束，在训练过程中可能会对训练数据中未出现过的、分布外的样本给予过高的概率估计
  - 论文研究了不同模型如DPO、IPO、SLiC的WinRate、KL、eval acc、loss等之间的关系

- Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs
  - 2024.06
  - 提出了通过正则化hidden state的方法来提高reward model的泛化能力
  - 该正则化的提出背景：传统的reward model的训练通常随机初始化分类header，这种方式会扭曲预训练权重特征
  - 该正则化的具体实现：计算reward model loss的同时，添加sft的loss同时训练，训练时对sft的header进行freeze，对backbone以及reward model header进行训练
- Regularized Best-of-N Sampling to Mitigate Reward Hacking for  Language Model Alignment
  - 2024.04
  - 通过正则化最佳N采样（Regularized Best-of-N，简称RBoN）来减轻大型语言模型在解码时对奖励模型的过度优化问题，即奖励黑客攻击问题

- Fine-Tuning Language Models with Reward Learning on Policy
  - 2024.03, NAACL2024, RLP, 解决reward model hacking问题
  - reward model的效果随着policy model的优化出现不准确的分布偏移，常用的方法是从policy model中重新采样、标注，训练新的reward model
  - RLP方法不需要重新采样数据训来练新reward model，提出了一种无监督的reward model微调方法，从而避免的分布偏移
  - 具体是使用了无监督的multi-view表示学习方法，来学习policy model的采样样本。二是提出了合成偏好数据的生成方法，进一步微调reward model。然后基于这两种方法微调reward model
- InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling
  - 2024.02

  - 信息瓶颈（Information Bottleneck, IB）目标：通过引入变分信息瓶颈目标，InfoRM能够在保留与人类偏好相关的信息的同时，过滤掉与偏好无关的冗余信息

  - 奖励过度优化检测：论文发现奖励过度优化与IB潜在空间中的异常值之间存在相关性，并基于此提出了簇分离指数，用于量化IB潜在空间中的偏差，作为奖励过度优化的指标。
- ODIN: Disentangled Reward Mitigates Hacking in RLHF
  - 2024.02

  - 本文主要研究reward hacking中最常见的回复长度问题，提出了一种公平的权衡score和response length的评估方法，本质是基于改进prompt的模型评估方法

  - 通过大量的实验，验证了几个超参设置对长度偏置的影响，比如KL loss系数、长度惩罚项、RM clip、PPO clip、从old policy采样数据等
  - 提出了一种改进的RM算法，ODIN，即使用length header和content header，推理时，只使用content header的奖励值
  - ODIN如何训练：首先可以容易构建Length Loss和Rank Loss，为了解耦出content  Loss，构建了一个正交Loss，即length header和content header权重的乘积，来间接的训练content  header。为了防止header权重为0，使用了weight norm。
- Improving Reinforcement Learning from Human Feedback with Efficient Reward Model Ensemble
  - 2024.01

  - 现有reward ensemble方法计算成本和资源消耗成本较高，因此提出了两个方法linear-layer ensemble和lora-based ensemble

  - linear-layer ensemble是使用共享的backbone，组合中的每个模型使用自己的reward header；
  - lora-based ensemble是组合中的每个模型使用自己的lora层，训练时先用部分数据基于linear-layer ensemble方法训练，然后再使用剩下的数据基于lora-based ensemble方法训练
  - 使用时提出了两种方法，一种是对奖励值取平均，第二种是计算lower confidence bound (LCB)
- Iterative Data Smoothing: Mitigating Reward Overfitting and  Overoptimization in RLHF
  - 2024.01

  - 为了缓解reward overoptimization，从理论视角设计了改进版的RM算法，即IDS

  - IDS的核心思想是，在每一个epoch训练期间，不仅用数据更新模型，还要用模型来更新数据，即使用soft labels来替代hard labels
  - 悲观最大似然估计（pessimistic MLE）通过降低对较少被选择的数据的估计奖励，有助于缓解奖励过度优化的问题。而IDS通过更新我们所训练数据的标签来实现这一点
- WARM: On the Benefits of Weight Averaged Reward Models
  - 2024.01
  - 引入了权重平均奖励建模的首个实例 WARM，可缓解奖励破解、提高分布变化下的可靠性和对标签损坏的鲁棒性。
  - 发现权重平均和预测平均的关键差异，权重平均能保持不变的预测机制，减少记忆（比如标签错误的训练样本），更关注可泛化特征
- Helping or Herding?  Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking
  - 2023.12, COLM
  - reward model overoptimization可以使用reward model组合的方式进行缓解
  - 使用不同预训练seed的ensemble方法相比使用不同微调seed的ensemble方法的效果更好，但是仍然不能消除reward model  hacking问题，经过实验探究，ensemble的reward model展示除了一些相似的错误pattern
- REWARD MODEL ENSEMBLES HELP MITIGATE  OVEROPTIMIZATION
  - 2023.10，ICML2024
  - 提出使用多个模型组合的方式来缓解reward model的过优化问题
  - 多个reward model的组合，使用WCO和UWO，相比计算均值的方式效果更好
  - 论文也研究了RM的size、数据size、组合模型的数据等对效果的影响
- Confronting Reward Model Overoptimization with Constrained RLHF
  - 2023.10

  - 论文通过实验确定了复合奖励模型的过度优化问题，这些组成部分之间的相关性对优化点的位置有显著影响。优化点是超过了该位置之后，proxy reward上升，ground truth reward下降。

  - 为了解决过度优化问题，论文提出了一种使用约束强化学习的方法。这种方法通过防止代理超过每个奖励模型的有用性阈值，来防止过度优化。论文提出的方法通过学习动态权重来解决组成部分奖励模型的权重问题，这些权重自然由拉格朗日乘数表示。
  - 为了在单次运行中识别和优化这些点，论文引入了一种使用无梯度优化的方法。这种方法可以在训练过程中动态地找到这些代理点，显著节省计算资源
- Scaling Laws for Reward Model Overoptimization
  - 2022.10
  - 主要研究了RM model的size，Policy model 的size，RM的训练集size等对reward model overoptimization的影响
  - 评估方法是随着KL的增加，计算RM model的score与Gold RM model的score的差异。KL增加，表明policy model与initial model差异更大，采样到的数据标注时，越容易hacking RM model