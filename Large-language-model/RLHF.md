[TOC]



## resource

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
  

## rlhf-related-algorithm

- ORPO: Monolithic Preference Optimization without Reference Model
  - 2024.03, ORPO, 无需参考模型
  - ORPO的核心方法在于它不需要参考模型，并且可以在单一步骤中通过赋予不希望生成的风格一个小的惩罚，来高效地进行偏好对齐的监督式微调
  - ORPO通过在传统的负对数似然损失函数中加入一个基于赔率比（Odds Ratio）的惩罚项来区分优选和非优选的生成风格
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
  - DPO通过一个简单的分类损失函数直接优化策略，而不是先拟合一个奖励模型，然后使用强化学习来最大化这个奖励
  - 与RLHF相比，DPO算法更稳定、性能更好，且计算成本更低。它不需要在微调期间从LM中采样，也不需要进行大量的超参数调整
  - DPO方法背后的理论基础是，存在一个从奖励函数到最优策略的解析映射，这使得研究者能够将基于奖励的损失函数转换为直接针对策略的损失函数
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

## reward model

### resource

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

### paper

- HAF-RM: A Hybrid Alignment Framework for Reward Model Training
  - 2024.07
  
  - 提出了一种reward model的混合对齐框架，通过共享backbone，header由value header（奖励模型的header）和prob header组成（生成的header）

  - 具体实现是由DPO的loss和奖励模型的loss组成
  
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

- Fine-Tuning Language Models with Reward Learning on Policy
  - 2024.03, NAACL2024, RLP, 解决reward model hacking问题
  - reward model的效果随着policy model的优化出现不准确的分布偏移，常用的方法是从policy model中重新采样、标准，训练新的reward model
  - RLP方法不需要重新采样数据训来练新reward model，提出了一种无监督的reward model微调方法，从而避免的分布偏移
  - 具体是使用了无监督的multi-view表示学习方法，来学习policy model的采样样本。二是提出了合成偏好数据的生成方法，进一步微调reward model。
- Interpretable Preferences via Multi-Objective Reward Modeling  and Mixture-of-Experts
  - 2024.06
  - 为了解决reward model不可解释行的问题，提出了ArmoRM模型，具体是在last token接一个多目标回归的header层，多目标对应偏好判断的不同方面
  - 为了将多方面的偏好值加权，提出了一种门控网络，该门控网络的输入是prompt的last token的hidden  state，输出是经过softmax的加和为1的权重值，与多方面的偏好值相乘之后得到总奖励值。门控网络的训练是冻结除了门控网络的其他所有权重，使用bradley-terry目标函数进行简介训练
  
- Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs
  - 2024.06
  - 提出了通过正则化hidden state的方法来提高reward model的泛化能力
  - 该正则化的提出背景：传统的reward model的训练通常随机初始化分类header，这种方式会扭曲预训练权重特征
  - 该正则化的具体实现：计算reward model loss的同时，添加sft的loss同时训练，训练时对sft的header进行freeze，对backbone以及reward model header进行训练
  
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
- RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment
  - 2023.07，ICML2024，RLCD
  - 论文提出基于positive prompt和negative prompt来生成对比性强、质量好的偏好对，然后训练reward模型，接下来的PPO训练部分与常见方案相同
  - 不同于RLAIF，仅使用同一个prompt生成两个回复并打分，RLCD是使用两个对比prompt生成回复。



