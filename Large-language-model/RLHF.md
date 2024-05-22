<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Blog](#blog)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Resource

- https://github.com/lucidrains/self-rewarding-lm-pytorch

- https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat

- https://github.com/OpenLMLab/MOSS-RLHF

- https://github.com/SupritYoung/Zhongjing
  - 医疗领域，使用了rlhf

- https://huggingface.co/blog/trl-peft

- DPO 
  - [Fine-tune a Mistral-7b model with Direct Preference Optimization](https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac)

# Paper

## rlhf-related-algorithm

- Black-Box Prompt Optimization: Aligning Large Language Models without Model Training
  - 2024, BPO
  - BPO 的关键思想是通过优化用户输入的提示，使其更适合 LLMs 的输入理解，从而在不更新 LLMs 参数的情况下实现用户意图的最佳表达
  - 与使用人类反馈进行强化学习的方法相比，BPO 提供了更好的可解释性，因为它通过优化提示来改善模型的响应，而不是直接修改模型参数
  - BPO 使用一个自动的提示优化器，通过比较人类偏好的响应对来学习，并指导 LLMs 重写输入提示，使其更加明确地包含将响应从不受欢迎转变为受欢迎的特征

- LARGE LANGUAGE MODELS AS OPTIMIZERS
  - 2024, OPRO
  - 本文提出了一种名为OPRO的方法，它利用大型语言模型作为优化器来解决各种优化问题，特别是那些缺乏梯度信息的优化问题
  - 在OPRO框架中，优化任务通过自然语言描述，LLM基于此描述和先前生成的解决方案来迭代生成新的解决方案，并通过评估和反馈进行优化
  - 论文在线性回归、旅行商问题和提示优化等案例研究中展示了OPRO的有效性，证明了LLMs能够通过自然语言提示有效优化解决方案，甚至在某些情况下超过了手工设计的启发式算法

- ORPO: Monolithic Preference Optimization without Reference Model
  - 2024
  - ORPO的核心方法在于它不需要参考模型，并且可以在单一步骤中通过赋予不希望生成的风格一个小的惩罚，来高效地进行偏好对齐的监督式微调
  - ORPO通过在传统的负对数似然损失函数中加入一个基于赔率比（Odds Ratio）的惩罚项来区分优选和非优选的生成风格

- Aligning Large Language Models with Human Preferences through Representation Engineering
  - 2023，RAFT
  - RAHF通过识别和操作LLMs内部与高级人类偏好相关的表示和活动模式，来实现对模型行为的精确控制
  - 与RLHF相比，RAHF方法计算成本更低，因为它不需要额外训练奖励模型和价值网络
  - RAHF包含两种方法来控制表示和提取活动模式：单一LLM方法和双LLM方法。单一LLM方法通过对比指令来微调单一模型，而双LLM方法则分别对偏好和不偏好的响应进行监督训练

- Preference Ranking Optimization for Human Alignment
  - 2023.06，AAAI 2024
  - 出了一种偏好排序方法，即采样不同源的回复并使用reward打分，并基于打分结果排序，然后计算首个最佳回复与其余回复的InfoNEC loss，然后drop掉最佳回复，然后使用第二位最佳回复与其余回复的InfoNCE loss，然后drop第二位最佳回复，重复过程，知道drop掉所有回复。将所有的infoNCE loss相加，作为loss1。
  - 选择首个最佳回复计算SFT loss，作为loss2。总loss未loss1和loss2的和。

- RRHF: Rank Responses to Align Language Models with Human Feedback without tears
  - 2023.04， NeurIPS 2023
  - 采样不同源的回复，并用reward打分。使用ranking loss + best response sft loss最为total loss

- RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment
  - 2023.04， TMLT
  - 使用reward模型过滤出高质量样本，然后使用高质量的样本微调模型

- Unveiling the Implicit Toxicity in Large Language Models
  - year: 2023
  - 研究者们提出了一种基于强化学习的攻击方法，旨在进一步诱导LLMs生成隐性有毒的文本

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
  - 2023, DPO
  - DPO通过一个简单的分类损失函数直接优化策略，而不是先拟合一个奖励模型，然后使用强化学习来最大化这个奖励
  - 与RLHF相比，DPO算法更稳定、性能更好，且计算成本更低。它不需要在微调期间从LM中采样，也不需要进行大量的超参数调整
  - DPO方法背后的理论基础是，存在一个从奖励函数到最优策略的解析映射，这使得研究者能够将基于奖励的损失函数转换为直接针对策略的损失函数

- A General Language Assistant as a Laboratory for Alignment
  - 2021, Anthropic
  - 偏好模型是在序列的最后一个token上加value head，value head负责预测一个标量值来表示得分；模仿学习是只用good example来微调模型
  - 排序偏好模型相比二进制偏好模型有更好的效果
  - context distillation: prompt会减少输入的长度等缺点，使用了一种基于KL的loss来对prompt微调
  - 偏好模型预训练的第二个阶段，使用二进制判别的预训练方法相比排序偏好方法有更好的收益

## reward model

- RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment
  - 2024，ICML，RLCD
  - 论文提出基于positive prompt和negative prompt来生成对比性强、质量好的偏好对，然后训练reward模型，接下来的PPO训练部分与常见方案相同
  - 不同于RLAIF，仅使用同一个prompt生成两个回复并打分，RLCD是使用两个对比prompt生成回复。

- Secrets of RLHF in Large Language Models Part II: Reward Modeling
  - 2024, 复旦大学
  - 提出了一种基于多奖励模型投票机制的方法来衡量数据中偏好的强度。这有助于区分数据集中的错误、模糊和正常偏好，并相应地调整标签，以减少模型对低质量数据的过度拟合
  - 引入对比学习来增强奖励模型区分被选择和被拒绝响应的能力，从而提高模型的泛化能力
  - 使用元学习来保持奖励模型对分布外（OOD）样本的微妙差异的区分能力，这可以用于迭代的RLHF优化
  - 基于偏好强度，引入自适应边界到奖励模型的损失函数中，使得模型更容易区分相似的响应
  - 为了减轻数据集中噪声的影响，提出了标签翻转和标签平滑技术，这些技术可以有效地减轻偏好噪声，并提高性能
  - 从数据角度出发，通过提出的方法来减轻错误和模糊偏好对的影响，并充分利用高质量偏好数据。从算法角度出发，通过对比学习和元学习方法来提高奖励模型的泛化能力

- Self-Rewarding Language Models
  - year：2024
  - 通过大模型生成回复，并用大模型自身对生成的回复进行打分
  - 基于打分结果筛选得分最高和最低的回复作为偏好数据对，然后使用DPO进行训练，相比直接用最高分数据微调的模型效果要好
  - 以上训练过程会经过多次迭代，每次迭代会用到之前创建的数据
  - [Meta发布自我奖励机制，Llama在3轮训练后超越GPT-4](https://zhuanlan.zhihu.com/p/680274984)




