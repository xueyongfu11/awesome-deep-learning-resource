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

- RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment
  - 2024，ICML，RLCD
  - 论文提出基于positive prompt和negative prompt来生成对比性强、质量好的偏好对，然后训练reward模型，接下来的PPO训练部分与常见方案相同
  - 不同于RLAIF，仅使用同一个prompt生成两个回复并打分，RLCD是使用两个对比prompt生成回复。

- Black-Box Prompt Optimization: Aligning Large Language Models without Model Training
  - 2024, BPO

- LARGE LANGUAGE MODELS AS OPTIMIZERS
  - 2024, OPRO

- Secrets of RLHF in Large Language Models
Part II: Reward Modeling
  - 2024

- ORPO: Monolithic Preference Optimization without Reference Model
  - 2024, ORPO

- Self-Rewarding Language Models
  - year：2024
  - 通过大模型生成回复，并用大模型自身对生成的回复进行打分
  - 基于打分结果筛选得分最高和最低的回复作为偏好数据对，然后使用DPO进行训练，相比直接用最高分数据微调的模型效果要好
  - 以上训练过程会经过多次迭代，每次迭代会用到之前创建的数据
  - [Meta发布自我奖励机制，Llama在3轮训练后超越GPT-4](https://zhuanlan.zhihu.com/p/680274984)

- Aligning Large Language Models with Human Preferences
through Representation Engineering
  - 2023

- Unveiling the Implicit Toxicity in Large Language Models
  - year: 2023
  - 提出了一种基于强化学习的方法，诱导LLM中的隐形毒性

- Direct Preference Optimization: Your Language Model is Secretly a Reward Model
  - 2023, DPO








