<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Repo](#repo)
- [Blog](#blog)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# Repo

- https://github.com/lucidrains/self-rewarding-lm-pytorch

- https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

- https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat

- https://github.com/OpenLMLab/MOSS-RLHF

- https://github.com/SupritYoung/Zhongjing
  - 医疗领域，使用了rlhf

- https://huggingface.co/blog/trl-peft


# Paper

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

# Blog

- [《大模型对齐方法》最新综述](https://zhuanlan.zhihu.com/p/686257781)

- [APO｜利用GAN的思想训练RLHF中的RM](https://zhuanlan.zhihu.com/p/674776494)

- [一些RLHF的平替汇总](https://zhuanlan.zhihu.com/p/667152180)

- [DEITA-大模型指令微调的数据高效筛选方法](https://zhuanlan.zhihu.com/p/675928711)

- [大模型人类对齐方法综述](https://mp.weixin.qq.com/s/Hzi5MtjsS6dk1br7DzJOGQ)

- [CML2023开会了！RLHF技术究竟是什么？167页HuggingFace等《通过人类反馈的强化学习（RLHF）》教程讲解](https://mp.weixin.qq.com/s/BX3m0c0NSuG6hesb_3gguw)

- [无需人类反馈即可对齐！田渊栋团队新作RLCD](https://mp.weixin.qq.com/s/sQolnpmBdCufVVR8q6GG8w)

- [大模型reward model的trick](https://mp.weixin.qq.com/s/G69w-Y2Jb_SgtvLcjCs_3g)


