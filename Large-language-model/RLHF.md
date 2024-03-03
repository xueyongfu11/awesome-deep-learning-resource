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

- Self-Rewarding Language Models
  - year：2024
  - 通过大模型生成回复，并用大模型自身对生成的回复进行打分
  - 基于打分结果筛选得分最高和最低的回复作为偏好数据对，然后使用DPO进行训练，相比直接用最高分数据微调的模型效果要好
  - 以上训练过程会经过多次迭代，每次迭代会用到之前创建的数据
  - [Meta发布自我奖励机制，Llama在3轮训练后超越GPT-4](https://zhuanlan.zhihu.com/p/680274984)

- Unveiling the Implicit Toxicity in Large Language Models
  - year: 2023
  - 提出了一种基于强化学习的方法，诱导LLM中的隐形毒性


# Blog

- [APO｜利用GAN的思想训练RLHF中的RM](https://zhuanlan.zhihu.com/p/674776494)

- [一些RLHF的平替汇总](https://zhuanlan.zhihu.com/p/667152180)

- [大模型免微调解锁对话能力，RLHF没必要了](https://zhuanlan.zhihu.com/p/670682075)

- [DEITA-大模型指令微调的数据高效筛选方法](https://zhuanlan.zhihu.com/p/675928711)

- [大模型人类对齐方法综述](https://mp.weixin.qq.com/s/Hzi5MtjsS6dk1br7DzJOGQ)

- [CML2023开会了！RLHF技术究竟是什么？167页HuggingFace等《通过人类反馈的强化学习（RLHF）》教程讲解](https://mp.weixin.qq.com/s/BX3m0c0NSuG6hesb_3gguw)

- [无需人类反馈即可对齐！田渊栋团队新作RLCD](https://mp.weixin.qq.com/s/sQolnpmBdCufVVR8q6GG8w)

- [RRTF：通过反馈提高代码生成的能力](https://mp.weixin.qq.com/s/3lgztkBGlfCdHwygDggBbw)

- [图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://mp.weixin.qq.com/s/J8c7rEmkQH4lBj1pWntv9w)

- [深挖RLHF潜力，复旦语言和视觉团队创新奖励模型优化，让大模型更对齐](https://mp.weixin.qq.com/s/BSaGLikARlvM8yitYtlA3w)

- [RLHF 和 DPO：简化和增强语言模型的微调](https://mp.weixin.qq.com/s/-5nzriCsoZIL3FKZxzbONw)

- [大模型reward model的trick](https://mp.weixin.qq.com/s/G69w-Y2Jb_SgtvLcjCs_3g)

- [无需训练实现价值观实时动态对齐：上交开源价值观对齐方法，闭源与开源大模型均适用](https://mp.weixin.qq.com/s/_CB0LBQVI_2NBiX63pyYSA)

- [使用KTO进行更好、更便宜、更快速的LLM对齐](https://mp.weixin.qq.com/s/vFrcW43jhraZT8ZaDBxl7A)

- [使用RLlib框架搭建大语言模型RLHF流程](https://zhuanlan.zhihu.com/p/648215474)


