[TOC]




# Repo
- 多任务学习开源代码 https://github.com/namisan/mt-dnn
- https://github.com/thuml/MTlearn
- https://github.com/brianlan/pytorch-grad-norm
- https://github.com/txsun1997/Multi-Task-Learning-using-Uncertainty-to-Weigh-Losses
- https://github.com/richardaecn/class-balanced-loss

- https://github.com/vandit15/Class-balanced-loss-pytorch
- https://github.com/yaringal/multi-task-learning-example
- https://github.com/choosewhatulike/sparse-sharing
- https://github.com/VinAIResearch/PhoNLP
- https://github.com/huggingface/hmtl

- https://github.com/monologg/JointBERT
- https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification
- https://github.com/JayYip/m3tl
- https://github.com/MenglinLu/LDA-based-on-partition-PLDA-
- https://github.com/gregversteeg/corex_topic
- https://github.com/yinizhilian/ACL_Paper

- https://github.com/wenliangdai/multi-task-offensive-language-detection
- https://github.com/PaddlePaddle/PALM
- https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning
- https://github.com/drawbridge/keras-mmoe
- https://github.com/facebookresearch/vilbert-multi-task

- https://github.com/hellohaptik/multi-task-NLP
- https://github.com/richardaecn/class-balanced-loss
- https://github.com/mbs0221/Multitask-Learning
- https://github.com/vandit15/Class-balanced-loss-pytorch


# Paper

- [多任务学习中的loss平衡](https://mp.weixin.qq.com/s/dSrpDoL8am4bYMUhKNmsZQ)
  - 每个任务的损失函数的量纲不同，因此可以使用初始任务损失的倒数作为权重
  - 用先验分布代替初始分布更加合理，因此可以使用先验分布的损失的倒数作为权重
  - 使用实时的损失值的倒数来作为权重，基于此提出基于广义平均的方法
  - 为了获取平移不变性，以梯度的模作为调节的权重

- GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks

# Blog

- [深度学习多目标优化的多个loss应该如何权衡？](https://mp.weixin.qq.com/s/ZcsZec8vgdcUXlbTbQEH4A)
  - uncertainty weight方法；grad norm方法
- [多任务学习漫谈：以损失之名](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247558428&idx=1&sn=10ae39d6cc92c4e1517231adfa6be45d&chksm=96eb3a9ca19cb38a92da1c62011dcb243ff61db4b77f7b3685f92e349b1c938e536e63ae0291&scene=21#wechat_redirect)
- [​多任务学习漫谈：行梯度之事](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247560539&idx=1&sn=7f7071bde0758be1b320938a2713e459&chksm=96eb32dba19cbbcdd41fb408999e273c428bd32f87e55b4e030e3e9a0f1b25dfc65ea8399c67&scene=21#wechat_redirect)
- [多任务学习漫谈：分主次之序](https://mp.weixin.qq.com/s/pE2X4o3ZCzf9Qw8COwKlWA)

- [GradNorm 梯度归一化](https://blog.csdn.net/Leon_winter/article/details/105014677)
- [精读论文：Multi-Task Learning as Multi-Objective Optimization](https://blog.csdn.net/m0_38088084/article/details/108009616)

- [多任务学习权重的动态调整](https://blog.csdn.net/u013453936/article/details/83475590?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.not_use_machine_learn_pai&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-3.not_use_machine_learn_pai)

