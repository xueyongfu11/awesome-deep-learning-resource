[TOC]



# 训练框架

- https://github.com/hpcaitech/ColossalAI
- Deepspeed
  - [Deepspeed的基本使用](https://zhuanlan.zhihu.com/p/650824387)
- https://github.com/NVIDIA/Megatron-LM
  - [图解大模型训练之：张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228?utm_campaign=shareopn&utm_medium=social&utm_oi=615941546193850368&utm_psn=1631231489340559360&utm_source=wechat_session)
- [大模型训练-并行策略](https://mp.weixin.qq.com/s/opALgF1G9d-Lp-AjxhzUVw)


# Train

## pipline parallelism

- deepspeed：https://www.deepspeed.ai/tutorials/pipeline/
  - 推荐使用pytorch Sequential来编排模型
  - 对于forward的多参数输入的情况，将多参数转化为元组，forward只输入一个元组
    类型的参数
  - 在pipline并行中前向和后向是交叉的，因此不能使用分开的前向、后向、step等，
    deepspeed提供了train_batch()
  - LayerSpec可以极大的节约内存占用
  - 使用TiedLayerSpec可以通过指定key来公用同一层的参数
  - demo
    - https://github.com/HuangLK/transpeeder
    - https://github.com/CoinCheung/gdGPT
    - https://github.com/liucongg/ChatGLM-Finetuning