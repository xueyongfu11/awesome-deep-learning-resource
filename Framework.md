

# 训练框架

- https://github.com/hpcaitech/ColossalAI

- Deepspeed
  - deepspeed config: https://www.deepspeed.ai/docs/config-json/
  - [Deepspeed的使用总结](https://zhuanlan.zhihu.com/p/650824387)
  - [DeepSpeed 通过系统优化加速大模型推理](https://zhuanlan.zhihu.com/p/629644249#%E5%9B%9B%EF%BC%8CDeepspeed%20Inference%20%E6%A8%A1%E5%9D%97%E7%9A%84%E7%89%B9%E6%80%A7)
  - [blog2](https://mp.weixin.qq.com/s/OXKg4f6bEso8E-Rp-m7scg)

- https://github.com/NVIDIA/Megatron-LM
  - [图解大模型训练之：张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228?utm_campaign=shareopn&utm_medium=social&utm_oi=615941546193850368&utm_psn=1631231489340559360&utm_source=wechat_session)


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
  - ChatGLM pipline并行demo：https://github.com/liucongg/ChatGLM-Finetuning