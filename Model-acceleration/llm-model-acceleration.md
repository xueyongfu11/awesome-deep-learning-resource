[TOC]

# llm-model-acceleration

- flash-attention
  - https://github.com/Dao-AILab/flash-attention
  - 原理介绍：[FlashAttention V1 学习笔记](https://blog.csdn.net/weixin_43378396/article/details/137635161)
  - [flash attention triton视频视频讲解](https://www.bilibili.com/video/BV1j59tY1EFt)：[my code](https://github.com/xueyongfu11/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
- [Multi Query Attention和Group-Query Attention介绍](https://mp.weixin.qq.com/s/wOyDpxcxKATxGrP8W-1w2Q)
- Efficient Streaming Language Models with Attention Sinks
- [Transformer参数量、计算量、显存占用分析](https://mp.weixin.qq.com/s/4_6J7-NZML5pTGTSH1-KMg)


- SELF-ATTENTION DOES NOT NEED O(n2) MEMORY
  - 将self-attention的内存占用优化到了O(logn)
  - 考虑一个query和长度为n的key、value列表。attention的计算可以表示为分子和分母的迭代计算，而不需要保存中间计算结果，即i=i+1
  - 传统attention的计算会减去一个最大值防止溢出，新的懒计算的方法无法使用该方法。维护一个当前时刻的最大值，来更新计算结果
  - <details>
    <summary>Image </summary>
    <img src="../assets/xFormer.png" align="middle" />
    </details>



# 推理框架

- https://github.com/alibaba/rtp-llm
- https://github.com/bentoml/OpenLLM
  - 支持multi-lora, 本质是peft的api调用
- https://github.com/huggingface/text-generation-inference
- DeepSpeed-FastGen
  - https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen
- Deepspeed Inference
  - https://www.deepspeed.ai/tutorials/inference-tutorial/
- https://github.com/ModelTC/lightllm
- https://github.com/NVIDIA/FasterTransformer
- https://github.com/NVIDIA/TensorRT-LLM
- https://github.com/Jittor/JittorLLMs
- https://github.com/InternLM/lmdeploy/
- Blog
  - [大模型部署的方案](https://mp.weixin.qq.com/s/hSFuULV-7bykz-zRmG5CXA)

  - [大语言模型推理性能优化汇总](https://mp.weixin.qq.com/s/9mfx5ePcWYvWogeOMPTnqA)
  
  - [推理部署工程师面试题库](https://zhuanlan.zhihu.com/p/673046520)
  
  - [triton中文站](https://triton.hyper.ai/)

