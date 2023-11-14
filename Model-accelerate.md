

# knowledge distillation
- https://github.com/HoyTta0/KnowledgeDistillation
- https://github.com/airaria/TextBrewer
- 


# C++重写
- https://github.com/NVIDIA/FasterTransformer


# inference tools
- tensorRT https://github.com/shouxieai/tensorRT_Pro
- 神经网络可视化工具 https://github.com/lutzroeder/Netron
- https://github.com/htqin/awesome-model-quantization
  - 模型量化paper整理
- https://github.com/airaria/TextPruner
- Asset eXchange models
  - https://developer.ibm.com/articles/introduction-to-the-model-asset-exchange-on-ibm-developer/

# 并行训练框架

- https://github.com/hpcaitech/ColossalAI

- Deepspeed
  - [blog1](https://zhuanlan.zhihu.com/p/629644249#%E5%9B%9B%EF%BC%8CDeepspeed%20Inference%20%E6%A8%A1%E5%9D%97%E7%9A%84%E7%89%B9%E6%80%A7)
  - [blog2](https://mp.weixin.qq.com/s/OXKg4f6bEso8E-Rp-m7scg)

- Lion: Adversarial Distillation of Closed-Source Large Language Model
  - [blog](https://mp.weixin.qq.com/s/_LQVHMJqPzMzIuM4wsO2Dw)
  - 先是根据chatgpt生成的数据进行模仿学习；基于小模型的生成结果来判别难样本；再生成数据来学习这种难样本

- https://github.com/ModelTC/lightllm

- Efficient Streaming Language Models with Attention Sinks

- [Multi Query Attention和Group-Query Attention介绍](https://mp.weixin.qq.com/s/wOyDpxcxKATxGrP8W-1w2Q)

- https://github.com/NVIDIA/FasterTransformer
- https://github.com/NVIDIA/TensorRT-LLM

- https://github.com/Jittor/JittorLLMs

- https://github.com/InternLM/lmdeploy/

- [大语言模型推理性能优化汇总](https://mp.weixin.qq.com/s/9mfx5ePcWYvWogeOMPTnqA)

- vllm
  - https://github.com/vllm-project/vllm
  - 高效的kv-cache管理，基于pageAttention

- flash-attention
  - llama使用flash-attention：https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py#L79


- CUDA C++ Programming Guide
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

- [《CUDA C 编程指南》导读](https://mp.weixin.qq.com/s/0wFD5Q_U0TT32NIxy45y0g)

- [详解PyTorch编译并调用自定义CUDA算子的三种方式](https://mp.weixin.qq.com/s/rG43pnWY8fBjyIX-mFWTqQ)

- 大模型部署的方案
  - https://mp.weixin.qq.com/s/hSFuULV-7bykz-zRmG5CXA