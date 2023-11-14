

# 显存优化

- flash-attention
  - llama使用flash-attention：https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py#L79

- [Multi Query Attention和Group-Query Attention介绍](https://mp.weixin.qq.com/s/wOyDpxcxKATxGrP8W-1w2Q)

- Efficient Streaming Language Models with Attention Sinks



# 训练框架

- https://github.com/hpcaitech/ColossalAI

- Deepspeed
  - [blog1](https://zhuanlan.zhihu.com/p/629644249#%E5%9B%9B%EF%BC%8CDeepspeed%20Inference%20%E6%A8%A1%E5%9D%97%E7%9A%84%E7%89%B9%E6%80%A7)
  - [blog2](https://mp.weixin.qq.com/s/OXKg4f6bEso8E-Rp-m7scg)

- https://github.com/NVIDIA/Megatron-LM


# 推理框架
- https://github.com/ModelTC/lightllm

- https://github.com/NVIDIA/FasterTransformer
- https://github.com/NVIDIA/TensorRT-LLM

- https://github.com/Jittor/JittorLLMs

- https://github.com/InternLM/lmdeploy/

- [大语言模型推理性能优化汇总](https://mp.weixin.qq.com/s/9mfx5ePcWYvWogeOMPTnqA)

- vllm
  - https://github.com/vllm-project/vllm
  - 高效的kv-cache管理，基于pageAttention

- 大模型部署的方案
  - https://mp.weixin.qq.com/s/hSFuULV-7bykz-zRmG5CXA


# CUDA

- CUDA C++ Programming Guide
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

- [《CUDA C 编程指南》导读](https://mp.weixin.qq.com/s/0wFD5Q_U0TT32NIxy45y0g)

- [详解PyTorch编译并调用自定义CUDA算子的三种方式](https://mp.weixin.qq.com/s/rG43pnWY8fBjyIX-mFWTqQ)

