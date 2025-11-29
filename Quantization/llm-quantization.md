[TOC]



- [量化那些事之KVCache的量化](https://zhuanlan.zhihu.com/p/691537237)

- https://github.com/openppl-public/ppq

- https://github.com/NVIDIA-AI-IOT/torch2trt

- https://github.com/HuangOwen/Awesome-LLM-Compression

- https://github.com/intel/neural-compressor
  - Intel开源的模型量化、稀疏、剪枝、蒸馏等技术框架

- [pytorch profiler 性能分析 demo](https://zhuanlan.zhihu.com/p/403957917)

- https://github.com/666DZY666/micronet
  - 剪枝、量化

- [大模型量化概述](https://mp.weixin.qq.com/s/_bF6nQ6jVoj-_fAY8L5RvQ)
  - 分为量化感知训练、量化感知微调、训练后量化
- [GGUF 格式完美指南](https://blog.mikihands.com/zh-hans/whitedec/2025/11/20/gguf-format-complete-guide-local-llm-new-standard/)


## Quantization-aware train/finetune

- LLM-FP4: 4-Bit Floating-Point Quantized Transformers
  - https://github.com/nbasyl/LLM-FP4/tree/main

- LLM-QAT: Data-Free Quantization Aware Training for Large Language Models
  - https://github.com/facebookresearch/LLM-QAT
  - 提出了一种data-free的量化感知训练方法
  - 使用训练好的大模型生成量化感知训练数据，相比使用训练集进行量化感知训练，证明由更好的效果
  - 使用了MinMax的对称量化，对激活、权重、KV-cache等进行了量化，损失函数使用交叉熵损失

- QLora
  - [blog](https://zhuanlan.zhihu.com/p/632229856)
  - 使用4位的NormalFloat（Int4）量化和Double Quantization技术。4位的NormalFloat使用block-wise量化和分位数量化，
    通过估计输入张量的分位数来确保每个区间分配的值相等，模型参数符合均值为0的正态分布，因此可以直接缩放到
    [-1,1]，然后使用正态分布N(0,1)的quatile quantization的量化值；
    Double Quantization是将额外的量化常数进行量化。
  - 梯度检查点会引起显存波动，从而造成显存不足问题。通过使用Paged Optimizers技术，使得在显存不足的情况下把优化器从GPU转义到CPU中。
  - QLora张量使用时，会把张量 反量化 为BF16，然后在16位计算精度下进行矩阵乘法。
  - https://github.com/artidoro/qlora
  - https://huggingface.co/blog/4bit-transformers-bitsandbytes
    - 介绍及使用

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
  - [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
  - 量化的基本原理
  - transformers中的8bit的矩阵乘法
  - W8A8量化，根据激活参数量级大小，从激活中选取outliers，使用fp16*fp16的矩阵乘法，对于激活中的其他行，使用int8*int8的量化矩阵乘法
  - 选取激活中的outliers，同时需要将权重矩阵中相应的列取出，与outliners进行矩阵相乘


## Post-training quantization

- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
  - 2023.06
  - 某个权重是否重要不仅取决于权重大小，还取决于该通道的输入规模。激活值大的输入通道，对输出误差的放大效应应该越严重，这些通道的权重越要被精细地量化。
  - 用少量数据跑一下模型，统计每个输入通道的绝对值的平均值。然后定义每个输出通道的绝对值，即输入通道和权重的乘积。基于结果挑选少量且重要的通道进行量化。
  - 论文引入缩放因子来减少关键权重的量化误差。将大于1的缩放因子乘上模型权重，然后进行量化，同时将输入除以缩放因子，这在数学上是等价的，但是却可以降低量化误差
  - [Blog 深入理解AWQ量化技术](https://zhuanlan.zhihu.com/p/697761176)
- FPTQ: Fine-grained Post-Training Quantization for Large Language Models
  - 相比smoothquant，使用了指数函数把激活量化的难度转移到权重量化上
  - 相比通道量化，使用了分组量化
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
  - 2022.10
  - GPTQ不是对权重进行量化，而是基于输出误差最小化的量化方法。GPTQ 目标函数展开后得到带 Hessian 的二次型，其中Hessian 矩阵反映了每个权重对输出的重要性。
  - OBQ的思想是在最小化输出误差的前提下，逐个权重量化，并动态修正后续权重。每量化一个权重，就必须更新逆Hessian，比较慢，不并行，不稳定。
  - GPTQ对OBQ的改进，不再使用全局顺序量化，而是按照block/group量化，互不依赖，因此支持并行，提高量化速度。
  - block/group内部则按顺序贪心量化，无法并行。假如一行为一个分组，则按列量化每个参数，每量化一个参数，便使用Hessian更新剩余未量化的参数。
  - 使用了Cholesky信息重组的方法，提高了稳定性
- Up or Down? Adaptive Rounding for Post-Training Quantization
  - [blog](https://zhuanlan.zhihu.com/p/363941822)
  - 核心：对weights进行量化时，不再是round to nearest，而是自适应的量化到最近右定点值还是左定点值
- SmoothQuant和增强型SmoothQuant
  - 增强的SmoothQuant使用了自动化确定alpha值的方法，而原始的SmoothQuant则是固定了alpha值
  - [相关blog](https://zhuanlan.zhihu.com/p/648016909)
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
  - 使用了alpha增强的将激活量化难度转移到权重量化上，同时保证矩阵乘积不变
  - 实现时只对计算密集型算法进行了smooth量化，而对LN，relu，softmax等访存密集型算子使用fp16计算
  - ICML2023