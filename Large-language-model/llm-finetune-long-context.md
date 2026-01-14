[TOC]




# 支持更长的序列

## train

- 常见方法
  - fp16、bf16
  - gradient checkpoint
    - 以牺牲训练时间为代价来节约显存
  - 混合精度训练
  - 梯度累计
  - FasterTransformer
  - 模型并行、管道并行、CPU-Offload、3D并行

- QLora
  - [QLoRA：一种高效LLMs微调方法，48G内存可调65B 模型，调优模型Guanaco 堪比Chatgpt的99.3%！](https://zhuanlan.zhihu.com/p/632229856)
  - 使用4位的NormalFloat（Int4）量化和Double Quantization技术。4位的NormalFloat使用分位数量化，通过估计输入张量的分位数来确保每个区间分配的值相等，
  Double Quantization是将额外的量化常数进行量化。
  - 梯度检查点会引起显存波动，从而造成显存不足问题。通过使用Paged Optimizers技术，使得在显存不足的情况下把优化器从GPU转义到CPU中。
  - QLora张量使用使，会把张量 反量化 为BF16，然后在16位计算精度下进行矩阵乘法。
  - https://github.com/artidoro/qlora
  - https://huggingface.co/blog/4bit-transformers-bitsandbytes
    - 介绍及使用

- FlashAttention
  - 将频繁HBM内存访问转化为矩阵分片，并在SRAM上一次性计算的方式

- self attention does not need O(n^2) memory
  - 通过简单的数学结合律方法，取消self-attention计算中产生的A和S矩阵，代价是计算时间，但是用cuda核算子使得计算时间的代价很小，其实是在后向梯度传播时需要重计算
  - https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention

- Alibi
  - [Alibi位置向量外推性：看起来很长其实还是短](https://developer.aliyun.com/article/842370)
  - [ALiBi - 给注意力加上线性偏置](https://zhuanlan.zhihu.com/p/632780188)
  - https://www.mosaicml.com/blog/mpt-7b 在相对短的文本上预训练，然后在长文本上微调

- https://kaiokendev.github.io/til#extending-context-to-8k
  - 基于RoPE，scaling down frequency window

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
  - ZeRO-DP：优化器状态、梯度、模型
  - ZeRO-R：中间激活分区，适当的时候把中间激活卸载到CPU；定义一个合适的临时buffer值，达到计算和内存占用的平衡；
  基于tensor的不同生命周期来管理内存，从而避免碎片内存的产生
  - ZeRO-DP和ZeRO-R都是对DP的改进。ZeRO可以和MP结合起来进一步降低显存的占用
  - ZeRO的通信策略: 使用了动态的通信调度，该技术利用了模型状态（固定模型参数）的时间特性
- ZeRO-Offload: Democratizing Billion-Scale Model Training
  - 卸载部分数据和计算到CPU，同时保持计算效率
- ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
  - 使用GPU、CPU、硬盘对大模型进行训练
  - 解决涉及到的相关内存和带宽问题：infinity offload engine、memory-centric tiling、bandwidth-centric partitioning、overlap-centric design、ease-inspired implementation

- [ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
- [ZeRO-2 & DeepSpeed: Shattering barriers of deep learning speed & scale](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)

- 稀疏attention
  - [【长文本处理】CoLT5与LongT5：针对长文本优化的T5模型](https://zhuanlan.zhihu.com/p/630197196)

## inference

- PagedAttention
  - [从 FlashAttention 到 PagedAttention, 如何进一步优化 Attention 性能](https://zhuanlan.zhihu.com/p/638468472)

