[TOC]




- https://github.com/NVIDIA/TransformerEngine


## Quantization-aware train/finetune

- FP4量化
  - 要理解FP4量化，需要先理解FP8量化
  - https://huggingface.co/blog/zh/4bit-transformers-bitsandbytes

- FP8 FORMATS FOR DEEP LEARNING
  - [FP8 量化-原理、实现与误差分析](https://zhuanlan.zhihu.com/p/574825662)
  - [FP8 量化基础](https://zhuanlan.zhihu.com/p/619431625)
  - [计算机如何表示小数，浮点数](https://zhuanlan.zhihu.com/p/358417700)

- [量化感知训练基础](https://zhuanlan.zhihu.com/p/158776813)
  - 量化感知训练需要在模型中插入伪量化算子，来模拟量化推理
  - 量化感知训练时由于插入了伪量化算子，引起梯度不能正常回传，最直接的方法时梯度直通器
  - 量化感知训练更新的参数可以是原模型fp32的权重，也可以是量化参数如scale