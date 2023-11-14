


## Quantization-aware train/finetune

- [量化感知训练基础](https://zhuanlan.zhihu.com/p/158776813)
  - 量化感知训练需要在模型中插入伪量化算子，来模拟量化推理
  - 量化感知训练时由于插入了伪量化算子，引起梯度不能正常回传，最直接的方法时梯度直通器
  - 量化感知训练更新的参数可以是原模型fp32的权重，也可以是量化参数如scale