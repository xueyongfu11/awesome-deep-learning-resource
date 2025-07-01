[TOC]



## Triton 操作的是physical view Contiguous()

1. CUDA编程中"连续性(Contiguity)"这个容易被忽视但非常重要的问题。它指出连续性问题可能会导致难以调试的静默bug，需要花费大量时间来解决。
2. 因此，完成一个triton kernel之后，只进行单元正确性和性能测试对于生产环境是不够的，因为生产环境会遇到张量连续性、张量形状和数据类型的差异。

2. 因此建议通过模拟真实的生产训练环境来验证模型输出(logits)、权重(weights)和损失值(loss)
3. 下面提供一个google colab脚本，用来进行Triton内核补丁版本与原始模型的逐层对比：[脚本](https://colab.research.google.com/drive/1e52FH0BcE739GZaVp-3_Dv7mc4jF1aif?usp=sharing)

## Triton 中的index越界bug

1. Triton的program_id是int32来表示的，然后在开发Cross Entropy时没有考虑到这一点，导致在较大的Vocab Size时index会越界。
2. 修复的方案是把program_id转换为int64
3. 不过，因为32位寻址可能会导致性能很慢，所以需要非常谨慎的处理这个问题。例如在PyTorch中，针对这两种不同的数据类型会通过C++模板来处理，它们的实现会共享一个kernel，但是可以避免这个index溢出的问题。
4. 这个脚本模拟了index越界的bug问题：[脚本](https://colab.research.google.com/drive/1WgaU_cmaxVzx8PcdKB5P9yHB6_WyGd4T?usp=sharing#scrollTo=X_Dn9wzVNpMC)

