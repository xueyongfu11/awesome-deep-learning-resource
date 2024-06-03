[TOC]




## mix

- End-To-End Memory Networks
  - year: 2015 NeuralIPS
  - 阅读笔记: 
    1. 给定一个memory set，通过一个矩阵A映射到一个向量空间，与query embedding进行点积计算，使用softmax计算weights
    2. 将memory set通过一个矩阵C映射到一个向量空间，使用weights加权后得到输出
    3. 将输出向量和query向量相加后，通过一个线性层softmax计算概率分布
  - code: 


## Normalization

- Layer Norm

- Batch Norm

- Root Mean Square Layer Normalization
  - RMSNorm
  - 相比Layer Norm，分子去掉了减去均值部分，分母的计算使用了平方和的均值再开
    平方

- DeepNorm
  - 对Post-LN的改进
  - 以alpha参数来扩大残差连接，LN(alpha * x + f(x))
  - 在Xavier初始化过程中以Bata减小部分参数的初始化范围


## activation func

- Gaussian Error Linear Units（GELUs）
  - GELU，Relu的平滑版本
  - 处处可微，使用了标准正态分布的累计分布函数来近似计算

- Swish: a Self-Gated Activation Function
  - Swish
  - 处处可微，使用了beta来控制函数曲线形状
  - 函数为f(x) = x * sigmoid(betan * x)

- SwiGLU
  - 是Swish激活函数和GLU（门控线性单元）的结合
  - GLU使用sigmoid函数来控制信息的通过，GLU = sigmoid(xW+b) 矩阵点积操作 
    (xV + c)
  - SwiGLU: swish(xW+b) 矩阵点积操作 (xV + c)


## Loss

- [PolyLoss超越Focal Loss](https://mp.weixin.qq.com/s/4Zig1wXNDHEjmK1afnBw4A)