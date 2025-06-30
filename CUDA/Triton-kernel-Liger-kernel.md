[TOC]



# Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 是一个专门为 LLM 训练设计的 Triton kernels 集合，由LinkedIn的工程师开发和维护。它能有效地将多 GPU 训练吞吐量提高 20%，并将内存使用量减少 60%。目前已经实现了与 HuggingFace 兼容的 `RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy` 等功能，未来还会有更多。Liger Kernel可以直接与 Flash Attention、PyTorch FSDP 和 Microsoft DeepSpeed 配合使用。

## RMSNorm

### RMSNorm的反向梯度推导

以下是 **RMSNorm 反向传播（backprop）** 的详细推导过程，基于链式法则（Chain Rule）逐步拆解：


#### 1. 正向传播回顾  
RMSNorm 的正向计算定义为：  
$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{k} x_k^2}} \cdot w_i
$$
其中：  
- $x = [x_1, x_2, \dots, x_n]$ 是输入向量（特征维度），  
- $y = [y_1, y_2, \dots, y_n]$ 是归一化后的输出，  
- $w = [w_1, w_2, \dots, w_n]$ 是可学习的缩放参数（类似 LayerNorm 的 affine 变换），  
- $n$ 是特征维度的大小（即向量长度）。  


#### 2. 反向传播目标  
反向传播需要计算 **损失对输入 $x_i$ 的梯度** $\displaystyle \frac{\partial \mathcal{L}}{\partial x_i}$（图中简写为 $\displaystyle dx_i = \frac{\partial o}{\partial x_i}$，这里 $o$ 可理解为后续层传递来的损失）。  

根据链式法则，梯度需通过中间变量 $y_k$ 传递：  
$$
dx_i = \frac{\partial o}{\partial x_i} = \sum_{k} \frac{\partial o}{\partial y_k} \cdot \frac{\partial y_k}{\partial x_i}
$$
核心是计算 $\displaystyle \frac{\partial y_k}{\partial x_i}$（即 $y_k$ 对 $x_i$ 的偏导），再结合上游梯度 $\displaystyle \frac{\partial o}{\partial y_k}$（记为 $dy_k$，由后续层反向传递而来）累加。  


#### 3. 计算 $\boldsymbol{\frac{\partial y_k}{\partial x_i}}$  
分两种情况讨论（$k = i$ 或 $k \neq i$），因为 $y_k$ 的表达式中，当 $k=i$ 时 $x_i$ 同时出现在分子和归一化分母，而 $k \neq i$ 时 $x_i$ 仅出现在分母：  


##### 情况 1：$\boldsymbol{k = i}$（对自身的偏导）  
此时 $y_i = \displaystyle \frac{x_i \cdot w_i}{\sqrt{\frac{1}{n} \sum_{m} x_m^2}}$，记归一化分母为 $\sigma = \displaystyle \sqrt{\frac{1}{n} \sum_{m} x_m^2}$，则 $y_i = \frac{x_i \cdot w_i}{\sigma}$。  

对 $x_i$ 求偏导（用商数法则或乘积法则）：  
$$
\frac{\partial y_i}{\partial x_i} = w_i \cdot \frac{ \sigma \cdot 1 - x_i \cdot \frac{\partial \sigma}{\partial x_i} }{ \sigma^2 }
$$

先计算 $\displaystyle \frac{\partial \sigma}{\partial x_i}$：  
$\sigma = \left( \frac{1}{n} \sum_{m} x_m^2 \right)^{\frac{1}{2}}$，所以  
$$
\frac{\partial \sigma}{\partial x_i} = \frac{1}{2} \left( \frac{1}{n} \sum_{m} x_m^2 \right)^{-\frac{1}{2}} \cdot \frac{2 x_i}{n} = \frac{x_i}{n \cdot \sigma}
$$

代入 $\displaystyle \frac{\partial y_i}{\partial x_i}$ 的表达式：  
$$
\frac{\partial y_i}{\partial x_i} = w_i \cdot \frac{ \sigma - x_i \cdot \frac{x_i}{n \cdot \sigma} }{ \sigma^2 } = w_i \cdot \frac{ n \sigma^2 - x_i^2 }{ n \cdot \sigma^3 }
$$

注意到 $\sigma^2 = \displaystyle \frac{1}{n} \sum_{m} x_m^2$，因此 $n \sigma^2 = \sum_{m} x_m^2$，代入后：  
$$
\frac{\partial y_i}{\partial x_i} = w_i \cdot \frac{ \sum_{m} x_m^2 - x_i^2 }{ n \cdot \sigma^3 }
$$


##### 情况 2：$\boldsymbol{k \neq i}$（对其他位置的偏导）  
此时 $y_k = \displaystyle \frac{x_k \cdot w_k}{\sigma}$（$\sigma$ 仍包含 $x_i$），因此对 $x_i$ 求偏导时，仅需对分母 $\sigma$ 求导：  

$$
\frac{\partial y_k}{\partial x_i} = x_k \cdot w_k \cdot \frac{ -1 }{ \sigma^2 } \cdot \frac{\partial \sigma}{\partial x_i}
$$

代入 $\displaystyle \frac{\partial \sigma}{\partial x_i} = \frac{x_i}{n \cdot \sigma}$（同前），得：  
$$
\frac{\partial y_k}{\partial x_i} = - w_k \cdot \frac{ x_k \cdot x_i }{ n \cdot \sigma^3 } \quad (k \neq i)
$$


#### 4. 整合链式法则  
上游梯度 $\displaystyle \frac{\partial o}{\partial y_k}$ 记为 $dy_k$（即反向传播中从后续层传来的梯度），则：  

$$
dx_i = \sum_{k} dy_k \cdot \frac{\partial y_k}{\partial x_i} = dy_i \cdot \frac{\partial y_i}{\partial x_i} + \sum_{k \neq i} dy_k \cdot \frac{\partial y_k}{\partial x_i}
$$

将两种情况的偏导代入：  

$$
dx_i = dy_i \cdot w_i \cdot \frac{ \sum_{m} x_m^2 - x_i^2 }{ n \cdot \sigma^3 } \; - \; \frac{1}{n \cdot \sigma^3} \sum_{k \neq i} dy_k \cdot w_k \cdot x_k \cdot x_i
$$


#### 5. 简化与实现  
为了高效计算，通常会预先缓存 $\sigma$（正向传播时的归一化分母），并利用向量运算批量处理。核心思路是：  
- 归一化操作让 $x_i$ 影响所有 $y_k$（因此反向传播需累加所有 $y_k$ 的梯度），  
- 通过预计算 $\sigma$ 和平方和 $\sum x_m^2$，可避免重复计算，提升效率。  


#### 总结  
RMSNorm 的反向传播本质是 **链式法则的应用**：由于输入 $x_i$ 参与了所有 $y_k$ 的归一化（分母含全局平方和），因此梯度需要遍历所有 $y_k$ 并累加其对 $x_i$ 的偏导。推导中需区分 $k=i$（自身梯度）和 $k \neq i$（交叉梯度）两种情况，最终整合得到 $dx_i$ 的表达式。  

若需代码实现，通常会用向量化运算（如 PyTorch 中的 `torch.sum`、`torch.pow` 等）高效计算这些梯度，避免显式循环。