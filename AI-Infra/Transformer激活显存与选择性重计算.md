# Transformer 激活显存与选择性重计算

## 1. No Parallelism 公式推导

在不使用模型并行时，给出的单层 Transformer 激活内存为：

$$
M_{\mathrm{no\ parallelism}}
=sbh\left(34+5\frac{as}{h}\right)
=34sbh+5as^2b.
$$

其中：

- $s$：序列长度；
- $b$：microbatch size；
- $h$：hidden size；
- $a$：attention head 数量。

假设普通激活使用 FP16，每个元素占 2 bytes；dropout mask 每个元素占 1 byte。公式结果的单位是 bytes。

### Attention：$11sbh+5as^2b$

与 hidden state 大小相关的激活为：


| 激活                                    |    内存 |
| --------------------------------------- | ------: |
| Attention 输出投影的输入                |  $2sbh$ |
| Attention 输出 dropout mask             |   $sbh$ |
| Q、K、V 三个线性层的共享输入            |  $2sbh$ |
| $QK^T$ 反向传播所需的 $Q$ 和 $K$        |  $4sbh$ |
| Attention over Values 反向传播所需的$V$ |  $2sbh$ |
| 合计                                    | $11sbh$ |

与 Attention 矩阵大小相关的激活为：


| 激活                 |     内存 |
| -------------------- | -------: |
| Softmax 输出         | $2as^2b$ |
| Softmax dropout mask |  $as^2b$ |
| Softmax dropout 输出 | $2as^2b$ |
| 合计                 | $5as^2b$ |

因此：

$$
M_{\mathrm{attention}}=11sbh+5as^2b.
$$

### MLP：$19sbh$

标准 MLP 先将 hidden size 从 $h$ 扩展到 $4h$，再降回 $h$：

$$
M_{\mathrm{MLP}}
=\underbrace{2sbh}_{\text{第一层 Linear 输入}}
+\underbrace{8sbh}_{\text{第二层 Linear 输入}}
+\underbrace{8sbh}_{\text{GeLU 输入}}
+\underbrace{sbh}_{\text{dropout mask}}
=19sbh.
$$

### 两个 LayerNorm：$4sbh$

每个 LayerNorm 保存一份 FP16 输入，因此：

$$
M_{\mathrm{LayerNorm}}=2sbh+2sbh=4sbh.
$$

三部分相加：

$$
\begin{aligned}
M_{\mathrm{no\ parallelism}}
&=(11sbh+5as^2b)+19sbh+4sbh \\
&=34sbh+5as^2b \\
&=sbh\left(34+5\frac{as}{h}\right).
\end{aligned}
$$

其中 $34sbh$ 随序列长度线性增长，$5as^2b$ 随序列长度平方增长。长序列训练时，后者通常是主要的激活内存来源。

## 2. Tensor Parallel 公式及 $10sbh$、$24sbh/t$ 来源

使用 $t$ 路 Tensor Parallel 后，单层、单个 TP rank 上的激活内存为：

$$
M_{\mathrm{TP}}
=sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right)
=10sbh+\frac{24sbh}{t}+\frac{5as^2b}{t}.
$$

它不是简单地将 No Parallelism 的结果整体除以 $t$，因为 TP 只切分 Attention 和 MLP 内部的激活，模块入口、LayerNorm 输入和模块出口处的部分激活仍会在所有 TP rank 上重复保存。

### 未被 TP 切分的 $10sbh$


| 激活                              |    内存 |
| --------------------------------- | ------: |
| 两个 LayerNorm 的输入             |  $4sbh$ |
| Q、K、V 线性层的共享输入          |  $2sbh$ |
| MLP 第一层$h\rightarrow4h$ 的输入 |  $2sbh$ |
| Attention 输出 dropout mask       |   $sbh$ |
| MLP 输出 dropout mask             |   $sbh$ |
| 合计                              | $10sbh$ |

这些张量在每个 TP rank 上都有完整副本，所以不带 $1/t$。

### 被 TP 切分的 $24sbh/t$

Attention 内部按 attention head 或 hidden dimension 切分的部分为：

$$
\frac{1}{t}
\left(
\underbrace{2sbh}_{\text{输出投影输入}}
+\underbrace{4sbh}_{Q,K}
+\underbrace{2sbh}_{V}
\right)
=\frac{8sbh}{t}.
$$

MLP 中间维度 $4h$ 被切分，因此：

$$
\frac{1}{t}
\left(
\underbrace{8sbh}_{\text{第二层 Linear 输入}}
+\underbrace{8sbh}_{\text{GeLU 输入}}
\right)
=\frac{16sbh}{t}.
$$

两者合计：

$$
\frac{8sbh}{t}+\frac{16sbh}{t}=\frac{24sbh}{t}.
$$

Attention 的 $a$ 个 head 同样分配到 $t$ 个 TP rank，因此 Attention 矩阵激活从 $5as^2b$ 变为：

$$
\frac{5as^2b}{t}.
$$

最终得到：

$$
M_{\mathrm{TP}}
=10sbh+\frac{24sbh}{t}+\frac{5as^2b}{t}.
$$

令 $t=1$，该公式会退化为：

$$
10sbh+24sbh+5as^2b=34sbh+5as^2b,
$$

与 No Parallelism 公式一致。

## 3. Selective Activation Recomputation 原理与公式

Selective Activation Recomputation 不重算整个 Transformer 层，而是只重算 Attention 中占用大量内存、计算成本相对较低的部分：

$$
QK^T
\rightarrow \operatorname{Softmax}
\rightarrow \operatorname{Dropout}
\rightarrow PV,
$$

其中 $P$ 表示 dropout 后的 Attention probability。

前向传播时不长期保存以下 $s\times s$ 中间激活：

$$
\underbrace{2as^2b}_{\text{Softmax 输出}}
+\underbrace{as^2b}_{\text{dropout mask}}
+\underbrace{2as^2b}_{\text{dropout 输出}}
=5as^2b.
$$

反向传播需要这些张量时，再根据已经保存的 $Q$、$K$、$V$ 等激活重新执行对应的 Attention 运算。因此，激活内存公式中的二次项被去掉。

Tensor Parallel 与 Selective Activation Recomputation 组合时：

$$
\begin{aligned}
M_{\mathrm{TP}}
&=sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right) \\
&\Downarrow \\
M_{\mathrm{TP+selective}}
&=sbh\left(10+\frac{24}{t}\right).
\end{aligned}
$$

Tensor Parallel、Sequence Parallel 与 Selective Activation Recomputation 组合时：

$$
\begin{aligned}
M_{\mathrm{TP+SP}}
&=\frac{sbh}{t}\left(34+5\frac{as}{h}\right) \\
&\Downarrow \\
M_{\mathrm{TP+SP+selective}}
&=\frac{34sbh}{t}.
\end{aligned}
$$

对于 $L$ 层 Transformer，在忽略 Transformer 层之外的激活和流水线调度修正项时，总激活内存为：

$$
M_{\mathrm{total}}=\frac{34sbhL}{t}.
$$

与 Full Activation Recomputation 重新执行完整 Transformer 层不同，Selective Activation Recomputation 只增加 Attention 矩阵计算和 Attention over Values 的前向计算。论文给出的主要 FLOPs 近似为：

$$
\frac{\text{hardware FLOPs}}{\text{model FLOPs}}
\approx 1+\frac{s}{6h}.
$$

它利用“Attention 矩阵激活大，但相对整个 Transformer 层的重算成本低”这一特点，以较小计算开销消除 $O(as^2b)$ 的激活存储。

## 4. TP、TP+SP、Selective、Full Recomputation 对比表

以下均为单层、单个设备或并行 rank 上的激活内存：


| 配置                                                 | 激活内存                                         | 关键特点                                                 |
| ---------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| No Parallelism                                       | $sbh\left(34+5\frac{as}{h}\right)$               | 所有激活完整保存                                         |
| Tensor Parallel                                      | $sbh\left(10+\frac{24}{t}+5\frac{as}{ht}\right)$ | 内部激活被切分，但$10sbh$ 仍在 TP ranks 间重复           |
| Tensor + Sequence Parallel                           | $\frac{sbh}{t}\left(34+5\frac{as}{h}\right)$     | 所有保留激活都按并行度$t$ 切分                           |
| Tensor Parallel + Selective Recomputation            | $sbh\left(10+\frac{24}{t}\right)$                | 去掉 Attention 的$O(as^2b/t)$ 存储，仍存在重复的 $10sbh$ |
| Tensor + Sequence Parallel + Selective Recomputation | $\frac{34sbh}{t}$                                | 同时去掉重复激活和 Attention 二次项                      |
| Full Activation Recomputation                        | $2sbh$                                           | 只 checkpoint 每层输入，但需要重算完整 Transformer 层    |

需要注意：

- TP 主要切分 Attention 和 MLP 内部激活，不能消除 $10sbh$ 的重复存储；
- SP 进一步切分 TP 未切分的激活，使 $34sbh$ 整体除以 $t$；
- Selective Recomputation 消除 $5as^2b/t$，即随序列长度平方增长的部分；
- Full Recomputation 内存最低，但计算开销显著高于 Selective Recomputation。
