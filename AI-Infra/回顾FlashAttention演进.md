# FlashAttention：FA1、FA2 与 FA3

## 1. 基础目标

Attention 的主要运算为：

```text
S = QKᵀ / sqrt(d)
P = softmax(S)       # 按 query 行归一化
O = PV
```

FlashAttention 通过分块、online softmax 和反向重计算，避免将完整 N×N 的 S、P 写入 HBM。分块算法不引入 Attention 近似，但浮点运算顺序变化仍可能造成数值差异；FP8 量化则会额外引入误差。

以下 FA1/FA2 的循环和线程划分，指原始论文比较中的典型前向方案，不代表所有发行版本与特殊 kernel。

## 2. FA1：K/V 外层，Q 内层

```python
# 一个 CTA 对应一个 (batch_id, head_id)
for j in range(KV块数):
    加载 K[j], V[j]
    for i in range(Q块数):
        加载 Q[i]、输出中间值及softmax状态
        计算当前分数块
        更新 O[i]、运行最大值和指数和
        写回状态
```

固定一个 KV 块，让它服务各个 Q 块。每个输出块随着 KV 遍历被多次更新。

一个 CTA 负责一个样本的一个 head，因此 CTA 总数是 B×H。序列变长只会增加 CTA 内部工作量，不会增加这种划分下的 CTA 数。

例如 B=1、H=32，总共 32 个 CTA；即使 GPU 有 108 个 SM，该 kernel 最多让 32 个 SM 承接这些 CTA，且不保证它们能充分利用各自 SM 的计算资源。

## 3. FA2：跨 CTA 按 Q 块并行

```python
# 一个 CTA 对应一个 (batch_id, head_id, q_block_id)
加载 Q[i]
初始化本块 softmax 状态和输出累加器
for j in range(KV块数):
    加载 K[j], V[j]
    更新当前 Q 块的输出
最终归一化并写回 O[i]
```

CTA 总数为：

```text
B × H × ceil(Nq / Br)
```

Nq 是 Q 的序列长度，Br 是 Q 块包含的 token 数。不同 Q 行对应不同输出，CTA 之间不需要合并输出。

真正增加并行度的是“将 Q 块映射为不同 CTA”，不仅是交换循环顺序。它在小 batch、长序列时尤其有用。

## 4. 为什么不按 KV 块分配 CTA

可以这样做，但会为同一输出产生多份部分结果，需要额外归约。

对单个 query：

```text
o = sum_j(exp(s_j) * v_j) / sum_j(exp(s_j))
```

假设两个 CTA 的局部分子、分母分别为 (20,2)、(90,3)，局部输出是 10、30，最终输出是 (20+90)/(2+3)=22，不能简单相加或平均局部输出。

数值稳定的归约可保存每个分块的：

```text
m_t = max(s in chunk t)
l_t = sum(exp(s - m_t))
u_t = sum(exp(s - m_t) * v)

m = max_t(m_t)
l = sum_t(exp(m_t - m) * l_t)
u = sum_t(exp(m_t - m) * u_t)
o = u / l
```

| 划分  | CTA 的输出           | 是否需要跨 CTA 归约 |
| ----- | -------------------- | ------------------- |
| 按 Q  | 不同输出行的完整结果 | 不需要              |
| 按 KV | 相同输出行的部分结果 | 需要                |

Q 很多时，优先按 Q 拆可避免归约成本。如果每个 KV CTA 遍历全部 Q，还会生成多份覆盖全部 Q 的中间输出。

decode 时 Q 常只有一个或少量 token，按 Q 拆不出足够任务，此时 split-KV 能以归约换并行度。Flash-Decoding 就使用这种思路；一个 split 也可以包含多个 KV tile，不要求每个最小 tile 都启动一个 CTA。

## 5. FA1 → FA2：CTA 内的 warp 分工

这与上一节的跨 CTA 并行是两个层级。

| 典型前向分工    | FA1                              | FA2                           |
| --------------- | -------------------------------- | ----------------------------- |
| warp 负责的维度 | 不同 K/V 切片，Q 由各 warp 访问  | 不同 Q 行，K/V 由各 warp 访问 |
| 输出关系        | 相同输出行的部分结果             | 不同输出行                    |
| 主要通信        | 共享内存写入、同步、归约部分结果 | 避免上述跨 warp 输出归约      |

FA1 的 split-K 需要合并部分结果。描述时不要把它固定成“写 shared → reduceMax → 缩放 O”的唯一执行流程，具体指令与 softmax 组织依实现而定。

FA2 中，各 warp 维护自己负责行的运行最大值 m、指数和 l 及输出累加器 u。warp 内仍可能有归约；协作加载和共享内存复用仍可能需要同步，不能说整个 CTA 不再同步。

## 6. FA2：减少非矩阵乘运算

FA2 循环中维护未归一化输出，最终统一归一化，减少重复归一化工作。

对单行 query，处理当前 KV 块，忽略缩放和 mask：

```text
s  = q @ K_jᵀ
m' = max(m, max(s))
p  = exp(s - m')
a  = exp(m - m')
l' = a*l + sum(p)
u' = a*u + p @ V_j

所有 KV 块处理完后：o = u/l
```

FA2 仍需要在最大值变化时重缩放旧状态，不能理解为取消 rescale。前向为反向保存的统计量可用 LSE=m+log(l)，而不必分别保存 m 和 l。

反向涉及不同梯度的累加依赖，不能直接套用前向“按 Q 切、无跨 CTA 归约”的结论。论文方案按 KV 列块并行并累加 dQ。

## 7. FA3：利用 Hopper 的异步硬件

FA3 的三项核心方向是搬运与计算重叠、matmul 与 softmax 重叠，以及 FP8 加速与误差控制。

### 7.1 Producer–consumer warp specialization

将不同 warp 分配为不同角色，不是把单个 warp 切成任意角色线程：

- Producer 发起 TMA 异步搬运，将数据从 HBM 加载到共享内存。
- Consumer 通过 WGMMA 执行矩阵乘，并进行 softmax 与输出更新。
- 多级缓冲和同步协调数据就绪、读取完成与缓冲区复用。

TMA 负责多维地址计算和搬运，减轻线程的访存指令、边界处理及寄存器负担。线程仍需发起操作并维护同步。Hopper WGMMA 由包含 4 个 warp 的 warpgroup 协作。

### 7.2 重叠 matmul 与 softmax

Tensor Core 执行矩阵乘，指数等操作使用其他硬件资源；若两者串行执行，就会交替空闲。

- **跨 warpgroup ping-pong**：一个 consumer warpgroup 做 softmax，另一个进行 matmul。
- **同一 warpgroup 内流水**：交错安排不同迭代中依赖允许的 matmul 和 softmax 工作。

不能将同一分数块尚未完成的 QKᵀ 与依赖它的 softmax 直接重叠。收益来自不同组、不同迭代及独立操作的安排，并伴随额外寄存器压力。

### 7.3 FP8、分块量化与非相干处理

FP8 Tensor Core 提供更高吞吐，但有效精度下降。论文采用：

1. 分块量化：不同块使用各自缩放因子，改善数值范围适配。
2. Incoherent processing：量化前，对 Q、K 的特征维度施加同一个随机正交变换，分散集中在少数维度的离群值。

若 Q'=QR、K'=KR，且 RRᵀ=I，则：

```text
Q'K'ᵀ = QRRᵀKᵀ = QKᵀ
```

变换本身保持点积，后续量化仍有误差。论文使用带随机符号的归一化 Hadamard 变换，将每行变换复杂度降为 O(d log d)。Q/K 必须使用匹配的变换，不能各自独立随机。具体软件版本是否实现论文中的全部精度措施，需要核对代码。

## 8. 版本对比

| 版本 | 主要解决的问题                          | 代表措施                                                 |
| ---- | --------------------------------------- | -------------------------------------------------------- |
| FA1  | N×N 中间矩阵的 HBM 读写与存储          | 分块、online softmax、反向重计算                         |
| FA2  | CTA 并行度不足、warp 通信、非矩阵乘开销 | 按 Q 块并行、warp 切 Q、延迟归一化                       |
| FA3  | Hopper 异步硬件利用不足及低精度效率     | TMA/WGMMA、warp specialization、matmul-softmax 流水、FP8 |
