# L7 sec1 Transformers and Generative Model(1)：新手学习课程

---

## 0. 这节课在讲什么

本节进入课程后半部分：

```text
VLA and VLM&LLM for Robot Learning
```

也就是：

```text
如何把大模型、视觉语言模型、语言模型、Transformer 和生成模型用于机器人学习。
```

前面几章主要讲：

```text
经典方法：位姿估计、抓取位姿、路径规划、IK、轨迹生成
监督学习方法：抓取姿态学习、模仿学习
强化学习方法：Q-learning、Policy Gradient、Actor-Critic
```

从第 7 章开始，重点转向：

```text
基于大模型的方法
```

包括：

- VLA：Vision-Language-Action 模型
- VLM：Vision-Language Model
- LLM：Large Language Model
- Transformer
- VAE、Diffusion 等生成模型
- ACT、Diffusion Policy 等机器人策略模型

本份 PPT 是第 7 章的开头，主要讲两件事：

```text
1. 为什么机器人学习需要 VLA / VLM / LLM
2. Transformer 和生成模型的基础概念
```

---

## 1. 课程整体位置

如果以机器人抓取和操作为例，整门课大致可以分成几条路线。

### 1.1 经典方法

经典机器人方法通常是模块化流程：

```text
目标位姿估计
-> 抓取位姿计算
-> 路径规划
-> IK 求解
-> 关节空间轨迹生成
-> 执行
```

它的优点是：

```text
结构清晰、可解释、可控。
```

缺点是：

```text
每个模块都可能失败，整体系统需要大量工程调参。
```

### 1.2 监督学习方法

监督学习方法希望从数据中学习某些模块。

例如：

```text
学习抓取姿态
学习视觉到动作的映射
学习端到端 visuomotor policy
```

它可以减少手工设计，但依赖大量数据。

### 1.3 强化学习方法

强化学习方法通过奖励学习策略。

例如：

```text
Policy learning
Q-learning
Actor-Critic
```

它适合学习交互式决策，但真实机器人上探索成本高，奖励设计也困难。

### 1.4 基于大模型的方法

第 7、8、9 章重点是大模型方向。

它希望解决的问题包括：

```text
语言交互
任务泛化
长时序任务
多任务统一建模
复杂场景理解
```

可以理解为：

```text
让机器人不只是会执行一个固定技能，
而是能理解自然语言、拆解任务、调用技能，并在不同场景中泛化。
```

---

## 2. 本章 Outline

第 7 章大纲是：

1. Introduction
2. Transformers and Generative Model
3. ACT & Variants, Diffusion Policy
4. VLM & LLM for Planning
5. VLA: RT-1, RT-2, Octo and OpenVLA
6. VLA: RDT, Pi0 and Others
7. Dataset and Benchmark
8. Others & Summary

本份 PPT 覆盖前两部分：

```text
Introduction
Transformers and Generative Model
```

也就是后面学习 ACT、Diffusion Policy、RT-1、RT-2、OpenVLA 等方法前的基础。

---

## 3. 为什么复杂机器人任务很难

复杂任务的挑战包括：

```text
Long horizon
Context dependent
Fine manipulation
Possible failure and re-work
```

逐个解释如下。

### 3.1 Long horizon：长时序

很多机器人任务不是一步完成的。

例如：

```text
把杯子从柜子里拿出来并放到桌上
```

可能包含：

```text
打开柜门
识别杯子
伸手接近杯子
抓住杯子
避开柜门
移动到桌面
放下杯子
松开夹爪
```

每一步都可能依赖前一步是否成功。

如果某一步失败，后面都可能受影响。

### 3.2 Context dependent：依赖上下文

同一句语言指令，在不同场景中含义可能不同。

例如：

```text
把它放到旁边
```

这里的“它”和“旁边”都需要结合当前视觉场景理解。

机器人必须知道：

```text
用户指的是哪个物体？
旁边是哪个位置？
是否有障碍物？
目标位置是否可达？
```

### 3.3 Fine manipulation：精细操作

很多任务需要非常精细的接触控制。

例如：

```text
插孔
拧瓶盖
夹起薄片
整理柔性物体
双臂协作
```

这类任务不是只靠“到达目标位置”就能完成，还需要控制力、姿态、接触关系。

### 3.4 Possible failure and re-work：失败与返工

真实世界中动作经常失败。

例如：

```text
抓杯子没抓稳
物体滑落
柜门没完全打开
目标被遮挡
路径被临时障碍阻挡
```

机器人不能只执行固定脚本，而需要：

```text
检测失败
重新规划
修正动作
必要时询问人类
```

这也是 VLM、LLM 和 VLA 被引入机器人学习的重要原因。

---

## 4. 两条大模型机器人路线

区分了两种框架：

```text
VLA：End-to-end framework
VLM/LLM + Visuomotor Policy：Hierarchical framework
```

这两条路线非常重要。

---

## 5. VLA：Vision-Language-Action

VLA 全称是：

```text
Vision-Language-Action
```

它试图把视觉、语言和动作放进一个统一模型中。

输入可能包括：

```text
图像
语言指令
机器人状态
历史观测
```

输出是：

```text
机器人动作
```

例如：

```text
输入：图像 + “把红色杯子放到盘子里”
输出：机械臂末端的 6DoF 动作或动作序列
```

### 5.1 End-to-End VLA for Robotics

VLA 用于机器人时，面向：

```text
Static environment
Various and long-horizon tasks
Language to spatial actions
Interaction with human
```

也就是：

- 在静态或相对可控环境中执行任务
- 支持多种任务和长时序任务
- 从语言映射到空间动作，例如 6DoF 机械臂动作
- 可以通过语言交互向人类澄清任务

### 5.2 VLA 的直观例子

一个 VLA 模型可以做这样的事：

```text
看到桌面图像
听到“把香蕉放进篮子”
理解香蕉和篮子的位置
生成抓取、移动、放置动作
```

它不像传统系统那样显式拆成很多模块：

```text
检测 -> 分割 -> 位姿估计 -> 规划 -> 控制
```

而是希望模型直接学习：

```text
(视觉 + 语言 + 状态) -> 动作
```

### 5.3 VLA 的优点和难点

优点：

- 端到端，模块间误差传播可能更少
- 能利用大规模视觉语言数据
- 有机会支持自然语言交互
- 更适合多任务统一建模

难点：

- 需要大量机器人数据
- 动作输出必须精确、安全、可执行
- 长时序任务容易误差累积
- 对真实世界分布变化仍然敏感

---

## 6. VLA 和自动驾驶端到端模型的类比

End-to-End VLA for robotics 可以和 End-to-End Autonomous Driving 放在一起比较。

自动驾驶端到端模型通常做：

```text
BEV / 多传感器融合表示
检测、跟踪、建图、预测、规划等多任务学习
输出规划轨迹
```

自动驾驶的任务目标通常很清楚：

```text
从 A 点开到 B 点
```

但环境非常动态：

```text
其他车辆
行人
交通信号
道路结构
交互行为
```

机器人操作和自动驾驶的区别：

| 对比 | 机器人 VLA | 自动驾驶端到端 |
|---|---|---|
| 空间 | 3D 操作空间，常涉及 6DoF | 最终多在 2D 路面空间规划 |
| 任务 | 多样、语言指定、开放性强 | 从 A 到 B，目标较明确 |
| 动作 | 抓取、放置、推、插、接触操作 | 轨迹、转向、加减速 |
| 难点 | 精细操作、接触、泛化 | 动态交通、多主体交互、安全 |

两者共同点是：

```text
都希望从复杂感知输入直接生成可执行行为。
```

---

## 7. VLM/LLM + Visuomotor Policy：分层框架

另一条路线不是让一个模型端到端完成所有事，而是分层。

可以写成：

```text
VLM/LLM + Visuomotor Policy
```

### 7.1 高层：VLM / LLM 做规划

VLM 或 LLM 负责高层理解和任务拆解。

例如用户说：

```text
帮我泡一杯茶
```

LLM 可以拆成：

```text
找到杯子
拿起茶包
把茶包放进杯子
倒水
等待
递给用户
```

VLM 可以结合图像理解：

```text
杯子在哪里？
茶包在哪里？
水壶在哪里？
当前场景是否满足下一步条件？
```

### 7.2 低层：Visuomotor Policy 执行技能

低层 visuomotor policy 负责具体动作。

例如：

```text
抓取杯子
打开抽屉
推开物体
把物体放到某处
```

一个低层模型可以服务多个操作任务，也可以有多个专用模型。

### 7.3 控制器：保证执行稳定

在最底层通常还有控制器：

```text
轨迹优化
滤波
力控制
阻抗控制
关节控制
```

它的作用是把高层动作变成稳定、平滑、安全的机器人运动。

### 7.4 分层框架的整体流程

可以理解为：

```text
人类语言指令
  -> LLM/VLM 理解和规划
  -> 分解成子任务
  -> 调用 visuomotor policy
  -> 控制器执行
  -> 观察结果
  -> 必要时重新规划
```

这种方法的优点是：

- 高层推理和低层控制分工明确
- 可以复用已有技能
- 更容易加入规则、安全约束和人类反馈

缺点是：

- 模块之间接口需要设计
- 高层计划可能和低层可执行性不匹配
- 错误可能在模块之间传播

---

## 8. VLA 与分层框架对比

| 框架 | 核心思想 | 优点 | 难点 |
|---|---|---|---|
| End-to-End VLA | 视觉+语言直接输出动作 | 统一、简洁、可扩展到多任务 | 数据需求大，动作精度和安全难 |
| VLM/LLM + Visuomotor Policy | 高层规划 + 低层技能 | 可解释、模块化、易接入控制器 | 接口设计复杂，模块误差传播 |

直观记忆：

```text
VLA：一个大模型从看懂到动手
分层框架：大模型负责想，技能模型负责做
```

---

## 9. Transformer 为什么重要

后面很多 VLM、LLM、VLA 和机器人生成策略都基于 Transformer。

Transformer 的核心能力是：

```text
处理序列，并建模序列中不同位置之间的关系。
```

在自然语言中，序列是：

```text
词 token 序列
```

在机器人中，序列可以是：

```text
图像 patch 序列
语言 token 序列
机器人状态序列
动作序列
轨迹序列
多模态 token 序列
```

所以 Transformer 可以统一处理：

```text
视觉、语言、动作、历史观测
```



## 10. Transformer 在机器人中的输入形式

在机器人学习中，Transformer 的输入不一定只是文字。

常见输入可以包括：

| 模态 | 例子 |
|---|---|
| Vision | RGB 图像、深度图、图像 patch、视频帧 |
| Language | 自然语言指令、任务描述 |
| State | 关节角、末端位姿、夹爪状态 |
| Action | 历史动作、未来动作 chunk |
| Reward / return | 强化学习轨迹中的奖励或目标回报 |

这些都可以被表示成 token 序列。

所以 Transformer 对机器人学习的意义是：

```text
提供一种统一的多模态序列建模框架。
```

---

## 11. Generative Model 是什么

接下来进入生成模型。

生成模型的目标是：

```text
学习数据分布，并从中采样生成新数据。
```

在机器人里，生成模型可以生成：

```text
图像
轨迹
动作序列
抓取姿态
未来状态
多步计划
```

和判别模型相比：

```text
判别模型：判断或预测一个标签
生成模型：生成一个样本
```

例如：

```text
判别：这张图里有没有杯子？
生成：根据当前场景生成一段抓杯子的动作。
```

---

## 12. VAE：Variational Auto Encoder

VAE 的核心是：

```text
An encoder to learn the latent distribution
A decoder to reconstruct the original input
Similar to Auto Encoder, but used differently
```

VAE 的结构是：

```text
Encoder -> Latent Distribution -> Decoder
```

### 12.1 普通 AutoEncoder

普通 autoencoder 做的是：

```text
输入 x
-> 编码成 latent vector z
-> 解码重建 x'
```

目标是让：

```text
x' ≈ x
```

它更像压缩和重建模型。

### 12.2 VAE 的关键区别

VAE 不只是编码成一个确定向量，而是编码成一个分布：

```text
z ~ N(μ, σ)
```

然后从这个分布采样：

```text
sample z
```

再用 decoder 生成：

```text
x' = Decoder(z)
```

这样 VAE 就具备生成能力。

### 12.3 VAE 在机器人中的意义

机器人行为经常是多模态的。

同一个任务可能有多种合理解：

```text
从左边绕过去
从右边绕过去
先推开障碍物
直接抓取
先调整姿态再抓
```

VAE 的 latent distribution 可以表达这种多样性。

例如：

```text
不同 z -> 不同但合理的动作序列
```

---

## 13. CVAE：Conditional VAE

要点如下：

```text
CVAE: Conditional VAE
Add conditional input to encoder and decoder
```

普通 VAE 是：

```text
z -> x
```

CVAE 是：

```text
condition c + z -> x
```

在机器人中，condition 可以是：

```text
当前图像
机器人状态
语言指令
目标物体
历史动作
任务类别
```

例如：

```text
condition：当前桌面图像 + “抓红色杯子”
z：潜变量
output：一段抓取动作
```

CVAE 的好处是：

```text
在给定当前场景和任务的条件下，生成合理动作。
```

这为后面的 ACT 等方法做铺垫。

---

## 14. Diffusion Model 是什么

Diffusion model 的核心过程是：

```text
Forward process: add noise
Backward process: denoise
Usually use gaussian noise and use network to reconstruct denoise process
```

扩散模型的基本思想是：

```text
先把真实数据逐步加噪，变成噪声；
再训练模型从噪声一步步还原数据。
```

### 14.1 Forward process：加噪

从真实样本 `x_0` 开始：

```text
x_0 -> x_1 -> x_2 -> ... -> x_T
```

每一步加入一点高斯噪声。

最后 `x_T` 接近纯噪声。

### 14.2 Backward process：去噪

反向过程是：

```text
x_T -> x_{T-1} -> ... -> x_0
```

模型学习如何逐步去噪，最终生成类似真实数据的样本。

在机器人动作生成中：

```text
真实数据可以是专家动作序列
噪声数据是被加噪后的动作序列
模型学习从噪声恢复动作序列
```

---

## 15. DDPM

要点如下：

```text
DDPM: Denoising Diffusion Probabilistic Models
```

DDPM 有两个核心组件：

```text
Component 1: DDPM process
  Forward: add noise
  Backward: denoise

Component 2: Neural Network
  Predict the noise based on noisy input and time step t
```

### 15.1 神经网络预测什么

在常见 DDPM 训练中，网络不是直接预测干净样本，而是预测噪声：

```text
εθ(x_t, t)
```

输入是：

```text
带噪样本 x_t
时间步 t
```

输出是：

```text
预测的噪声 ε
```

训练目标是让预测噪声接近真实加入的噪声。

---

## 16. Diffusion Model 的训练

可以写成：

```text
Training:
Only use DDPM forward step
Add noise and predict the noise
MSE Loss
```

训练流程可以整理为：

```text
1. 从数据集中取一个真实样本 x_0
2. 随机采样时间步 t
3. 按照 forward process 给 x_0 加噪，得到 x_t
4. 把 x_t 和 t 输入神经网络
5. 网络预测噪声 εθ(x_t, t)
6. 用 MSE loss 让预测噪声接近真实噪声
```

损失可以写成：

```text
L = || ε - εθ(x_t, t) ||^2
```

直观解释：

```text
训练时，模型学习“这张带噪图片或这段带噪动作里，噪声是什么”。
生成时，模型不断减去预测噪声，逐步得到清晰样本。
```

---

## 17. Diffusion Model 在机器人中的意义

扩散模型特别适合机器人动作生成，因为它能表达复杂、多峰分布。

同一个状态下，专家可能有多种合理动作：

```text
左侧接近
右侧接近
先推再抓
直接抓
慢慢接近
快速接近
```

如果用普通回归模型，可能把多个动作平均掉，得到一个不合理动作。

扩散模型可以通过采样生成多种合理动作序列。

例如：

```text
condition：当前图像 + 语言指令
noise：随机动作序列噪声
denoise：逐步生成可执行动作序列
```

这就是后面 Diffusion Policy 的基础。

---

## 18. VAE 和 Diffusion 的对比

| 方法 | 生成方式 | 优点 | 局限 |
|---|---|---|---|
| VAE | 从潜变量 `z` 采样，再 decoder 一次生成 | 结构清晰，推理快 | 生成质量可能较模糊，表达复杂分布有限 |
| Diffusion | 从噪声开始多步去噪 | 生成质量高，能表达复杂多模态分布 | 推理通常更慢，训练和采样流程更复杂 |

直观理解：

```text
VAE：从隐空间抽一个想法，一次画出来
Diffusion：从噪声草稿开始，一步步修成真实样本
```

在机器人中：

```text
VAE 常用于动作潜变量、多模态动作 chunk
Diffusion 常用于高质量动作序列或轨迹生成
```

---

## 19. Transformer 与生成模型如何结合

后续机器人模型往往会把 Transformer 和生成模型结合。

例如：

```text
Transformer 负责建模上下文和序列关系
VAE / Diffusion 负责生成多模态动作
```

一个典型结构可能是：

```text
图像 token + 语言 token + 机器人状态 token
  -> Transformer 编码
  -> 生成模型输出动作序列
```

或者：

```text
Transformer 直接作为 denoising network
```

用于 diffusion policy 中预测动作噪声。

### 19.1 为什么这对机器人重要

机器人任务同时需要：

```text
理解场景
理解语言
利用历史
生成连续动作
处理多种可能方案
```

Transformer 解决“理解和序列建模”，生成模型解决“多样化动作生成”。

两者结合，就形成后面很多 VLA 和 generative policy 的基础。

---

## 20. 本节核心关系图

可以把本节内容整理成：

```text
Robot Learning with Foundation Models
|
|-- End-to-End VLA
|   |-- Vision + Language + Robot State -> Action
|   |-- 代表方向：RT-1, RT-2, OpenVLA 等
|
|-- VLM/LLM + Visuomotor Policy
|   |-- VLM/LLM：高层任务理解和规划
|   |-- Visuomotor Policy：低层操作技能
|   |-- Controller：轨迹、力控、滤波、安全执行
|
|-- Transformer
|   |-- Scaled Dot-Product Attention
|   |-- Multi-Head Attention
|   |-- Encoder / Decoder
|   |-- Encoder-only / Decoder-only / Encoder-decoder
|
|-- Generative Model
    |-- VAE / CVAE
    |-- Diffusion Model / DDPM
    |-- 后续连接 ACT、Diffusion Policy
```

