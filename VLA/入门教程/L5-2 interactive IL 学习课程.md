# L5-2 interactive IL：新手学习课程

---

## 0. 这节课在讲什么

本节课讲的是 **Interactive Imitation Learning，交互式模仿学习**。

前面常见的 Behavior Cloning，简称 BC，可以理解为：

```text
先收集专家演示数据
再把“状态 -> 动作”当成监督学习问题训练策略
```

Interactive IL 的思路更进一步：

```text
让学习中的策略真的去环境里执行
观察它会走到哪些状态
再让专家对这些状态给出动作或干预
把新数据加入训练集
继续训练
```

所以本节课的核心问题是：

1. 为什么普通 BC 不够
2. Interactive IL 和 BC / RL / Inverse RL 有什么区别
3. DAgger 为什么能缓解分布偏移
4. data aggregation 和 policy aggregation 分别是什么意思
5. HG-DAgger、privileged teacher 这些扩展方法在解决什么问题

---

## 1. 从 Behavior Cloning 的问题开始

### 1.1 BC 的基本形式

Behavior Cloning 是最直接的模仿学习方法。

假设专家策略是 `pi*`，我们收集专家演示数据：

```text
D = {(s, a*)}
```

其中：

| 符号 | 含义 |
|---|---|
| `s` | 状态或观测 |
| `a*` | 专家在该状态下采取的动作 |
| `pi` | 学到的策略 |
| `pi*` | 专家策略 |

然后训练一个模型：

```text
pi(s) ≈ pi*(s)
```

也就是给定状态，预测专家会做什么。

### 1.2 BC 最大的问题：训练分布和执行分布不一致

BC 的训练数据来自专家轨迹。

但是模型部署时，机器人执行的是自己学到的策略 `pi`，不是专家策略 `pi*`。

这会导致一个关键问题：

```text
训练时看到的是专家会到达的状态
执行时看到的是学习策略自己会到达的状态
```

一旦学习策略犯一个小错，它就可能进入专家数据中很少出现的状态。模型没见过这些状态，就更容易继续犯错，错误会沿着时间序列累积。

这就是模仿学习里的 **distribution shift，分布偏移** 或 **covariate shift**。

直观例子：

```text
专家开车时一直在车道中间
BC 模型只学过“车在车道中间时怎么开”

模型执行时稍微偏右
它从没认真学过“车已经偏右时怎么纠正”
于是可能越开越偏
```

Interactive IL 的核心价值，就是把这些“学习策略自己会遇到的状态”也纳入训练。

---

## 2. Interactive IL 的基本概念

Interactive IL 可以理解为：

```text
在训练过程中，把专家放进循环里
让专家根据学习策略实际遇到的状态继续提供反馈
```

核心要点有三点：

1. 学习目标仍然是模仿专家策略
2. 需要 interactive demonstrator，也就是能交互式回答问题或干预的专家
3. BC 可以看作 Interactive IL 的 1-step 特例

### 2.1 为什么说 BC 是 1-step 特例

BC 只做一次：

```text
专家演示 -> 训练策略 -> 部署
```

Interactive IL 会循环多次：

```text
训练初始策略
-> 策略 rollout
-> 收集策略访问到的状态
-> 查询专家动作
-> 聚合数据
-> 重新训练策略
-> 再 rollout
```

所以 BC 可以看成没有后续交互迭代的特殊情况。

### 2.2 Interactive IL 和 RL 的区别

Interactive IL 和强化学习都需要和环境交互，但它们要的监督信号不同。

| 方法 | 是否需要奖励函数 | 是否需要专家策略 | 主要监督信号 |
|---|---:|---:|---|
| BC | 否 | 需要专家数据 | 专家动作 |
| Interactive IL | 否 | 需要可交互专家 | 专家动作或专家干预 |
| RL | 需要 | 不一定需要 | reward |
| Inverse RL | 从专家行为中推断 reward | 需要专家行为 | 推断出来的 reward |

Interactive IL 的典型设定是：

```text
不需要手写 reward
但需要专家能在训练过程中回答：这个状态下应该怎么做
```

这在很多机器人任务中很重要，因为 reward 可能难设计，但专家动作相对容易给。

---

## 3. 状态分布：为什么 rollout 很关键

一个核心概念是：

```text
d_pi(s)：由策略 pi 诱导出来的状态分布
```

意思是：

```text
如果机器人一直按策略 pi 执行，它会更经常遇到哪些状态？
```

不同策略会带来不同状态分布。

```text
专家策略 pi* -> 专家状态分布 d_pi*
学习策略 pi -> 学习策略状态分布 d_pi
```

BC 的问题是训练数据主要来自 `d_pi*`，但部署时面对的是 `d_pi`。

Interactive IL 的做法是让当前策略 `pi` 去 rollout，也就是实际运行一段轨迹，看看它会访问哪些状态，然后在这些状态上询问专家动作。

### 3.1 朴素的迭代思路

naive reduction method 可以理解成：

```text
固定状态分布 P，训练策略 pi
固定策略 pi，rollout 得到新的状态分布 P
同时收集专家动作 a*
重复这个过程
```

更直白地说：

```text
先学一个策略
让它去试
看它会遇到什么状态
让专家告诉它这些状态下该怎么做
再继续学
```

这就是从普通监督学习走向交互式序列学习的关键。

---

## 4. Sequential Learning Reductions

Interactive IL 可以看作一种 **sequential learning reduction**：

```text
把序列决策中的模仿学习问题
转化成一系列监督学习问题
```

这句话很重要。

模仿学习本来不是一个标准 IID 监督学习问题，因为每一步动作都会影响未来会看到哪些状态。

但 Interactive IL 通过 rollout 和专家查询，把问题拆成多轮监督学习：

```text
第 1 轮：在初始数据上训练策略
第 2 轮：当前策略 rollout，收集新状态，查询专家，重新训练
第 3 轮：再 rollout，再查询，再训练
...
```

### 4.1 通用流程

流程可以整理成：

```text
输入：初始策略 pi_0，专家策略 pi*

for m = 1, 2, ...:
    1. 使用当前策略 rollout，收集轨迹
    2. 得到当前策略访问到的状态集合
    3. 对这些状态查询专家动作
    4. 根据收集到的数据更新策略
```

更新策略时有两类典型做法：

```text
Data Aggregation：聚合数据，再训练策略
Policy Aggregation：聚合策略，形成新策略
```

---

## 5. Data Aggregation：DAgger

DAgger 是本节最核心的方法。

DAgger 的全称是：

```text
Dataset Aggregation
```

它的核心思想非常直接：

```text
不要只在专家原始数据上训练
要把学习策略自己遇到的状态也加入数据集
```

### 5.1 DAgger 的直观流程

```text
1. 用专家数据训练一个初始策略 pi_1
2. 让 pi_1 在环境中 rollout
3. 记录 pi_1 访问到的状态
4. 对这些状态查询专家动作
5. 把新数据加入总数据集 D
6. 用聚合后的 D 训练新策略 pi_2
7. 重复 2-6
```

用伪代码表示：

```text
D = 初始专家数据
train pi_1 on D

for i = 1 ... N:
    rollout pi_i
    collect states S_i
    query expert actions A_i = pi*(S_i)
    D = D ∪ {(S_i, A_i)}
    train pi_{i+1} on D
```

### 5.2 DAgger 解决了什么

DAgger 的关键不是换了一个模型，而是换了数据分布。

BC 学到的是：

```text
在专家常去的状态上模仿专家
```

DAgger 学到的是：

```text
在自己实际会遇到的状态上模仿专家
```

这使得策略更容易学会“纠错”。

例如自动驾驶中：

```text
BC：只学车道中间怎么开
DAgger：如果模型偏出车道，也能请专家标注如何回到车道中心
```

### 5.3 DAgger 的代价

DAgger 虽然有效，但有明显成本：

1. 需要能 rollout，也就是能让当前策略和环境交互
2. 需要专家在训练过程中持续标注
3. 如果当前策略很差，rollout 可能不安全
4. 人类专家频繁标注会很累

所以实际机器人系统里，DAgger 往往需要安全机制、仿真环境、人类接管或 gate 机制。

---

## 6. Policy Aggregation：SIMLe

除了聚合数据，还可以聚合策略。

SIMLe 属于 policy aggregation 的代表思路。

Data aggregation 的重点是：

```text
把所有轮次的数据合起来，重新训练一个策略
```

Policy aggregation 的重点是：

```text
每一轮训练一个中间策略
再把新策略和旧策略组合起来
```

可以写成类似形式：

```text
pi_m = beta * pi'_m + (1 - beta) * pi_{m-1}
```

其中：

| 符号 | 含义 |
|---|---|
| `pi'_m` | 第 m 轮新训练出的中间策略 |
| `pi_{m-1}` | 上一轮策略 |
| `beta` | 新策略占比 |

直观理解：

```text
不要一下子完全相信新策略
而是把新策略和旧策略按比例混合
```

这可以让策略更新更平滑，降低每轮变化过大带来的不稳定。

---

## 7. HG-DAgger：Human-Gated DAgger

HG-DAgger 指 Human-Gated DAgger。

它关心的问题是：

```text
什么时候让人类专家介入？
什么时候让机器人自己执行？
```

普通 DAgger 可能要求专家频繁标注或监控，非常昂贵。HG-DAgger 引入 human gate：

```text
人类观察机器人执行
当判断机器人快要出错或不安全时，进行接管或给出反馈
```

这样做的目的：

1. 降低人类专家的标注负担
2. 避免机器人在危险状态下继续执行
3. 让训练数据集中包含更关键的失败边界状态

可以把 HG-DAgger 理解成：

```text
DAgger + 人类安全接管机制
```

### 7.1 Human-gated 和 robot-gated

最后可以区分 human-gated 和 robot-gated method。

区别在于 gate 由谁决定：

| 方法 | 谁决定是否干预 | 适用直觉 |
|---|---|---|
| Human-gated | 人类专家决定 | 人类能可靠判断风险 |
| Robot-gated | 机器人或模型自己判断 | 需要自动化、降低人类负担 |

Robot-gated 常见思路是让模型估计不确定性或风险：

```text
如果模型不确定，就请求专家
如果模型很确定，就自己执行
```

---

## 8. IL with Privileged Teacher

另一个重要概念是 privileged teacher。

它不完全等同于标准 Interactive IL，但和“如何获得专家策略”有关。

### 8.1 什么是 privileged information

Privileged information 指训练时可用、部署时不可用的信息。

例如自动驾驶中：

```text
训练时可以直接知道其他车辆的位置、速度、交通灯状态
部署时只能通过摄像头、雷达等传感器估计
```

这些真实状态信息就是 privileged information。

### 8.2 两阶段训练思路

整体思路可以整理成两步：

```text
Step 1:
    用专家数据训练一个 privileged agent
    这个 agent 可以看到 ground-truth state

Step 2:
    训练一个 sensorimotor student
    student 只能看真实部署时可获得的传感器输入
    student 学习模仿 privileged agent
```

直观理解：

```text
先训练一个“开卷老师”
再让“闭卷学生”模仿老师
```

这样做的好处是，老师策略可能更强、更稳定，因为它训练时看到的信息更完整。学生最终只需要学习如何从传感器观测中复现老师行为。

---

## 9. 方法对比

| 方法 | 数据来源 | 是否交互 | 是否需要专家在线反馈 | 核心问题 |
|---|---|---:|---:|---|
| BC | 离线专家演示 | 否 | 否 | 容易分布偏移 |
| DAgger | 专家数据 + 当前策略 rollout 状态 | 是 | 是 | 专家标注成本高 |
| SIMLe / Policy Aggregation | 多轮策略与反馈 | 是 | 是 | 策略如何平滑更新 |
| HG-DAgger | 人类接管时的数据 | 是 | 是 | 何时让人介入 |
| RL | 环境交互 | 是 | 否 | reward 设计和采样效率 |
| Inverse RL | 专家行为 | 通常需要环境 | 不一定在线 | 从行为推断 reward |
| Privileged Teacher | 专家数据 + privileged agent | 可交互也可离线 | 不一定 | 如何利用训练期额外信息 |

---

## 10. 本节课最重要的理解

### 10.1 IL 不是普通监督学习

普通监督学习默认训练样本近似独立同分布。

但模仿学习是序列决策：

```text
当前动作会改变未来状态
未来状态又会影响未来动作
```

所以一个小错误可能改变后续整条轨迹。

### 10.2 Interactive IL 的核心是修正训练分布

Interactive IL 不是简单地“多收点专家数据”。

更准确地说，它收集的是：

```text
当前学习策略实际会访问到的状态
```

然后在这些状态上向专家学习。

