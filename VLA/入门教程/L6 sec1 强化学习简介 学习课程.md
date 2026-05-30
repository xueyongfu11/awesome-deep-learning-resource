# L6 sec1 强化学习简介：新手学习课程

---

## 0. 这节课在讲什么

本节课进入第六章 **Reinforcement Learning，强化学习**。

前几章可以大致对应成：

```text
经典方法：位姿估计、抓取位姿计算、路径规划、IK、轨迹生成
监督学习方法：从数据中学习抓取姿态、路径或 policy
强化学习方法：通过和环境交互，学习能最大化奖励的 policy
大模型方法：VLA、语言交互、分层模型与泛化
```

本节是 RL 的入门，重点不是某个具体算法，而是建立基础概念：

1. RL 的基本组成：agent、environment、reward、policy
2. MDP 和 POMDP 是什么
3. return、value function、Q function 分别表示什么
4. optimal value 和 optimal policy 如何理解
5. value iteration 和 policy iteration 的基本思路
6. RL 算法有哪些常见分类

---

## 1. 强化学习的基本问题

强化学习研究的是：

```text
一个智能体 agent 在环境 environment 中连续行动，
通过奖励 reward 判断行为好坏，
最终学到一个策略 policy，
让长期累计奖励最大。
```

更直白地说：

```text
机器人不断尝试动作
环境给它反馈
它慢慢学会怎样做更好
```

### 1.1 一个典型 RL 流程

```text
观察当前状态 s_t
-> 策略 pi 根据状态选择动作 a_t
-> 环境执行动作并转移到新状态 s_{t+1}
-> 环境给出奖励 r_t
-> agent 根据经验更新策略
```

循环形式：

```text
s_t -> a_t -> r_t, s_{t+1} -> a_{t+1} -> ...
```

### 1.2 RL 和机器人任务的关系

以机器人抓取为例，RL 可以学习：

```text
看到当前物体和机械臂状态后
下一步应该移动到哪里
什么时候闭合夹爪
如何调整动作以提高成功率
```

在有些系统里，RL 可以直接学习 policy；在有些系统里，RL 只负责部分模块，例如学习局部控制、抓取动作调整或策略选择。

---

## 2. RL 和 IL 的区别

上一节 L5 讲的是 Imitation Learning，模仿学习。RL 和 IL 的目标都可以是学习 policy，但监督信号不同。

| 方法 | 学习依据 | 是否需要专家 | 是否需要 reward | 核心目标 |
|---|---|---:|---:|---|
| IL | 专家演示或专家动作 | 需要 | 不一定 | 模仿专家 |
| RL | 与环境交互得到的奖励 | 不需要 | 需要 | 最大化累计奖励 |

可以这样记：

```text
IL：专家告诉我怎么做
RL：环境告诉我做得好不好
```

RL 的核心特点包括：

1. 没有 expert policy 可直接模仿
2. 需要和环境交互
3. 需要 reward function
4. 目标是找到最大化 cumulative reward 的 policy

---

## 3. MDP：强化学习的数学模型

MDP 是 Markov Decision Process，马尔可夫决策过程。

RL 通常用 MDP 来描述序列决策问题。

### 3.1 MDP 的组成

一个 MDP 通常写作：

```text
(S, A, P, R, gamma)
```

也可以写成：

```text
(s, a, p, r)
或
(s, a, p, r, gamma)
```

各符号含义如下：

| 符号 | 含义 |
|---|---|
| `s_t in S` | t 时刻的状态，可以是离散或连续 |
| `a_t in A` | t 时刻的动作，可以是离散或连续 |
| `P` 或 `T` | 状态转移概率，也可理解为系统动力学 |
| `p(s_{t+1} | s_t, a_t)` | 在状态 `s_t` 执行动作 `a_t` 后转移到 `s_{t+1}` 的概率 |
| `r(s_t, a_t)` | 奖励函数 |
| `gamma` | 折扣因子 |
| `pi` | 策略 |

### 3.2 policy 是什么

policy 表示 agent 如何根据状态选择动作。

确定性策略：

```text
a = pi_theta(s)
```

意思是一个状态对应一个确定动作。

随机策略：

```text
pi_theta(a | s)
```

意思是在状态 `s` 下，以一定概率选择动作 `a`。

### 3.3 rollout 是什么

Rollout 指从某个初始状态开始，按策略连续执行，得到一条轨迹。

```text
s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T
```

在机器人里，rollout 可以理解为：

```text
让当前策略真的控制机器人或仿真机器人跑一段任务
```

---

## 4. Markov Property：马尔可夫性

MDP 的关键假设是 Markov property。

它的含义是：

```text
未来只依赖当前状态和当前动作
不需要依赖完整历史
```

公式上可以写成：

```text
p(s_{t+1} | s_t, a_t)
```

而不是：

```text
p(s_{t+1} | s_0, a_0, s_1, a_1, ..., s_t, a_t)
```

直观理解：

```text
如果当前状态 s_t 已经包含了做决策需要的全部信息，
那过去发生过什么就不再额外重要。
```

例如棋类游戏中，如果棋盘状态完整给出，那么过去怎么走到这个棋盘，通常不影响下一步决策。

---

## 5. POMDP：部分可观测情形

现实机器人任务常常不是完全可观测的。

机器人可能看不到真实状态，只能看到传感器观测：

```text
真实状态 s_t：物体真实位姿、速度、接触状态
观测 o_t：相机图像、深度图、力传感器读数
```

这就是 POMDP，Partially Observable Markov Decision Process。

### 5.1 POMDP 的组成

POMDP 可以写作：

```text
(S, A, P, O, H, R, gamma)
```

其中：

| 符号 | 含义 |
|---|---|
| `o_t` | t 时刻的观测 |
| `H` 或 `h` | observation model，观测模型 |
| `o = h(s)` | 状态生成观测 |
| `h(o | s)` | 给定状态时观测出现的概率 |

MDP 和 POMDP 的区别：

```text
MDP：agent 能看到完整状态 s
POMDP：agent 只能看到观测 o，状态 s 可能隐藏
```

机器人视觉任务更接近 POMDP，因为相机图像通常不是完整真实状态。

---

## 6. Reward 和 Return

### 6.1 reward 是即时反馈

reward 是每一步动作后环境给的反馈。

例如机器人抓取：

```text
抓取成功：+1
抓取失败：0 或 -1
碰撞：-1
动作太大：小惩罚
```

reward 设计会直接影响学到的策略。

### 6.2 return 是累计奖励

RL 优化的不是单步 reward，而是长期累计奖励，也叫 return。

这里使用 discounted return：

```text
R = sum_{i=0}^{T} gamma^i r_i(s_i, a_i)
```

一些教材会把 return 写成 `G`。

### 6.3 gamma 折扣因子

`gamma` 是 discount factor，折扣因子。

| gamma | 含义 |
|---|---|
| 接近 0 | 更重视眼前奖励 |
| 接近 1 | 更重视长期奖励 |

例如：

```text
gamma = 0.9
```

表示未来奖励仍然重要，但越远的奖励权重越小。

---

## 7. Value Function：状态价值函数

状态价值函数 `V^pi(s)` 表示：

```text
从状态 s 开始
之后一直按照策略 pi 行动
期望能获得多少 return
```

公式：

```text
V^pi(s) = E_pi[R | s_0 = s]
```

展开就是：

```text
V^pi(s) = E_pi[sum_{i=0}^{T} gamma^i r_i(s_i, a_i) | s_0 = s]
```

直观理解：

```text
V^pi(s) 衡量状态 s 在策略 pi 下有多好。
```

例如自动驾驶：

```text
车在车道中央、速度合适、前方安全
这个状态的 V 值可能较高
```

---

## 8. Q Function：动作价值函数

动作价值函数 `Q^pi(s, a)` 表示：

```text
从状态 s 开始
先执行动作 a
之后再按照策略 pi 行动
期望能获得多少 return
```

公式：

```text
Q^pi(s, a) = E_pi[R | s_0 = s, a_0 = a]
```

直观理解：

```text
Q^pi(s, a) 衡量“在状态 s 下做动作 a”有多好。
```

V 和 Q 的区别：

| 函数 | 评价对象 | 问的问题 |
|---|---|---|
| `V^pi(s)` | 状态 | 这个状态好不好 |
| `Q^pi(s, a)` | 状态-动作对 | 这个状态下做这个动作好不好 |

---

## 9. Optimal Value 和 Optimal Policy

RL 的最终目标是找到最优策略 `pi*`。

### 9.1 最优 Q 函数

```text
Q*(s, a) = max_pi E_pi[R | s_0 = s, a_0 = a]
```

意思是：

```text
在状态 s 先做动作 a，
之后用所有可能策略中最好的策略继续执行，
能得到的最大期望 return。
```

### 9.2 最优 V 函数

```text
V*(s) = max_pi E_pi[R | s_0 = s]
```

意思是：

```text
从状态 s 开始，能达到的最大期望 return。
```

### 9.3 从 Q 得到最优策略

如果已经知道 `Q*(s, a)`，最优策略就很直接：

```text
pi*(s) = argmax_a Q*(s, a)
```

也就是说：

```text
在每个状态下，选择 Q 值最大的动作。
```

同时：

```text
V*(s) = max_a Q*(s, a)
```

---

## 10. Value Iteration：价值迭代

Value iteration 的目标是不断更新价值函数，让它逐渐接近最优价值函数。

### 10.1 状态价值迭代

初始化：

```text
对所有状态 s，令 V_0(s) = 0
```

迭代更新：

```text
V_{k+1}(s) = max_a sum_{s'} p(s' | s, a) [r(s, a) + gamma V_k(s')]
```

含义：

```text
对当前状态 s，
尝试所有可能动作 a，
计算每个动作带来的即时奖励 + 未来价值，
选择最好的动作对应的价值。
```

### 10.2 Q 函数迭代

Q 的更新形式：

```text
Q_{k+1}(s, a) =
    r(s, a) + gamma sum_{s'} p(s' | s, a) max_{a'} Q_k(s', a')
```

直观理解：

```text
当前动作的价值 =
当前奖励 + 折扣后的下一状态最优动作价值
```

这也是后面 Q-learning 的基础。

---

## 11. Policy Iteration：策略迭代

Policy iteration 不直接只更新价值函数，而是在两个步骤之间循环：

```text
Policy Evaluation
Policy Improvement
```

### 11.1 Policy Evaluation

Policy evaluation 的问题是：

```text
给定一个策略 pi，它到底有多好？
```

也就是计算：

```text
V^pi(s)
```

它评估的是当前策略，而不是立刻找最优策略。

### 11.2 Policy Improvement

Policy improvement 的问题是：

```text
基于当前评估结果，怎样让策略变得更好？
```

常见方式包括：

1. 选择价值更高的动作
2. 对策略参数做梯度更新
3. 用无梯度搜索方法改进策略

### 11.3 General Policy Iteration

General Policy Iteration 简称 GPI。

它的核心循环是：

```text
1. Policy evaluation：评价当前策略
2. Policy improvement：改进当前策略
3. 重复直到策略稳定
```

可以理解为：

```text
先知道自己现在做得怎么样
再根据评价结果调整行为
```

---

## 12. Value Iteration 和 Policy Iteration 对比

| 方法 | 核心对象 | 主要动作 | 直观理解 |
|---|---|---|---|
| Value Iteration | 价值函数 | 反复更新 V 或 Q | 先算出什么状态/动作最值钱 |
| Policy Iteration | 策略和值函数 | 评价策略，再改进策略 | 先评估当前做法，再优化做法 |

两者都属于动态规划思想，需要知道或利用环境转移模型 `p(s' | s, a)`。

在大规模或未知环境中，后续课程会引出：

```text
Q-learning
Policy Learning
Actor-Critic
Offline RL
Inverse RL
```

---

## 13. RL 算法分类

RL 算法可以按不同方式分类。

### 13.1 Model-based RL 和 Model-free RL

| 类型 | 是否显式学习/使用环境模型 | 说明 |
|---|---:|---|
| Model-based RL | 是 | 学习或已知系统动力学，再基于模型规划 |
| Model-free RL | 否 | 不显式建模环境，直接学 value 或 policy |

环境模型指的是：

```text
p(s' | s, a)
```

也就是执行动作后环境会怎样变化。

### 13.2 Value-based、Policy-based、Actor-Critic

| 类型 | 学什么 | 代表直觉 |
|---|---|---|
| Value-based | 学 V 或 Q | 先判断动作好坏，再选动作 |
| Policy-based | 直接学 pi | 直接输出动作或动作概率 |
| Actor-Critic | 同时学 policy 和 value | actor 负责行动，critic 负责评价 |

后续几节课正好对应：

```text
6.2 Q Learning：value-based
6.3 Policy Learning：policy-based
6.4 Actor-Critic：结合两者
```

### 13.3 On-policy 和 Off-policy

| 类型 | 数据来自谁 | 说明 |
|---|---|---|
| On-policy | 当前正在学习的策略 | 用自己当前策略采样的数据更新自己 |
| Off-policy | 可以来自其他策略 | 可以用旧策略、行为策略或历史数据学习 |

直观记忆：

```text
On-policy：边用当前策略跑，边学当前策略
Off-policy：不一定用当前策略采样，也能学习目标策略
```

### 13.4 Online RL、Offline RL、Inverse RL

| 类型 | 核心特点 |
|---|---|
| Online RL | 训练时持续和环境交互 |
| Offline RL | 只用已有数据集训练，不能随意探索环境 |
| Inverse RL | 从专家行为中反推 reward function |

机器人里 Offline RL 很重要，因为真实机器人在线探索成本高、风险大。

