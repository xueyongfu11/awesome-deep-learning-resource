# L6-3 Policy Learning：新手学习课程

---

## 0. 这节课在讲什么

本节课讲 **Policy Learning**，也就是直接学习策略。

上一节 Q-learning 属于 value-based method，核心是学习：

```text
Q(s, a)：在状态 s 下做动作 a 有多好
```

然后通过：

```text
pi(s) = argmax_a Q(s, a)
```

来选择动作。

Policy Learning 的思路不同，它直接学习：

```text
pi_theta(a | s)
或
a = pi_theta(s)
```

也就是直接把状态映射到动作或动作概率。

本节重点是：

1. 为什么 Q-learning 在连续动作空间里会困难
2. policy gradient 的目标函数是什么
3. REINFORCE 算法怎样用采样估计梯度
4. baseline / advantage 为什么能降低方差
5. on-policy 和 off-policy policy gradient 有什么区别
6. importance sampling 在 off-policy PG 里起什么作用
7. DDPG 如何处理连续动作空间
8. policy gradient 在机器人任务中的应用

---

## 1. 从 Q-learning 的局限说起

Q-learning 的特点是：

| 特点 | 含义 |
|---|---|
| Model-free | 不需要显式知道环境转移模型 `p(s' | s, a)` |
| Off-policy | 可以用其他策略采样的数据学习 |
| 没有显式 policy | 通过 `argmax_a Q(s, a)` 间接选动作 |
| 处理连续动作困难 | 因为需要对动作做 `max` 或 `argmax` |

Q-learning 的动作选择依赖：

```text
argmax_a Q(s, a)
```

如果动作空间是离散的，例如：

```text
上、下、左、右
```

那直接枚举动作就可以。

但机器人里很多动作是连续的：

```text
机械臂关节速度
末端执行器位移
方向盘转角
油门和刹车值
```

这时 `argmax` 不再容易，因为动作有无限多种可能。

所以我们自然会想到：

```text
能不能直接学一个 policy，让它直接输出动作？
```

这就是 policy learning 的动机。

---

## 2. 强化学习目标回顾

RL 的目标是找到一个策略，使期望累计奖励最大。

可以写成：

```text
pi* = argmax_pi E_{tau ~ p_pi(tau)} [R(tau)]
```

其中：

| 符号 | 含义 |
|---|---|
| `tau` | 一条轨迹 trajectory |
| `p_pi(tau)` | 策略 `pi` 产生轨迹 `tau` 的概率 |
| `R(tau)` | 轨迹的累计奖励，也就是 return |

一条轨迹可以写成：

```text
tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
```

Policy Gradient 的目标是直接优化参数化策略：

```text
pi_theta
```

让它产生的轨迹 return 尽可能大。

---

## 3. Policy Gradient 的基本思想

Policy Gradient，简称 PG，目标是直接对策略参数 `theta` 做梯度上升。

定义目标函数：

```text
J(theta) = E_{tau ~ p_theta(tau)} [R(tau)]
```

也就是：

```text
当前策略 pi_theta 产生轨迹后，期望能拿到多少 return。
```

我们希望：

```text
theta* = argmax_theta J(theta)
```

更新方式是：

```text
theta = theta + alpha * grad_theta J(theta)
```

这里是梯度上升，因为目标是最大化 return。

### 3.1 theta 影响什么

需要强调：`theta` 同时影响两件事。

1. 动作选择：

```text
pi_theta(a_t | s_t)
```

2. 状态分布：

```text
d_theta(s)
```

因为动作变了，之后会走到哪些状态也会变。

这也是 RL 比普通监督学习难的地方：

```text
模型参数不只影响当前输出
还会通过环境交互影响未来数据分布
```

---

## 4. Policy Gradient 的核心公式

Policy Gradient 的经典形式可以写成：

```text
grad_theta J(theta)
= E_{tau ~ p_theta(tau)} [
    sum_t grad_theta log pi_theta(a_t | s_t) * R(tau)
  ]
```

直观理解：

```text
如果一条轨迹的 return 高，
就提高这条轨迹中动作被选中的概率。

如果一条轨迹的 return 低，
就降低这些动作被选中的概率。
```

也给了采样近似形式：

```text
grad_theta J(theta)
≈ 1/N * sum_i sum_t
    grad_theta log pi_theta(a_{i,t} | s_{i,t}) * R(tau_i)
```

其中：

| 符号 | 含义 |
|---|---|
| `N` | 采样轨迹数量 |
| `i` | 第 i 条轨迹 |
| `t` | 轨迹中的时间步 |
| `R(tau_i)` | 第 i 条轨迹的 return |

---

## 5. REINFORCE 算法

REINFORCE 是最经典的 Monte Carlo policy gradient 方法。

它的思路很简单：

```text
用当前策略采样若干条轨迹
计算每条轨迹的 return
用 return 加权 policy gradient
更新策略网络
```

### 5.1 REINFORCE 流程

```text
初始化策略 pi_theta

重复：
    1. 用当前策略 pi_theta 在环境中生成轨迹
    2. 计算每条轨迹的累计奖励 R
    3. 估计 grad_theta J(theta)
    4. theta = theta + alpha * grad_theta J(theta)
```

### 5.2 一句话理解 REINFORCE

```text
好结果中的动作，以后更可能被选到；
坏结果中的动作，以后更不可能被选到。
```

这就是 vanilla policy gradient。

和 IL 相比：

```text
IL 的监督信号来自专家动作
PG 的监督信号来自累计奖励 R
```

---

## 6. 为什么 Policy Gradient 方差大

Policy Gradient 通常需要采样估计梯度。

问题是：

```text
采样轨迹数量有限
每条轨迹的 return 波动可能很大
```

这会导致梯度估计方差很大，训练不稳定。

例如同一个动作在不同随机环境中可能得到不同结果：

```text
一次成功，不代表这个动作总是好
一次失败，也不代表这个动作一定差
```

如果样本不够，梯度方向可能很噪。

---

## 7. Baseline 和 Advantage

为了降低方差，policy gradient 常引入 baseline。

原始形式：

```text
grad log pi_theta(a_t | s_t) * R
```

加入 baseline 后：

```text
grad log pi_theta(a_t | s_t) * (R - b)
```

其中 `b` 是 baseline。

常见选择是状态价值函数：

```text
b = V(s_t)
```

于是：

```text
A(s_t, a_t) = R - V(s_t)
```

这个量叫 advantage，优势函数。

### 7.1 Advantage 的直观含义

`R` 表示这次结果有多好。

`V(s_t)` 表示从状态 `s_t` 出发通常能有多好。

所以：

```text
R - V(s_t)
```

表示：

```text
这次动作比这个状态下的平均表现好多少。
```

如果 advantage 为正：

```text
这个动作比预期好，应该提高概率。
```

如果 advantage 为负：

```text
这个动作比预期差，应该降低概率。
```

---

## 8. On-policy Policy Gradient

Vanilla policy gradient 通常是 on-policy。

意思是：

```text
用当前策略 pi_theta 采样数据
再用这些数据更新 pi_theta
```

需要强调：

```text
tau ~ p_theta(tau)
```

这要求轨迹必须来自当前策略。

### 8.1 On-policy 的问题

On-policy PG 的主要问题是采样效率低。

原因：

1. 每次策略更新后，旧轨迹和当前策略就不完全匹配
2. 需要不断用新策略和环境交互采样
3. 神经网络每轮更新通常不能太大，否则训练不稳定
4. 在真实机器人上 rollout 成本很高

这和 Interactive IL 中 rollout 的代价类似：

```text
都需要在环境中实际执行策略，收集新轨迹。
```

### 8.2 On-policy 的优点

On-policy PG 的优点是概念简单：

```text
数据就是当前策略产生的
梯度估计和目标策略直接对应
```

但代价是不能充分复用旧数据。

---

## 9. Off-policy 和 Importance Sampling

Off-policy 的目标是：

```text
复用以前的轨迹数据
或使用其他策略采样的数据
来更新当前策略。
```

问题是：

```text
我们想评估当前策略 pi_theta'
但数据可能是旧策略 pi_theta 采样出来的。
```

这时需要 importance sampling。

### 9.1 Importance Sampling 的作用

假设旧策略是 `pi_old`，新策略是 `pi_new`。

如果某个动作在新策略下更可能出现，而在旧策略下不太可能出现，就需要提高它的权重。

常见权重形式是：

```text
pi_new(a | s) / pi_old(a | s)
```

off-policy policy gradient 可以理解为：

```text
用旧数据估计新策略表现时，
根据新旧策略生成这些动作的概率比值进行校正。
```

### 9.2 Off-policy 的利弊

优点：

1. 可以复用历史数据
2. 采样效率更高
3. 对真实机器人更有吸引力

缺点：

1. importance sampling 可能带来高方差
2. 新旧策略差异太大时估计不稳定
3. 算法实现比 on-policy 更复杂

---

## 10. Policy Gradient 小结

Vanilla PG 的特点：

| 特点 | 说明 |
|---|---|
| On-policy | 需要当前策略采样数据 |
| Model-free | 不需要环境转移模型 `p(s' | s, a)` |
| 直接优化 policy | 不一定先学习 V 或 Q |
| 采样效率低 | 需要大量环境交互 |
| 方差较大 | 常用 baseline / advantage 降低方差 |

Policy Gradient 的核心直觉：

```text
用 return 或 advantage 给动作打分，
让好动作概率上升，让坏动作概率下降。
```

---

## 11. DDPG：Deep Deterministic Policy Gradient

后半部分进入 DDPG。

DDPG 的全称是：

```text
Deep Deterministic Policy Gradient
```

它可以看作：

```text
用于连续动作空间的 deep Q-learning 思路
```

也是一种 deterministic actor-critic 方法。

### 11.1 DQN 到 DDPG 的动机

DQN / Q-learning 更新里需要：

```text
max_{a'} Q(s', a')
```

连续动作空间下，这个 `argmax` 很难直接求。

DDPG 的做法是训练一个 actor 网络：

```text
mu_theta(s) ≈ argmax_a Q(s, a)
```

也就是说：

```text
不用真的枚举所有动作找最大值，
而是让 policy 网络直接输出近似最优动作。
```

### 11.2 DDPG 的两个网络

DDPG 有两个核心部分：

| 模块 | 作用 |
|---|---|
| Actor | 输入状态，输出连续动作 |
| Critic | 输入状态和动作，评估 Q 值 |

Actor：

```text
a = mu_theta(s)
```

Critic：

```text
Q_phi(s, a)
```

### 11.3 DDPG 如何学习

Critic 学习类似 Q-learning：

```text
目标：让 Q(s, a) 接近 r + gamma Q(s', mu(s'))
```

Actor 学习：

```text
调整策略参数，让 Q(s, mu(s)) 更大。
```

直观理解：

```text
Critic 负责评价动作好不好；
Actor 负责产生能被 Critic 评高分的动作。
```

### 11.4 DDPG 的探索

DDPG 是 deterministic policy：

```text
a = mu(s)
```

确定性策略本身不会随机探索，所以需要额外加噪声：

```text
a = mu(s) + noise
```

这样才能探索不同动作。

---

## 12. DDPG 和普通 Policy Gradient 的区别

| 方法 | 策略形式 | 是否学 Q | 典型数据方式 | 适合场景 |
|---|---|---:|---|---|
| REINFORCE / Vanilla PG | 随机策略 `pi(a|s)` | 不一定 | On-policy | 概念简单，但采样效率低 |
| DDPG | 确定性策略 `a = mu(s)` | 是 | 常用于 off-policy | 连续动作控制 |

DDPG 属于 actor-critic：

```text
actor：学 policy
critic：学 Q function
```

所以它已经不只是“纯 policy gradient”，而是把 policy learning 和 value learning 结合起来。

---

## 13. 机器人应用案例 1：自动驾驶仿真

第一个应用是 TORCS 自动驾驶仿真。

任务：

```text
在 Open Racing Car Simulator 中驾驶赛车
```

Observation：

```text
向量形式输入
例如车辆状态、道路信息等
```

Action 是 3 维连续向量：

| 动作维度 | 含义 |
|---|---|
| Acceleration | 油门，0 表示无油门，1 表示满油门 |
| Brake | 刹车，0 表示不刹车，1 表示满刹车 |
| Steering | 转向，-1 表示最大右转，+1 表示最大左转 |

这个案例说明：

```text
连续控制问题很适合用 policy learning 或 actor-critic 方法。
```

---

## 14. 机器人应用案例 2：机器人操作

第二个应用是 robotic manipulation。

任务：

```text
机器人完成插接、装配等操作任务
```

Observation 包括：

1. 关节位置
2. 关节速度
3. 关节力矩反馈
4. socket 和 plug 的全局位姿

Action：

```text
关节速度
```

Reward：

```text
sparse reward + shaped reward
```

其中：

| 奖励类型 | 含义 |
|---|---|
| Sparse reward | 只在成功等关键事件给奖励 |
| Shaped reward | 给中间过程设计辅助奖励 |

该任务使用：

```text
DDPG with demonstrations
```

也就是：

```text
RL + IL
```

这说明在机器人任务中，纯 RL 可能采样效率不够，结合专家演示可以帮助学习。

---

## 15. 本节课最重要的理解

### 15.1 Policy Learning 直接学动作规则

Value-based 方法先学动作价值，再通过 `argmax` 选动作。

Policy learning 直接学：

```text
状态 -> 动作
```

或：

```text
状态 -> 动作概率分布
```

这对连续动作空间尤其重要。

### 15.2 Policy Gradient 是 trial-and-error

PG 没有专家动作监督。

它靠环境 reward 判断：

```text
哪些动作最终导致高回报
哪些动作最终导致低回报
```

然后调整动作概率。

### 15.3 Baseline 不改变目标，但能降低方差

加入 baseline 后，动作更新依据从：

```text
这次结果好不好
```

变成：

```text
这次结果是否比预期更好
```

这通常更稳定。

### 15.4 DDPG 是连续控制的重要方法

DDPG 用 actor 近似连续动作空间中的 argmax，用 critic 评价动作价值。

可以记成：

```text
Actor 负责做动作
Critic 负责打分
```

---

## 16. 学习检查

学完本节后，应该能回答这些问题：

1. Q-learning 为什么在连续动作空间里困难？
2. policy learning 和 value-based learning 的区别是什么？
3. Policy Gradient 的目标函数 `J(theta)` 表示什么？
4. `grad log pi(a|s) * R` 的直观含义是什么？
5. REINFORCE 的基本流程是什么？
6. 为什么 vanilla policy gradient 方差大？
7. baseline 和 advantage 分别是什么？
8. on-policy PG 为什么采样效率低？
9. importance sampling 在 off-policy PG 中解决什么问题？
10. DDPG 为什么适合连续动作控制？
11. DDPG 中 actor 和 critic 分别做什么？
12. 为什么机器人任务中常把 RL 和 demonstration 结合？

---

## 17. 快速复习版

```text
Q-learning：
    学 Q(s, a)，通过 argmax_a Q(s, a) 选动作。
    连续动作空间下 argmax 很难。

Policy Learning：
    直接学习 policy。
    可以是随机策略 pi(a|s)，也可以是确定性策略 a = mu(s)。

Policy Gradient：
    直接最大化 J(theta) = E[R]。
    高 return 的动作被鼓励，低 return 的动作被抑制。

REINFORCE：
    用当前策略采样轨迹。
    用 Monte Carlo return 估计梯度。
    更新策略参数。

Baseline / Advantage：
    用 R - V(s) 代替 R，降低方差。

On-policy PG：
    数据必须来自当前策略，采样效率低。

Off-policy PG：
    复用旧数据，需要 importance sampling 校正分布差异。

DDPG：
    deterministic actor-critic。
    Actor 输出连续动作。
    Critic 估计 Q(s, a)。
    用 actor 近似连续动作空间中的 argmax。
```

