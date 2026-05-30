# L6-5 inverse and offline RL：新手学习课程

---

## 0. 这节课在讲什么

本节课主要讲两类强化学习相关方法：

```text
Inverse RL：逆强化学习，学习 reward
Offline RL：离线强化学习，用已有数据学习 policy
```

前面几节课已经讲过：

```text
Q-learning：学习 Q(s, a)
Policy Learning：直接学习 policy
Actor-Critic：同时学习 policy 和 value / Q
```

本节继续回答两个更实际的问题：

1. 如果我们没有 reward function，但有专家轨迹，能不能反推出 reward？
2. 如果真实机器人不能随便在线探索，能不能只用已有数据做 RL？

所以本节重点是：

1. Inverse RL 和 BC / IL / RL 的区别
2. 线性 reward 假设和 feature expectation
3. LP-IRL 的基本思路
4. Maximum Entropy IRL 的训练流程
5. Offline RL 的定义和数据格式
6. TD3+BC、CQL、Decision Transformer 的基本直觉
7. Offline RL 在机器人中的应用
8. SAC、PPO、Model-based RL 的简要补充
9. RL 在真实工程中的困难

---

## 1. Recap：什么是 Inverse RL

Inverse RL，简称 IRL，中文通常叫 **逆强化学习**。

普通 RL 的设定是：

```text
给定 reward function
学习 policy
```

Inverse RL 的设定是反过来：

```text
给定专家演示轨迹
学习 reward function
```

定义是：

```text
Given expert demonstration data,
Goal: learn a reward function assuming the experts are optimal.
```

也就是：

```text
假设专家是接近最优的，
从专家行为中推断专家到底在优化什么 reward。
```

### 1.1 直观例子

如果我们观察一个优秀司机：

```text
保持车道
避免急刹
不撞车
尽量快速到达目的地
```

我们不一定知道他的 reward function 是什么。

IRL 希望从这些轨迹中反推出：

```text
不碰撞很重要
舒适性也重要
效率也重要
偏离车道要被惩罚
```

之后再用这个 reward 训练新的 RL policy。

---

## 2. BC、Interactive IL、IRL、RL 对比

这四类方法都和“学习行为”有关，但学习目标不同。

| 方法 | 是否直接学 policy | 是否学 reward | 是否需要环境交互 | 是否需要交互专家 | 是否需要专家轨迹 |
|---|---:|---:|---:|---:|---:|
| Behavioral Cloning | 是 | 否 | 否 | 否 | 是 |
| Interactive IL | 是 | 否 | 是 | 是 | 可选 |
| Inverse RL | 否 | 是 | 是 | 否 | 是 |
| RL | 是 | 否 | 是 | 否 | 否 |

更直白地记：

```text
BC：模仿专家动作
Interactive IL：边执行边问专家
IRL：从专家轨迹反推 reward
RL：用 reward 自己探索学习
```

### 2.1 IRL 为什么有价值

有些任务中，专家行为容易获得，但 reward 很难手写。

例如机器人操作：

```text
什么叫“自然地插入插头”？
什么叫“稳定地抓取物体”？
什么叫“像人一样驾驶”？
```

这些目标很难直接写成 reward，但可以通过专家轨迹间接学习。

---

## 3. IRL 的基本形式：线性 Reward

IRL 的一个常见假设是：reward 是特征的线性组合。

### 3.1 Feature 是什么

feature 是从状态中提取出来的描述量：

```text
phi(s)
```

例如自动驾驶中，feature 可以包括：

1. 距离车道中心的偏移
2. 与前车距离
3. 速度大小
4. 加速度变化
5. 是否接近障碍物

这些 feature 可以是状态的非线性映射。

### 3.2 线性 reward

线性 reward 假设写作：

```text
r(s) = w · phi(s)
```

其中：

| 符号 | 含义 |
|---|---|
| `phi(s)` | 状态特征 |
| `w` | 每个特征的权重 |
| `r(s)` | reward |

IRL 的目标就是学习 `w`。

直观理解：

```text
专家轨迹告诉我们哪些 feature 重要，
IRL 要推断每个 feature 应该给多大权重。
```

### 3.3 Expected Feature

一条轨迹的累计 reward 可以写成：

```text
sum_t r(s_t)
= sum_t w · phi(s_t)
= w · sum_t phi(s_t)
```

所以如果 reward 是线性的，那么比较策略好坏可以转化为比较：

```text
expected feature counts
```

也就是策略平均会访问到哪些特征。

专家策略应该在特征统计上优于其他策略。

---

## 4. LP-IRL：Linear Programming IRL

LP-IRL 指 Linear Programming IRL。

它的核心思想是：

```text
找到一个 reward，使专家策略比其他策略更好。
```

如果专家策略是 `pi*`，其他策略是 `pi_i`，我们希望：

```text
V^{pi*} > V^{pi_i}
```

在 feature 形式下就是：

```text
w · mu_E > w · mu_i
```

其中：

| 符号 | 含义 |
|---|---|
| `mu_E` | 专家策略的 expected features |
| `mu_i` | 其他策略的 expected features |
| `w` | reward 权重 |

### 4.1 LP-IRL 的流程直觉

```text
1. 从专家轨迹计算专家 feature expectation
2. 收集或生成若干其他策略
3. 寻找 reward 权重 w，使专家策略和其他策略差距尽量大
4. 用学到的 reward 训练新策略
5. 如果需要，加入新策略继续迭代
```

其他策略可以来自：

1. 手工策略
2. RL 训练出来的策略
3. 迭代过程中产生的新策略

所以 LP-IRL 可以是迭代式的：

```text
解 reward -> 用 reward 训练 policy -> 加入新 policy -> 再解 reward
```

---

## 5. Maximum Entropy IRL

Maximum Entropy IRL，最大熵逆强化学习，是 IRL 中很经典的一类方法。

### 5.1 为什么需要 maximum entropy

专家轨迹通常不是唯一最优轨迹。

同一个任务可能有很多合理做法：

```text
从左边绕过障碍物
从右边绕过障碍物
稍微慢一点但更稳
稍微快一点但仍然安全
```

如果只要求“专家比其他策略好”，可能会过度偏向某一种轨迹。

Maximum Entropy 的思想是：

```text
在匹配专家行为的同时，尽量保持轨迹分布高熵。
```

也就是不要无理由地过度确定。

### 5.2 MaxEnt IRL 流程

流程可以整理为：

```text
1. 初始化 reward function
   r_theta(s) = theta · phi(s)

2. 给定专家轨迹数据 D

3. 根据当前 reward 求最优 policy
   也就是用当前 reward 训练一个 policy

4. 求当前 policy 的 state visitation frequency
   即它会以多大频率访问各个状态

5. 计算专家和当前 policy 的 feature / visitation 差异

6. 根据梯度更新 theta

7. 重复
```

### 5.3 一句话理解 MaxEnt IRL

```text
学习一个 reward，
让由它诱导出的策略访问状态的方式尽量接近专家，
同时不要对未被数据强约束的选择做过度假设。
```

---

## 6. Offline RL：离线强化学习

Offline RL 是本节第二条主线。

普通 RL 通常需要在线和环境交互：

```text
执行策略 -> 得到数据 -> 更新策略 -> 再执行
```

Offline RL 的限制是：

```text
只能使用预先收集好的数据集
不能再和环境交互
```

可以描述为：

```text
Pre-collected trajectory
No interaction with environment
Data driven RL
```

### 6.1 Offline RL 和 On-policy / Off-policy RL

先回顾：

| 类型 | 特点 | 例子 |
|---|---|---|
| On-policy RL | 必须用当前策略的轨迹评估和更新 | VPG, A2C |
| Off-policy RL | 可以使用其他策略产生的轨迹 | DQN, DDPG |
| Offline RL | 只能用固定离线数据集，不能继续交互 | TD3+BC, CQL |

Offline RL 可以看作更严格的 off-policy：

```text
off-policy：可以用别的策略数据，但通常仍可继续采样
offline RL：只有固定数据，不能再采样新数据
```

### 6.2 Offline RL 和 IL / RL 的关系

可以这样理解：

```text
Offline RL = Imitation Learning + reward + non-expert trajectory
Offline RL = RL - interaction with environment
```

也就是说：

1. 它像 IL，因为都是从已有数据中学习
2. 它比 BC 多了 reward，可以利用非专家数据
3. 它像 RL，因为目标仍然是最大化 reward
4. 它不像 online RL，因为不能继续探索环境

---

## 7. Offline RL 数据集

Offline RL 数据通常由 transition 或 trajectory 组成。

常见格式：

```text
(s, a, s', r)
```

也可以写作：

```text
(s, a, [s'], r)
```

其中：

| 字段 | 含义 |
|---|---|
| `s` | 当前状态 |
| `a` | 执行动作 |
| `s'` | 下一状态 |
| `r` | reward |

相关数据集：

| 数据集 | 说明 |
|---|---|
| D4RL / Minari | 常见 offline RL benchmark / 数据格式 |
| BridgeData | 真实机器人操作数据 |

Offline RL 数据也可以用于 imitation learning：

```text
如果数据质量很高，可以直接做 BC
如果数据质量混杂，可以利用 reward 做选择和优化
```

---

## 8. Offline RL 的核心困难

Offline RL 最大的问题是：

```text
学到的 policy 可能选择数据集中很少出现甚至没出现过的动作。
```

这会导致 Q function 对这些动作估计不可靠。

### 8.1 Distribution Shift

训练数据来自行为策略：

```text
pi_beta
```

学出来的目标策略是：

```text
pi
```

如果 `pi` 选择的数据外动作很多，就会产生分布偏移。

### 8.2 Extrapolation Error

Q 网络可能会对没见过的动作给出过高估计。

于是 policy 会被这些错误高估吸引：

```text
Q 错误地觉得某个未见动作很好
policy 选择这个动作
真实执行效果很差
但 offline 训练时无法通过环境纠正
```

所以 offline RL 通常需要保守约束：

```text
不要太相信数据外动作
不要让 policy 偏离数据分布太远
```

---

## 9. TD3+BC

第一个 offline RL 方法是 TD3+BC。

### 9.1 TD3 是什么

TD3 全称：

```text
Twin Delayed Deep Deterministic Policy Gradient
```

它是 DDPG 的改进版本，主要包含：

| 改进 | 作用 |
|---|---|
| Clipped Double-Q Learning | 用两个 Q 网络，减少 Q 过高估计 |
| Delayed Policy Updates | policy 更新频率低于 Q 更新 |
| Target Policy Smoothing | 给 target action 加噪声，提升稳定性 |

### 9.2 TD3+BC 的思路

TD3+BC 是 TD3 的 offline learning 版本。

它在 actor 训练中加入 BC loss：

```text
policy objective = maximize Q + behavior cloning regularization
```

直观理解：

```text
既希望 policy 选择 Q 值高的动作，
又希望 policy 不要离数据集里的动作太远。
```

### 9.3 为什么加 BC loss

Offline RL 不能在线试错。

如果 actor 只追求 Q 高，可能会选择数据外动作。

加入 BC 项后：

```text
policy 被拉回数据分布附近
```

这能降低 extrapolation error。

一句话总结 TD3+BC：

```text
在 TD3 的 actor 更新里加一个模仿数据动作的正则项。
```

---

## 10. CQL：Conservative Q-Learning

CQL 全称：

```text
Conservative Q-Learning
```

它的核心思想是：

```text
对没怎么见过的动作，保守地估计 Q 值。
```

### 10.1 为什么要保守

Offline RL 中，Q 网络容易对数据外动作过度乐观。

CQL 通过额外 loss 惩罚这种情况：

```text
降低 unseen actions 的 Q 值
保持 dataset actions 的 Q 值相对可靠
```

这样 actor 或策略在选择动作时，就不容易被虚假的高 Q 动作吸引。

### 10.2 CQL 的直觉

可以把 CQL 理解成：

```text
宁愿低估没见过的动作，
也不要高估它们。
```

这非常适合 offline RL，因为没有环境交互来纠正错误估计。

---

## 11. Trajectory Transformer 和 Decision Transformer

还可以用 sequence modeling 做 offline RL。

代表方法：

1. Trajectory Transformer
2. Decision Transformer

### 11.1 基本思想

传统 RL 通常从 value、policy、Bellman equation 出发。

Transformer 方法换了视角：

```text
把轨迹看成一个序列建模问题。
```

轨迹可以表示为：

```text
return, state, action, return, state, action, ...
```

模型学习：

```text
给定历史状态、动作和目标 return，预测下一步动作。
```

### 11.2 Decision Transformer 的直觉

Decision Transformer 可以理解为：

```text
把 RL 问题改写成条件序列生成问题。
```

给模型一个目标：

```text
我想获得高 return
```

模型根据离线数据学习在这种目标下应该输出什么动作序列。

这类方法和前面的大模型、Transformer 章节有联系。

---

## 12. 机器人应用 1：TriFinger 离线 RL

第一个机器人案例是 TriFinger manipulation platform。

任务：

```text
三根手指，每根 3 DoF，共同完成操作任务
```

Observation：

1. 机器人关节位置
2. 机器人关节速度
3. 手指力反馈
4. 相机图像 keypoints
5. 可选原始图像

Action：

```text
机器人控制命令，频率 50 Hz
```

Reward：

```text
kernel distance
```

例如 pushing 任务中，reward 和目标位置距离相关。

特点：

```text
有仿真和真实世界数据，可用于 benchmark offline RL。
```

---

## 13. 机器人应用 2：BridgeData 和少样本微调

第二个案例是多任务机器人操作。

任务：

```text
多个真实机器人 manipulation tasks
```

Observation：

```text
真实世界图像 + 机器人状态
```

机器人平台：

```text
WidowX robot
```

Action：

```text
机器人控制命令
```

训练方式：

```text
用 robot data 做 offline RL pre-training
再对新任务做 few-shot fine tuning
```

数据集：

```text
BridgeData
```

核心价值：

```text
先用大量历史机器人数据学到通用操作能力，
再用少量新任务数据快速适应。
```

---

## 14. 机器人应用 3：从互联网视频到机器人 Offline RL

第三个案例是 Robotic Offline RL from Internet Videos。

它使用两阶段预训练：

```text
Stage 1：用互联网视频预训练
Stage 2：用机器人数据继续预训练
然后 few-shot fine tuning
```

第一阶段学习：

```text
intent-conditioned value function, ICVF
```

直观理解：

```text
互联网视频提供大量关于人类动作、物体交互和任务意图的信息。
机器人数据再把这些信息对齐到可执行的机器人控制上。
```

这种路线说明 Offline RL 正在和大规模视频数据、表征学习结合。

---

## 15. 其他方法：SAC

SAC 全称：

```text
Soft Actor-Critic
```

SAC 可以总结为：

1. 流程上类似 DDPG / actor-critic
2. 在 value function 中加入 entropy 项
3. 使用两个 Q network

### 15.1 Entropy 的作用

Entropy 可以理解为策略随机性。

加入 entropy 项后，目标不只是：

```text
拿高 reward
```

还包括：

```text
保持足够探索
不要过早变得确定
```

所以 SAC 的直觉是：

```text
在最大化奖励的同时，也鼓励策略保持一定随机性。
```

这通常会让训练更稳定，探索更充分。

---

## 16. 其他方法：PPO

PPO 全称：

```text
Proximal Policy Optimization
```

先看 TRPO：

```text
Trust Region Policy Optimization
```

TRPO 的思想是：

```text
每次 policy 更新不要离旧 policy 太远。
```

通常用 KL divergence 定义 trust region。

PPO 把这个约束简化为 clip：

```text
用 clipped surrogate loss 限制策略更新幅度。
```

### 16.1 PPO 的直觉

Policy gradient 如果更新太大，策略可能突然崩掉。

PPO 的核心就是：

```text
允许策略改进，
但限制每次改动不要太激进。
```

Stable-Baselines3 的 PPO 实现中也使用：

1. Entropy
2. GAE，Generalized Advantage Estimation

---

## 17. Model-Based RL

最后补充 Model-Based RL。

### 17.1 Model-Based Planning and Control

传统控制里常见方法包括：

| 方法 | 含义 |
|---|---|
| MPC | Model Predictive Control，模型预测控制 |
| LQR | Linear Quadratic Regulator，线性二次调节器 |

MPC 的直觉是：

```text
利用系统模型预测未来一段时间
求解一个带约束的优化问题
得到当前应该执行的动作
下一步再重新规划
```

这和前面 Ch2 的经典规划控制方法有关。

### 17.2 Model-based RL

Model-based RL 会显式学习或使用环境模型。

常见方向：

```text
学习 dynamics model
用模型做 planning
用模型生成 imagined rollout
```

也可以结合 MCTS。

和 model-free RL 相比：

```text
model-based RL 可能采样效率更高，
但模型误差会影响规划和策略学习。
```

---

## 18. 总结：RL 方法谱系

几类方法可以总结为：

### 18.1 Model-Free RL

Value-based：

```text
Q-learning：
用一步 transition 和 bootstrapping 估计 Q，
通过隐式 argmax policy 选动作。
```

Policy-based：

```text
VPG：
用 policy net 估计 policy gradient，
直接优化策略。
```

Actor-Critic：

```text
用 value net 估计 value / advantage，
用 policy net 输出动作。
```

### 18.2 Offline RL

```text
没有环境交互的 RL，
或者说带 reward 的 imitation-style learning。
```

### 18.3 Inverse RL

```text
不是先给 reward 学 policy，
而是从专家轨迹中学习 reward。
```

---

## 19. 真实机器人中使用 RL 的困难

RL 工程落地通常很难。

### 19.1 训练流程复杂

真实 RL workflow 往往需要：

1. 仿真环境
2. 分布式计算
3. CPU-GPU 协调
4. 大量 rollout
5. 日志、评估和调参系统

### 19.2 Sim-to-real 挑战

机器人通常先在仿真中训练，再迁移到真实世界。

问题是：

```text
仿真世界和真实世界永远有差距。
```

这会导致 sim-to-real gap。

### 19.3 Debugging 困难

RL 难调试的原因：

1. 算法里有很多近似
2. 策略在闭环控制中运行，不只是做预测
3. 评估结果波动大
4. 超参数非常敏感
5. 失败原因可能来自环境、reward、网络、探索或实现细节

### 19.4 为什么 IL 常作为 baseline

实践中可以使用 imitation learning 作为 baseline。

原因是：

```text
如果 BC / IL 都做不好，
直接上复杂 RL 往往更难判断问题在哪里。
```

常见做法：

1. 先用 IL 预训练
2. 再用 RL fine-tune
3. 或在 off-policy 方法中加入 IL 数据

---

## 20. 本节课最重要的理解

### 20.1 IRL 学的是 reward，不是直接学 policy

IRL 的关键假设是：

```text
专家行为背后存在某个 reward function。
```

我们从专家轨迹中反推这个 reward，再用它训练 policy。

### 20.2 Offline RL 的核心约束是不能在线纠错

Online RL 可以探索并从错误中学习。

Offline RL 只能看固定数据：

```text
如果学到的 policy 跑到数据分布之外，
训练时没有环境反馈来纠正。
```

所以 offline RL 方法通常都带有保守性或行为约束。

### 20.3 TD3+BC 和 CQL 都是在防止数据外动作问题

TD3+BC：

```text
让 policy 不要偏离数据动作太远。
```

CQL：

```text
让 Q function 不要高估未见动作。
```

两者都服务于同一个目标：

```text
在没有在线交互的情况下，让学习过程更稳。
```

