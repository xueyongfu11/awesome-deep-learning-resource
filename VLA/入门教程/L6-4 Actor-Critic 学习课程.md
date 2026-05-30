# L6-4 Actor-Critic：新手学习课程

---

## 0. 这节课在讲什么

本节讲强化学习里非常重要的一类方法：

```text
Actor-Critic
```

它的核心思想是：

```text
同时学习一个策略网络和一个价值网络。
```

其中：

```text
Actor：负责选择动作，也就是 policy network
Critic：负责评价动作或状态，也就是 value network
```

可以把前面几类方法放在一起看：

| 方法 | 学什么 | 直观理解 |
|---|---|---|
| Q-learning / DQN | 只学价值函数 `Q(s, a)` | value net-only |
| Policy Gradient / VPG | 只学策略 `π(a|s)` | actor-only |
| Actor-Critic | 同时学策略和价值 | actor + critic |

一句话概括：

```text
Actor 负责做决策，Critic 负责告诉 Actor 这个决策好不好。
```

---

## 1. 从 Q-learning 到 Actor-Critic

Q-learning 是 value-based method。

它学习：

```text
Q(s, a)
```

然后用：

```text
π(s) = argmax_a Q(s, a)
```

得到隐式策略。

也就是说，Q-learning 不直接训练一个 policy network。

### 1.1 Q-learning 的优点

Q-learning 的好处是：

- 可以 off-policy 学习
- 可以使用 replay buffer
- 对离散动作很自然
- 学到 `Q(s, a)` 后可以直接比较不同动作的好坏

### 1.2 Q-learning 的问题

但它也有问题：

```text
如果动作空间是连续的，argmax_a Q(s, a) 很难算。
```

例如机器人控制动作可能是：

```text
机械臂末端位移
关节速度
夹爪力
车辆加速度和转向角
```

这些动作不是有限个按钮，而是连续值。

所以只靠 `argmax Q` 不方便。

这时可以让一个 actor 网络直接输出动作：

```text
a = πθ(s)
```

再用 critic 网络评价这个动作：

```text
Qφ(s, a)
```

这就是 actor-critic 的基本直觉。

---

## 2. 从 Policy Gradient 到 Actor-Critic

Policy Gradient 直接学习策略：

```text
πθ(a | s)
```

目标是让策略产生更高回报的轨迹。

Vanilla Policy Gradient，简称 VPG，常用 Monte Carlo return 估计每个动作到底好不好。

### 2.1 VPG 的基本形式

策略梯度可以直观写成：

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * G_t ]
```

其中：

```text
G_t：从 t 时刻开始的累计回报，也叫 return 或 reward-to-go
```

这条公式的意思是：

```text
如果某个动作后面带来了高回报，
就提高策略以后选择这个动作的概率。

如果某个动作后面回报很差，
就降低策略选择它的概率。
```

### 2.2 VPG 的问题

VPG 依赖 Monte Carlo return。

也就是要等一条轨迹跑完，再计算：

```text
G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
```

这会带来两个问题：

- 方差比较大
- 样本利用率不高

所以需要一个价值网络帮助估计回报或优势函数。

这就是 critic 的作用。

---

## 3. Actor 和 Critic 分别是什么

Actor-Critic 中有两个网络。

### 3.1 Actor

Actor 是策略网络。

它负责根据状态输出动作。

离散动作中：

```text
πθ(a | s) -> 每个动作的概率
```

连续动作中：

```text
πθ(s) -> 一个连续动作
```

例如机械臂任务中，actor 可以输出：

```text
末端执行器向 x, y, z 方向移动多少
夹爪是否闭合
关节目标角度
```

### 3.2 Critic

Critic 是价值网络。

它可以估计不同形式的价值：

```text
V(s)
Q(s, a)
A(s, a)
```

其中：

| 符号 | 含义 |
|---|---|
| `V(s)` | 当前状态本身有多好 |
| `Q(s, a)` | 当前状态执行动作 `a` 后有多好 |
| `A(s, a)` | 动作 `a` 比这个状态下的平均动作好多少 |

critic 的输出不是直接执行的动作，而是给 actor 提供训练信号。

---

## 4. Advantage Function

Actor-Critic 中经常使用 advantage function：

```text
A(s, a) = Q(s, a) - V(s)
```

它表示：

```text
在状态 s 下，动作 a 比平均水平好多少。
```

如果：

```text
A(s, a) > 0
```

说明这个动作比当前状态下的平均动作好，应该提高它的概率。

如果：

```text
A(s, a) < 0
```

说明这个动作相对不好，应该降低它的概率。

### 4.1 为什么不用 Q 直接更新

如果直接用 `Q(s, a)`，策略更新会受到状态本身好坏的影响。

例如：

```text
状态 s1 本来就很容易成功，所有动作 Q 都高
状态 s2 本来很危险，所有动作 Q 都低
```

但我们真正想知道的是：

```text
在同一个状态下，这个动作是否比其他动作更好？
```

所以 advantage 更适合指导 policy gradient。

---

## 5. Baseline 到 Critic

VPG 中常见一个技巧：

```text
加入 baseline 降低方差
```

策略梯度可以写成：

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * (G_t - b(s_t)) ]
```

如果 baseline 取：

```text
b(s_t) = V(s_t)
```

就得到：

```text
G_t - V(s_t)
```

这就是 advantage 的一种估计。

要点如下：

```text
VPG with Baseline 使用 value net 估计 baseline。
Actor-Critic 则进一步用 value net 估计 V、Q 或 A。
```

也就是说：

```text
baseline 技巧逐渐发展成 critic 网络。
```

---

## 6. Actor-Critic 的策略梯度形式

Actor-Critic 一般仍然遵循 policy gradient 的训练逻辑：

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * A(s_t, a_t) ]
```

这里的关键变化是：

```text
A(s_t, a_t) 不一定由完整 Monte Carlo return 得到，
而是由 critic 网络估计。
```

所以可以理解为：

```text
VPG：用真实轨迹累计回报估计动作好坏
Actor-Critic：用 critic 估计动作好坏
```

---

## 7. Critic 如何训练

critic 通常用 temporal difference，简称 TD，来训练。

以状态价值函数 `V(s)` 为例，一步 TD target 是：

```text
y = r + γ V(s')
```

训练目标是：

```text
min [ V(s) - y ]^2
```

也就是：

```text
当前状态价值 V(s)
应该接近
当前奖励 r + 折扣后的下一状态价值 γV(s')
```

这和 Q-learning 的 Bellman 更新很像。

### 7.1 TD error

TD error 可以写成：

```text
δ = r + γ V(s') - V(s)
```

它有两个用途：

1. 训练 critic
2. 作为 advantage 的一种估计

直观上：

```text
如果实际得到的 r + γV(s') 比 V(s) 大，
说明这一步比预期更好。

如果实际得到的 r + γV(s') 比 V(s) 小，
说明这一步比预期更差。
```

所以：

```text
A(s, a) ≈ δ
```

是很常见的一步 advantage 估计。

---

## 8. Actor-Critic 的基本流程

的 procedure 可以整理为：

```text
循环：
1. 用当前 actor 与环境交互，采样 transition 或 trajectory
2. 用采样数据训练 critic
   例如用 1-step TD target 或 reward-to-go
3. 用 critic 估计 advantage
4. 用 advantage 估计 policy gradient
5. 更新 actor
6. 重复
```

一条 transition 是：

```text
(s, a, r, s')
```

实际实现中通常不是只用一条 transition，而是用一个 batch：

```text
[(s1, a1, r1, s1'), ..., (sn, an, rn, sn')]
```

### 8.1 Actor 和 Critic 谁先更新

常见做法是：

```text
先根据 transition 更新 critic
再用更新后的 critic 估计 advantage
最后更新 actor
```

但不同算法实现会有差异。

核心不变：

```text
critic 提供评价信号，actor 根据评价信号改进策略。
```

---

## 9. Actor-Critic 为什么通常是 On-policy

需要强调：

```text
Actor critic is on policy,
because the policy evaluation requires the samples from current policy.
```

意思是：

```text
要评估当前 actor 的策略梯度，最好使用当前 actor 采样出来的数据。
```

因为策略梯度公式里有：

```text
log πθ(a | s)
```

它要求数据分布和当前策略匹配。

如果样本来自很久以前的旧策略，梯度估计就会偏。

### 9.1 实践中为什么会稍微 off-policy

实际工程中，为了效率，采样和训练常常不是完全同步。

例如：

```text
多个环境并行采样
采样线程和训练线程异步运行
使用 replay buffer 重用旧数据
```

这样会让算法在实现上变得 slightly off-policy。

要点如下：

```text
A2C 在实现中可能稍微 off-policy
A3C 更明显地引入异步和旧样本
```

---

## 10. Batched Actor-Critic

Batched Actor-Critic 指的是用 batch 数据更新网络。

要点如下：

```text
AC only requires transition data to estimate policy gradient and target for value function.
```

也就是说：

```text
actor 的更新需要 transition 或 trajectory
critic 的更新也可以用 transition 构造 TD target
```

所以采样和训练可以在实现上稍微解耦。

### 10.1 同步并行 Actor-Critic

一种做法是：

```text
同时运行 N 个环境或线程
每个环境都用当前策略采样
收集一批数据后统一更新网络
```

优点：

```text
数据更丰富，batch 更稳定。
```

问题：

```text
需要多个环境或线程，成本较高。
```

### 10.2 异步并行 Actor-Critic

另一种做法是：

```text
多个 worker 异步采样和更新
```

优点：

```text
效率更高。
```

问题：

```text
不同 worker 使用的策略参数可能不是完全最新的，
因此样本会稍微 off-policy。
```

---

## 11. A2C 和 A3C

A2C 和 A3C 都是 Advantage Actor-Critic 的代表方法。

### 11.1 A2C

A2C 通常指：

```text
Advantage Actor-Critic
```

可以理解为同步版本。

它用 advantage 来更新 actor，同时训练 critic。

典型结构：

```text
多个环境同步采样
合并 batch
计算 advantage
更新 actor 和 critic
```

### 11.2 A3C

A3C 是：

```text
Asynchronous Advantage Actor-Critic
```

A3C 的特点包括：

- 类似 Q-learning 使用旧样本
- 使用当前策略计算 log probability
- 异步采样和学习
- 更高效

可以理解为：

```text
多个 worker 各自和环境交互，
异步把梯度或参数更新到共享网络。
```

这样能减少单个环境采样慢的问题。

---

## 12. Replay Buffer 在 Actor-Critic 中的角色

在 Q-learning 中，replay buffer 很自然，因为 Q-learning 是 off-policy。

Actor-Critic 理论上更偏 on-policy，所以 replay buffer 使用要更谨慎。

A3C 中的 replay buffer：

```text
samples from previous policy
```

这说明它会复用旧策略样本，因此带来 off-policy 成分。

### 12.1 为什么还要用旧样本

因为强化学习采样成本高。

尤其是真实机器人中：

```text
每一次试错都耗时
可能损坏物体或机器人
需要人工复位环境
```

复用旧样本能提高效率。

但代价是：

```text
数据不完全来自当前策略，理论上会引入偏差。
```

---

## 13. GAE：Generalized Advantage Estimation

要点如下：

```text
GAE: General Advantage Estimation
From 1-step TD to general n-step TD estimation
```

GAE 是估计 advantage 的一种常用方法。

它的目标是在两个极端之间折中：

```text
1-step TD：方差低，但偏差可能大
Monte Carlo return：偏差低，但方差大
```

### 13.1 1-step TD advantage

最简单的 advantage 估计是：

```text
A_t ≈ δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

优点：

```text
只需要一步 transition，方差低。
```

缺点：

```text
依赖 V(s) 的估计，如果 critic 不准，会有偏差。
```

### 13.2 Monte Carlo advantage

也可以用完整 return：

```text
A_t = G_t - V(s_t)
```

优点：

```text
更接近真实累计回报。
```

缺点：

```text
要等轨迹结束，方差大。
```

### 13.3 GAE 的直觉

GAE 把多步 TD error 按权重加起来：

```text
A_t^{GAE} = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
```

其中 `λ` 控制偏差和方差的折中。

可以粗略理解为：

```text
λ 越小，越接近 1-step TD，方差低但偏差大
λ 越大，越接近 Monte Carlo，偏差低但方差大
```

GAE 在 PPO 等现代 actor-critic 算法中很常见。

---

## 14. Actor-Critic 小结

的 summary 可以整理为：

```text
Actor-Critic = Policy Gradient with Value Network
```

它相比 VPG 的关键变化是：

```text
用 value network 替代纯 Monte Carlo return 估计。
```

Actor-Critic 仍然像 policy gradient 一样更新策略，但 advantage 或 return 由 critic 提供。

### 14.1 Advantage 的不同估计方式

advantage 可以有多种形式：

```text
Monte Carlo estimation
Q(s, a)
Q(s, a) - V(s)
r + γV(s') - V(s)
GAE
```

它们都在回答同一个问题：

```text
当前动作比预期好多少？
```

---

## 15. DDPG 和 Actor-Critic 的关系

DDPG 也同时使用 Q-net 和 policy net。

DDPG 可以看成一种 actor-critic 方法：

```text
Actor：输出连续动作 a = μ(s)
Critic：估计 Q(s, a)
```

但它的训练逻辑更接近 Q-learning：

```text
critic 用 Q-learning 式的 TD target 更新
actor 学习让 Q(s, μ(s)) 更大
```

所以 DDPG 兼具：

```text
value-based 的 critic 更新
policy-based 的 actor 更新
```

这也是 actor-critic 方法适合连续控制的重要原因。

---

## 16. PPO、SAC 与 Actor-Critic

相关现代算法包括：

```text
PPO
SAC
async NAF
```

这里重点理解 PPO 和 SAC。

### 16.1 PPO

PPO 是 Proximal Policy Optimization。

它通常属于 on-policy actor-critic。

特点是：

```text
限制每次策略更新幅度，避免策略变化太大导致训练崩掉。
```

PPO 常用于：

```text
机器人装配
灵巧手操作
自动驾驶决策
```

### 16.2 SAC

SAC 是 Soft Actor-Critic。

它是 off-policy actor-critic 方法。

特点是引入 entropy regularization：

```text
不仅追求高奖励，也鼓励策略保持一定随机性。
```

直观上：

```text
不要太早变得过于确定，多探索一些可能动作。
```

SAC 在连续控制和机器人任务中非常常用。

---

## 17. 机器人应用一：Robotic Manipulation with SAC

一个机器人操作任务是：

```text
pick and place、pushing 等 robotic manipulation
```

设置可以整理为：

| 项目 | 内容 |
|---|---|
| Task | 机器人操作，例如抓取、放置、推动 |
| Observation | 图像输入和机器人状态 |
| Feature | auto-encoder with keypoints |
| Action | 笛卡尔空间中的机器人位置 |
| Algorithm | SAC |
| Experiment | 仿真和真实机器人 |

这里的关键点是：

```text
图像输入通常不能直接简单使用，需要提取稳定特征。
```

keypoints 表示可以帮助模型关注物体和机器人之间的关键几何关系。

---

## 18. 机器人应用二：IndustReal 装配任务

IndustReal 的核心是：

```text
Transferring Contact-Rich Assembly Tasks from Simulation to Reality
```

任务：

```text
机器人装配，尤其是接触丰富的 assembly tasks。
```

设置：

| 项目 | 内容 |
|---|---|
| Observation | 关节角、夹爪或物体姿态、目标姿态 |
| Action | 任务空间阻抗控制器的增量位姿目标 |
| Low-level controller | task-space impedance controller |
| Algorithm | PPO |

### 18.1 为什么使用低层控制器

真实机器人中，强化学习策略通常不直接输出电机电流。

更常见的是：

```text
RL policy 输出高层动作
低层控制器负责稳定执行
```

例如：

```text
policy 输出目标位姿增量 dx, dq
阻抗控制器把目标变成平滑、稳定的接触运动
```

这样更安全，也更容易从仿真迁移到真实机器人。

---

## 19. 机器人应用三：Dextreme 灵巧手操作

Dextreme 的核心是：

```text
Transfer of agile in-hand manipulation from simulation to reality
```

任务：

```text
机器人灵巧手 in-hand manipulation
```

设置：

| 项目 | 内容 |
|---|---|
| Observation | 向量状态输入 |
| Action | 16 个手指关节的 PD controller target |
| Algorithm | PPO |
| Experiment | 仿真和真实机器人 |

灵巧手任务很难，因为：

- 自由度多
- 接触复杂
- 物体容易滑动
- 动作需要精细协调

这里策略输出 PD 控制器目标，而不是直接输出底层力矩。

这体现了机器人强化学习中的常见工程原则：

```text
学习高层目标，控制器保证低层稳定。
```

---

## 20. 机器人应用四：异步 Off-policy 机器人操作

课程要点如下：

```text
Deep Reinforcement Learning for Robotic Manipulation with Asynchronous Off-Policy Updates
```

任务包括：

```text
open the door
pick and place
reaching
```

状态表示包括：

```text
关节角
末端执行器位置
这些量的时间导数
目标位置
```

动作：

```text
joint command
```

算法：

```text
async NAF
```

这里的重要点是：

```text
异步 off-policy 更新可以提高真实机器人采样效率。
```

真实机器人采样慢，所以并行、异步、复用旧数据很重要。

---

## 21. 应用五：自动驾驶层次化 Actor-Critic

最后看自动驾驶场景：

```text
Cola-HRL: Continuous-Lattice Hierarchical Reinforcement Learning for Autonomous Driving
```

任务：

```text
自动驾驶车辆到达目标
```

状态输入：

```text
车辆 bounding box
矢量化 HD map
静态障碍物
VectorNet + attention layer
```

动作是层次化策略：

```text
高层策略：在 SL 坐标系中选择目标
低层策略：用 lattice planner 生成轨迹
```

奖励包括：

```text
step reward
termination reward：碰撞、超时、成功
```

算法：

```text
PPO
```

### 21.1 为什么用层次化策略

自动驾驶动作空间很复杂。

直接输出完整轨迹很难。

层次化方法把问题拆成：

```text
高层：决定去哪里
低层：生成可执行轨迹
```

这和机器人操作中的“策略 + 低层控制器”思想一致。

---

## 22. Actor-Critic 的优点和局限

### 22.1 优点

Actor-Critic 的优点包括：

- 能处理连续动作空间
- 比纯 VPG 更省样本
- critic 可以降低策略梯度方差
- 可以结合 TD learning、GAE、replay buffer 等技巧
- 是 PPO、SAC、DDPG、TD3 等现代 RL 方法的基础

### 22.2 局限

主要问题包括：

- actor 和 critic 同时训练，稳定性更复杂
- critic 估计不准会误导 actor
- on-policy 方法样本效率可能低
- off-policy actor-critic 需要处理分布偏差
- 真实机器人探索成本高，安全性要求高

---

## 23. 本节核心公式整理

### 23.1 Policy Gradient

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * G_t ]
```

### 23.2 加 baseline 的 Policy Gradient

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * (G_t - V(s_t)) ]
```

### 23.3 Advantage Function

```text
A(s, a) = Q(s, a) - V(s)
```

### 23.4 TD Error

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
```

### 23.5 Actor-Critic 更新方向

```text
∇θ J(θ) = E[ ∇θ log πθ(a_t | s_t) * A(s_t, a_t) ]
```

### 23.6 GAE

```text
A_t^{GAE} = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
```

---

## 24. 易混点整理

### 24.1 Actor-Critic 不是只有一个网络

它至少有两个功能模块：

```text
Actor：学策略
Critic：学价值
```

有些实现会共享前几层特征，但逻辑上仍然是两个角色。

### 24.2 Critic 不是环境奖励

环境奖励 `r` 是真实反馈。

critic 是模型估计：

```text
V(s)
Q(s, a)
A(s, a)
```

critic 会被训练，也可能估错。

### 24.3 Advantage 不是 reward

reward 是环境给的一步反馈。

advantage 是：

```text
某个动作相对当前状态平均水平好多少。
```

### 24.4 Actor-Critic 和 DQN 的区别

DQN：

```text
只学 Q 网络，通过 argmax 选动作。
```

Actor-Critic：

```text
actor 直接输出动作或动作分布，
critic 负责评价。
```

### 24.5 On-policy 和 off-policy 不只是有没有 replay buffer

核心区别是：

```text
训练数据是否来自当前正在优化的策略。
```

replay buffer 通常意味着复用旧策略数据，所以常引入 off-policy 成分。

---

## 25. 和前后课程的关系

本节处在 value-based 和 policy-based 方法之间。

可以这样串起来：

```text
Q-learning / DQN
  -> 学价值，适合离散动作

Policy Gradient / VPG
  -> 直接学策略，但 Monte Carlo 方差大

Actor-Critic
  -> 同时学策略和价值
  -> 用 critic 帮 actor 降低方差、提高效率

PPO / SAC / DDPG / TD3
  -> 现代 actor-critic 方法
```

如果只记一句：

```text
Actor-Critic 是把 policy gradient 和 value learning 结合起来。
```

---

## 26. 学习本节时的主线

建议按下面顺序理解：

```text
1. Q-learning 只学价值，策略由 argmax 得到
2. VPG 只学策略，动作好坏靠 Monte Carlo return 估计
3. Monte Carlo return 方差大，所以引入 value baseline
4. baseline 扩展成 critic
5. actor 负责选择动作，critic 负责评价动作
6. advantage 告诉 actor 当前动作比预期好多少
7. critic 可以用 TD error 训练
8. A2C/A3C 用并行和异步提高采样效率
9. GAE 在偏差和方差之间折中
10. PPO、SAC、DDPG 等都是 actor-critic 思想的现代延伸
```

---

## 27. 复习问题

### 27.1 概念题

1. Actor 和 Critic 分别负责什么？
2. Actor-Critic 和 Q-learning 的主要区别是什么？
3. Actor-Critic 和 VPG 的主要区别是什么？
4. 为什么 baseline 可以降低 policy gradient 的方差？
5. Advantage function 的含义是什么？
6. TD error 如何用于估计 advantage？
7. 为什么 Actor-Critic 理论上通常是 on-policy？
8. A2C 和 A3C 的区别是什么？
9. GAE 想解决什么问题？
10. 为什么机器人任务中常用 PPO、SAC 这类 actor-critic 方法？

### 27.2 对比题

| 对比 | 关键区别 |
|---|---|
| Q-learning vs Actor-Critic | Q-learning 只学价值，Actor-Critic 同时学策略和价值 |
| VPG vs Actor-Critic | VPG 常用 Monte Carlo return，Actor-Critic 用 critic 估计回报或优势 |
| A2C vs A3C | A2C 偏同步，A3C 偏异步 |
| 1-step TD vs Monte Carlo | 1-step TD 方差低但偏差可能大，Monte Carlo 偏差低但方差大 |
| PPO vs SAC | PPO 常见为 on-policy，SAC 是 off-policy 且带 entropy 正则 |

### 27.3 思考题

如果你要用 Actor-Critic 训练机械臂完成 pick and place，可以思考：

1. Actor 应该输出关节命令、末端位姿，还是低层控制器目标？
2. Critic 应该估计 `V(s)` 还是 `Q(s, a)`？
3. 奖励如何设计才能避免只学到投机行为？
4. 是否需要使用仿真预训练再迁移到真实机器人？
5. 真实机器人上如何减少探索风险？

---

## 28. 一句话总结

Actor-Critic 把策略学习和价值学习结合起来：Actor 学习如何行动，Critic 学习如何评价行动，并用 advantage、TD error、GAE 等方法给 Actor 提供更稳定的更新信号；PPO、SAC、DDPG、A2C/A3C 等现代强化学习算法都可以看作这一思想在不同场景下的具体实现。

