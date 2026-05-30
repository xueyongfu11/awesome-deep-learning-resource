# L6 sec2 Q Learning：新手学习课程

---

## 0. 这节课在讲什么

本节讲强化学习中的一类重要方法：

```text
Value-based Method
```

也就是基于价值函数的方法。

它和直接学习策略的 policy-based method 不同。value-based 方法通常不直接输出“当前状态下应该执行什么动作”，而是先学习：

```text
这个状态好不好
或者
在这个状态做这个动作好不好
```

然后再根据价值函数选动作。

本节主线可以整理成：

```text
Value Iteration
  -> Q Function
  -> Fitted Q Iteration
  -> Online Q Learning
  -> Replay Buffer + Target Network
  -> DQN
  -> Double DQN / Dueling DQN / DDPG
  -> 机器人应用
```

一句话概括：

```text
Q-learning 的核心是学习 Q(s, a)，也就是“在状态 s 做动作 a 之后，长期来看有多好”。
```

---

## 1. Value-based Method 是什么

强化学习的目标是学习一个策略：

```text
π(a | s)
```

也就是在状态 `s` 下选择动作 `a`。

但 value-based method 不一定直接学习策略，而是学习价值函数。

常见价值函数有两类：

| 名称 | 形式 | 含义 |
|---|---|---|
| State value function | `V(s)` | 从状态 `s` 出发，长期回报有多大 |
| State-action value function | `Q(s, a)` | 在状态 `s` 做动作 `a` 后，长期回报有多大 |

如果已经学到了 `Q(s, a)`，策略可以隐式得到：

```text
π(s) = argmax_a Q(s, a)
```

意思是：

```text
在当前状态下，选择 Q 值最大的动作。
```

所以说：

```text
Don't learn a policy explicitly.
Learn value or Q-function.
```

---

## 2. 先理解 V(s)

`V(s)` 表示从状态 `s` 开始，按照某种策略行动，未来累计能拿到多少奖励。

例如一个机器人在迷宫里：

```text
离目标越近的格子，V(s) 越大
离障碍物越近或容易失败的格子，V(s) 越小
```

如果知道每个状态的价值，就可以选择让自己走向高价值状态的动作。

### 2.1 Bellman 思想

价值函数最核心的思想是 Bellman equation。

直观上：

```text
当前状态的价值
= 当前一步奖励
+ 折扣后的下一状态价值
```

写成简化形式：

```text
V(s) = max_a [ R(s, a) + γ V(s') ]
```

如果环境有随机性，下一状态不唯一，就要对所有可能的下一状态求期望：

```text
V(s) = max_a Σ_{s'} P(s' | s, a) [ R(s, a) + γ V(s') ]
```

其中：

```text
γ：discount factor，折扣因子
P(s' | s, a)：状态转移概率
```

---

## 3. 从 V(s) 到 Q(s, a)

`V(s)` 只评价状态。

但实际做决策时，我们更关心：

```text
在当前状态下，哪个动作最好？
```

所以引入 Q 函数：

```text
Q(s, a)
```

它表示：

```text
在状态 s 执行动作 a，然后继续按最优方式行动，最终能得到的长期回报。
```

Q 函数的 Bellman 更新可以写成：

```text
Q_{k+1}(s, a)
= Σ_{s'} P(s' | s, a) [ R(s, a) + γ max_{a'} Q_k(s', a') ]
```

拆开理解：

```text
先在 s 做 a，得到当前奖励 R(s, a)
环境转移到 s'
下一步开始选择 Q 值最大的动作 a'
把未来价值打折 γ
对所有可能的 s' 求期望
```

如果环境是确定性的，可以简化成：

```text
Q(s, a) = R(s, a) + γ max_{a'} Q(s', a')
```

---

## 4. Value Iteration 和 Q Iteration

Value Iteration 是一种动态规划方法。

它反复更新价值函数：

```text
V_0 -> V_1 -> V_2 -> ... -> V*
```

每次更新都用 Bellman optimality backup。

Q Iteration 类似，只是更新的是：

```text
Q(s, a)
```

最终得到最优 Q 函数：

```text
Q*(s, a)
```

再用：

```text
π*(s) = argmax_a Q*(s, a)
```

得到最优策略。

### 4.1 它的限制

标准 Q Iteration 需要知道系统动力学：

```text
P(s' | s, a)
```

也就是：

```text
在状态 s 做动作 a，会以多大概率到达每个 s'
```

但真实机器人任务里，系统动力学往往不知道，或者太复杂。

例如：

```text
机械臂推一个物体后，物体具体怎么滑动？
移动机器人碰到杂物后，环境怎么变化？
自动驾驶车辆换道后，其他车怎么反应？
```

这些都很难精确写出 `P(s' | s, a)`。

所以要引入基于数据的近似方法。

---

## 5. Fitted Q Iteration

Fitted Q Iteration 可以理解为：

```text
用采集到的 transition 数据近似 Q Iteration。
```

一条 transition 数据是：

```text
(s, a, s', r)
```

含义是：

```text
在状态 s 执行动作 a
环境到了下一状态 s'
获得奖励 r
```

标准 Q Iteration 需要系统动力学 `P(s' | s, a)`。

Fitted Q Iteration 不显式知道动力学，而是用样本来近似。

### 5.1 训练目标

给定一条样本：

```text
(s, a, s', r)
```

当前 Q 网络输出：

```text
Q(s, a)
```

目标值可以设为：

```text
y = r + γ max_{a'} Q(s', a')
```

训练就是让网络输出接近这个目标：

```text
min [ Q(s, a) - y ]^2
```

这就是 Q-learning 中最常见的一步 target。

### 5.2 为什么叫 fitted

因为它把 Q 函数学习变成了一个拟合问题。

```text
输入：(s, a)
标签：r + γ max Q(s', a')
模型：函数近似器，比如神经网络
损失：预测 Q 和目标 Q 的误差
```

所以 fitted Q iteration 是：

```text
反复构造 target
反复拟合 Q function
```

### 5.3 Off-policy 特性

需要强调：

```text
Off-policy: no need to collect samples from current policy.
```

意思是：

```text
训练 Q 函数的数据，不一定必须来自当前正在学习的策略。
```

例如数据可以来自：

```text
随机策略
旧版本策略
人类示范
其他控制器
历史 replay buffer
```

这对机器人很重要，因为真实机器人采样成本高，能复用旧数据会更省样本。

---

## 6. Online Q Learning

Fitted Q Learning 可以用固定数据集训练。

Online Q Learning 则是在和环境交互的过程中边采样边更新。

流程是：

```text
1. 当前状态 s
2. 根据某个行为策略选择动作 a
3. 执行动作，得到 s' 和 r
4. 用 (s, a, s', r) 更新 Q 函数
5. 进入下一状态，重复
```

注意：采样用的策略不一定是：

```text
argmax_a Q(s, a)
```

它可以加入探索。

例如：

```text
大多数时候选择 Q 值最大的动作
少数时候随机选动作
```

这就是后面 DQN 里常见的 epsilon-greedy。

---

## 7. Replay Buffer

Replay Buffer 是 Q-learning 和 DQN 中非常重要的工程组件。

它做的事很简单：

```text
把交互产生的 transition 存起来
```

每条数据形如：

```text
(s, a, s', r)
```

训练时不是只用最新一条，而是从 buffer 中采样一批历史数据。

### 7.1 为什么需要 Replay Buffer

主要有两个原因。

第一，提高样本效率：

```text
真实交互很贵，一条数据应该被多次使用。
```

第二，降低样本相关性：

```text
连续采集的数据高度相关。
如果直接按时间顺序训练，神经网络容易不稳定。
随机从 replay buffer 采样可以打乱相关性。
```

所以 replay buffer 让训练更接近普通监督学习中的 mini-batch 训练。

---

## 8. Target Network

要点如下：

```text
Target Network stabilizes the moving target.
```

这句话很关键。

Q-learning 的 target 是：

```text
y = r + γ max_{a'} Q(s', a')
```

但这里的 `Q` 本身也是正在训练的神经网络。

这会导致一个问题：

```text
模型一边学习，一边改变自己的标签。
```

这就是 moving target。

### 8.1 Target Network 的做法

DQN 通常维护两个网络：

```text
Online network：Q(s, a; θ)
Target network：Q(s, a; θ^-)
```

online network 用来更新参数。

target network 用来计算训练目标：

```text
y = r + γ max_{a'} Q(s', a'; θ^-)
```

target network 的参数不是每一步都更新，而是隔一段时间从 online network 复制：

```text
θ^- <- θ
```

这样 target 变化得慢一些，训练更稳定。

---

## 9. General Q Learning 流程

的 general Q learning 可以整理为：

```text
循环：
1. 采样 transition：(s, a, s', r)
2. 构造 1-step target：
   y = r + γ max_{a'} Q_target(s', a')
3. 计算当前网络输出：
   Q_online(s, a)
4. 最小化误差：
   L = [Q_online(s, a) - y]^2
5. 用梯度下降更新 Q 网络
6. 根据需要更新 target network
```

这里最重要的是 1-step transition target。

它只看一步真实环境反馈：

```text
当前奖励 r
下一状态 s'
```

再用当前估计的 Q 函数补上未来价值。

---

## 10. DQN

DQN 是 Deep Q-Network。

它把 Q-learning 和深度神经网络结合起来：

```text
Q table -> Q network
```

传统 Q-learning 可以为每个 `(s, a)` 存一个表格值。

但 Atari 游戏或机器人图像输入中，状态空间太大，没法用表格。

所以 DQN 用神经网络近似：

```text
Q(s, a; θ)
```

### 10.1 DQN 在 Atari 游戏中的设置

DQN 的经典应用是 Atari game。

可以整理为：

| 项目 | Atari DQN 设置 |
|---|---|
| Task | 玩 Atari 游戏 |
| Observation | 连续游戏帧，经典设置中常用 4 帧 |
| Action | 离散控制输入，类似人类按键 |
| Reward | 游戏分数变化 |
| Exploration | epsilon-greedy |

为什么使用连续帧？

因为单帧图像可能看不出速度。

例如只看一张图，很难知道球在往左还是往右运动。连续 4 帧可以提供运动信息。

### 10.2 epsilon-greedy

epsilon-greedy 是 DQN 常用探索策略。

```text
以 1 - ε 的概率选择 argmax_a Q(s, a)
以 ε 的概率随机选择动作
```

作用是：

```text
既利用当前学到的 Q 函数
又保留探索新动作的机会
```

如果完全 greedy，模型可能过早陷入局部策略。

---

## 11. DQN 的过估计问题

课程要点如下：

```text
Overestimate of Q value
Because of errors in max operation
```

Q-learning target 中有一项：

```text
max_{a'} Q(s', a')
```

如果 Q 估计有噪声，`max` 操作容易选中被高估的动作。

例如真实 Q 值都差不多：

```text
动作 A：真实 10，估计 12
动作 B：真实 10，估计 9
动作 C：真实 10，估计 8
```

`max` 会选择估计为 12 的 A，于是 target 也偏高。

长期迭代后，Q 值可能系统性过估计。

---

## 12. Double DQN

Double DQN 的目标是缓解过估计。

核心思想是：

```text
动作选择和动作评估分开。
```

普通 DQN target：

```text
y = r + γ max_{a'} Q_target(s', a')
```

Double DQN 可以写成：

```text
a* = argmax_{a'} Q_online(s', a')
y = r + γ Q_target(s', a*)
```

也就是：

```text
online network 负责选哪个动作最好
target network 负责评估这个动作的价值
```

它可以用 online network 和 target network 实现。

---

## 13. Dueling DQN

Dueling DQN 改的是网络结构。

它把 Q 值拆成两部分：

```text
Q(s, a) = V(s) + A(s, a)
```

其中：

```text
V(s)：状态本身有多好
A(s, a)：在这个状态下，某个动作相对其他动作有多好
```

`A` 叫 advantage function。

### 13.1 为什么这样拆

有些状态下，动作选择不太重要。

例如在 Atari 游戏里，某些画面中无论按哪个键，短期结果都差不多。

这时先学状态价值 `V(s)` 会更有效。

Dueling DQN 让网络分别学习：

```text
这个状态整体好不好
这个动作相对好不好
```

再合成 Q 值。

说它可以和大多数 DQN-like 结构一起使用。

---

## 14. 连续动作问题与 DDPG

DQN 适合离散动作空间。

例如 Atari：

```text
上、下、左、右、开火
```

但机器人控制里，动作往往是连续的：

```text
关节速度
末端位移
夹爪力
转向角
加速度
```

这时 `max_a Q(s, a)` 很难直接枚举，因为动作有无限多个。

### 14.1 DDPG 是什么

课程要点如下：

```text
DDPG: Deep neural network version of DPG
Continuous version of DQN
```

DDPG 是 Deep Deterministic Policy Gradient。

它通常包含：

```text
Actor：输出连续动作 a = μ(s)
Critic：评估 Q(s, a)
```

可以理解为：

```text
DQN 用 Q 函数在离散动作中选最大值
DDPG 用 actor 直接给出连续动作，再由 critic 评价
```

所以 DDPG 更适合连续控制任务。

---

## 15. Q-learning 在机器人中的应用

课程最后列了几个机器人和自动驾驶应用案例。

这些案例的共同点是：

```text
把感知输入映射到动作价值 Q(s, a)，再选 Q 值高的动作执行。
```

---

## 16. 应用案例一：Push and Grasp

任务：

```text
在杂乱环境中完成多个物体抓取，可以同时使用 push 和 grasp。
```

这是机器人操作里很典型的问题。

只抓取有时不够，因为物体可能：

```text
互相遮挡
太贴近
夹爪插不进去
姿态不适合直接抓
```

所以机器人需要学会：

```text
什么时候推
往哪里推
什么时候抓
从什么方向抓
```

### 16.1 状态、动作、奖励

给出的设置：

| 项目 | 内容 |
|---|---|
| Observation | RGB-D image |
| Action: pushing | 从某个起点出发，沿 `k = 16` 个方向之一推 10cm |
| Action: grasping | 2D 平面抓取，选择 `k = 16` 个方向之一 |
| Reward | 抓取成功 `R = 1`；推动作造成可检测变化 `R = 0.5` |

这里动作虽然来自机器人连续空间，但被离散化了：

```text
16 个推的方向
16 个抓的方向
```

这样就可以用类似 DQN 的 value-based 方法。

---

## 17. 应用案例二：Vision-based Robot Reaching

任务：

```text
机器人 reaching，也就是控制机械臂到达目标位置。
```

给出的设置：

| 项目 | 内容 |
|---|---|
| Observation | 原始像素输入 |
| Action space | 9 个离散动作 |
| Action detail | 每个关节有三个选项：角度增加、角度减少、保持 |
| Step size | 每次增减 `0.02 rad` |

如果有 3 个关节，每个关节 3 个选项：

```text
increase / decrease / hold
```

就可以得到离散动作组合。

这种做法的好处是：

```text
把连续控制问题离散化，从而能用 Q-learning。
```

缺点是：

```text
动作粒度固定，控制可能不够平滑；
关节数增加时，动作组合会快速变多。
```

---

## 18. 应用案例三：Mobile Manipulation with Spatial Action Maps

任务：

```text
移动机器人把物体推到目标 receptacle 中。
```

要点如下：

| 项目 | 内容 |
|---|---|
| Observation | 局部 bird's-eye view 的 4 通道视觉图像 |
| Action | 用 action map 表示 |
| Reward | 由多个部分组成 |

奖励包括：

```text
1. 每个物体到达目标给 +1
2. 物体更接近目标给部分奖励，远离目标给惩罚
3. 碰撞或不移动等不良行为给小惩罚，例如 -0.25
```

这个案例说明机器人任务里的 reward 常常不是单一项，而是由多个目标组合。

例如：

```text
完成任务
接近目标
避免碰撞
避免无效动作
```

---

## 19. 应用案例四：自动驾驶换道

任务：

```text
Lane change decision-making
```

也就是自动驾驶中的换道决策。

状态表示可以把周围车辆填入网格：

```text
用若干 cell 表示周围空间
每辆车对应的 cell 中填入归一化速度
```

这类方法把交通场景变成结构化状态输入。

动作可以包括：

```text
保持车道
向左换道
向右换道
加速
减速
```

奖励通常会综合考虑：

```text
安全
效率
舒适
规则约束
是否完成换道目标
```

还提到 rule-based constraints，说明自动驾驶中的深度强化学习通常不会完全无约束地探索，而会结合规则约束保证安全。

---

## 20. Q-learning 的优点和局限

### 20.1 优点

Q-learning 的优点包括：

- 不需要显式学习策略，策略可以由 `argmax Q(s, a)` 得到
- 可以 off-policy 学习，复用旧数据
- 和 replay buffer 结合后样本效率更高
- 适合离散动作问题
- DQN 可以处理图像等高维输入

### 20.2 局限

主要局限包括：

- 对连续动作不方便，需要 DDPG、SAC 等 actor-critic 方法
- `max` 操作可能导致 Q 值过估计
- 神经网络训练不稳定，需要 target network 等技巧
- reward 设计仍然很关键
- 在真实机器人上探索有安全和成本问题

---

## 21. 本节核心公式整理

### 21.1 策略由 Q 函数隐式得到

```text
π(s) = argmax_a Q(s, a)
```

### 21.2 Q 函数 Bellman 更新

```text
Q(s, a) = R(s, a) + γ max_{a'} Q(s', a')
```

随机环境下：

```text
Q(s, a)
= Σ_{s'} P(s' | s, a) [ R(s, a) + γ max_{a'} Q(s', a') ]
```

### 21.3 Q-learning target

```text
y = r + γ max_{a'} Q_target(s', a')
```

### 21.4 Q-learning loss

```text
L = [ Q_online(s, a) - y ]^2
```

### 21.5 Double DQN target

```text
a* = argmax_{a'} Q_online(s', a')
y = r + γ Q_target(s', a*)
```

---

## 22. 易混点整理

### 22.1 Q-learning 不是直接输出动作

Q 网络输出的是动作价值。

动作是通过：

```text
argmax_a Q(s, a)
```

选出来的。

### 22.2 Fitted Q Iteration 和标准 Q Iteration 的区别

标准 Q Iteration：

```text
需要知道环境动力学 P(s' | s, a)
```

Fitted Q Iteration：

```text
从采集到的 transition 数据中学习
```

### 22.3 Replay Buffer 不是模型

Replay Buffer 只是存储数据的容器。

它的作用是：

```text
复用样本
打乱样本相关性
提高训练稳定性
```

### 22.4 Target Network 不是另一个独立智能体

Target Network 是 online network 的延迟副本。

它用来计算更稳定的训练目标。

### 22.5 DQN 主要适合离散动作

如果动作是连续的，直接 `max_a Q(s, a)` 很难做。

这时通常需要：

```text
DDPG
TD3
SAC
```

这类 actor-critic 方法。

---

## 23. 和其他强化学习方法的关系

可以按“学什么”来区分：

| 方法类型 | 学习对象 | 典型方法 |
|---|---|---|
| Value-based | `V(s)` 或 `Q(s, a)` | Q-learning, DQN |
| Policy-based | 直接学习 `π(a | s)` | Policy Gradient |
| Actor-Critic | 同时学策略和价值 | DDPG, PPO, SAC |

Q-learning 属于 value-based。

DDPG 虽然说是连续版 DQN，但从结构上看已经是 actor-critic：

```text
Actor 负责选连续动作
Critic 负责估计 Q 值
```

---

## 24. 学习本节时的主线

建议按下面这条线理解：

```text
1. 强化学习想最大化长期奖励
2. Q(s, a) 表示某个状态动作的长期价值
3. 有了 Q，就能用 argmax 选动作
4. Bellman equation 让 Q 可以递推更新
5. 真实环境没有精确动力学，所以用数据样本学习 Q
6. 神经网络版本就是 DQN
7. 为了稳定训练，需要 replay buffer 和 target network
8. 为了修正问题，又有 Double DQN、Dueling DQN 等扩展
9. 在机器人中，经常把动作离散化后用 Q-learning
```

