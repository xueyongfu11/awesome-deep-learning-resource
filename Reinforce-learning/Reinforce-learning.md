[TOC]




## Repo
- https://github.com/google/dopamine
- https://github.com/openai/gym

- https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
- https://github.com/aikorea/awesome-rl

- https://github.com/datawhalechina/easy-rl
  - 经典的入门开源书，附代码

- https://github.com/xiaochus/Deep-Reinforcement-Learning-Practice
- https://github.com/openai/baselines

- https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


## reference material

- https://spinningup.qiwihui.com/zh-cn/latest/
  - openAI RL学习资料
- https://www.youtube.com/@HungyiLeeNTU/playlists
  - 李宏毅强化学习课程
- 动手学强化学习
  - 在线书籍：https://hrl.boyuai.com/chapter
  - 视频讲解：https://www.boyuai.com/elites/course/xVqhU42F5IDky94x
  

## Blog

- [强化学习中action value减去state value之后为什么减小了方差](https://www.zhihu.com/question/344367451/answer/1891095552030115567)

## 常见算法

### Q价值函数和状态价值函数

Action-Value function：$Q(s, a)$是agent在状态s下执行某一个动作（如向上走），所获得的及时奖励和未来折扣的累计奖励

State-Value function：$V(s)$是agent在状态s下执行每个动作（上、下、左、右），所获得的加权奖励值（期望奖励值），主要用来评估状态s的好坏，与动作无关

$Q(s, a)$和$V(s)$之间的关系：
$$
V_\pi(s_t)=\mathbb{E}_A\left[Q_\pi(s_t,A)\right]=\sum_a\pi(a|s_t)\cdot Q_\pi(s_t,a).
$$

$$
V_\pi(s_t)=\mathbb{E}_A\left[Q_\pi(s_t,A)\right]=\int\pi(a|s_t)\cdot Q_\pi(s_t,a)da
$$

### Q-Learning

Q-learning使用下一个状态的最优动作来更新Q值
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R(s_t, a_t) + \gamma \max_{a_{t + 1}} Q(s_{t + 1}, a_{t + 1}) - Q(s_t, a_t)]
$$


### Sarsa

Sarsa使用下一个状态的实际动作来更新Q值
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R(s_t, a_t) + \gamma Q(s_{t + 1}, a_{t + 1}) - Q(s_t, a_t)]
$$

### Policy Gradient

策略函数是在给定状态下，给出在该状态下执行各个动作的概率分布。我们的目标是要寻找一个最优策略并最大化这个策略在环境中的期望回报

策略学习的目标函数：$J(\theta)=\mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]$，其中$s_0$是初始状态，然后对目标函数求梯度：

$\nabla_\theta J(\theta)\propto \mathbb{E}_{\pi_\theta}[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)]$

这里不进行具体证明，直观理解一下策略梯度这个公式，可以发现在每一个状态下，梯度的修改是让策略更多地去采样到带来较高Q值的动作，更少地去采样到带来较低Q值的动作。

### REINFORCE

REINFORCE是一种policy gradient算法，使用蒙特卡洛算法估计$Q^{\pi_\theta}(s, a)$

即：$Q^{\pi_\theta}(s_t, a_t) \approx \sum_{t^{\prime}=t}^T\gamma^{t^{\prime}-t}r_{t^{\prime}}$

```python
def update(self, transition_dict):
    reward_list = transition_dict['rewards']  # 一个回合中每个时间步的奖励值
    state_list = transition_dict['states']   # 一个回合中每个时间步的状态
    action_list = transition_dict['actions']   # 一个回合中每个时间步的执行动作

    G = 0
    self.optimizer.zero_grad()
    for i in reversed(range(len(reward_list))):  # 从最后一步算起，主要是应为t时间步的累计奖励依赖第t+1时间步的累计奖励值
        reward = reward_list[i]
        state = torch.tensor([state_list[i]],
                             dtype=torch.float).to(self.device)
        action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
        # 对应公式中的$log\pi_\theta(a|s)$
        log_prob = torch.log(self.policy_net(state).gather(1, action))
        G = self.gamma * G + reward
        # G表示当前时间步的累计奖励
        loss = -log_prob * G  # 每一步的损失函数
        loss.backward()  # 反向传播计算梯度
    self.optimizer.step()  # 梯度下降
```

### Actor-Critic

策略梯度更加一般的形式：
$$
\nabla_\theta J(\theta) &\propto \mathbb{E}_{\pi_\theta}[Q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)] \\
&= \mathbb{E}\left[\sum_{t = 0}^{T} \psi_t \nabla_{\theta} \log \pi_{\theta}(a_t \vert s_t)\right]
$$
其中$\psi_t$可以取的值如下：

1. $\sum_{t' = 0}^{T} \gamma^{t'} r_{t'}$$A^{\pi_{\theta}}(s_t, a_t)$ ---------- 轨迹的总回报
2. $\sum_{t' = t}^{T} \gamma^{t' - t} r_{t'}$$A^{\pi_{\theta}}(s_t, a_t)$ ---------- 动作$a_t$之后的回报 
3. $\sum_{t' = t}^{T} \gamma^{t' - t} r_{t'} - b(s_t)$$A^{\pi_{\theta}}(s_t, a_t)$ ---------- 基准线版本的改进 
4. $Q^{\pi_{\theta}}(s_t, a_t)$$A^{\pi_{\theta}}(s_t, a_t)$ ---------- 动作价值函数 
5. $A^{\pi_{\theta}}(s_t, a_t)$ ---------- 优势函数 
6. $A^{\pi_{\theta}}(s_t, a_t)$ ---------- 时序差分残差

标号2对应到REINFORCE算法采样蒙特卡洛采样的算法，这种方法对策略梯度的估计是无偏的，但是方差比较大

标号3引入baseline，可以降低REINFORCE算法方差过大的问题

标号4对应Actor-Critic算法，使用动态价值函数Q，代替蒙特卡洛采样得到的回报

标号5对Actor-Critic算法进一步改进，把状态价值函数V作为baseline，用Q函数减去V函数，得到A函数，即优势函数A，即：$A=Q^{\pi_{\theta}}(s_t, a_t) - V^{\pi_{\theta}}(s_t)$

标号6对标号5算法进一步改进，利用了$Q=r + \gamma V$，即$r_t + \gamma V^{\pi_{\theta}}(s_{t + 1}) - V^{\pi_{\theta}}(s_t)$



这里介绍基于时序差分残差来指导策略梯度更新的Actor-Critic算法，已知Actor采用策略梯度更新的原则，下面重点介绍Critic的梯度更新原则

将Critic网络表示为$V_w$，参数为$w$，直接采用时序差分残差的学习方式，Critic价值网络的损失是：

$\mathcal{L}(\omega)=\frac{1}{2}(r+\gamma V_\omega(s_{t+1})-V_\omega(s_t))^2$

对应的梯度是：$\nabla_\omega\mathcal{L}(\omega)=-(r+\gamma V_\omega(s_{t+1})-V_\omega(s_t))\nabla_\omega V_\omega(s_t)$

总结Actor - Critic算法的具体流程如下:
- 初始化策略网络参数$\theta$，价值网络参数$\omega$
- for序列$e = 1 \to E$ do:
    - 用当前策略$\pi_{\theta}$采样轨迹$\{s_1, a_1, r_1, s_2, a_2, r_2, \ldots\}$
    - 为每一步数据计算: $\delta_t = r_t+\gamma V_{\omega}(s_{t + 1})-V_{\omega}(s_t)$
    - 更新价值参数$w = w+\alpha_{\omega}\sum_t\delta_t\nabla_{\omega}V_{\omega}(s_t)$
    - 更新策略参数$\theta=\theta+\alpha_{\theta}\sum_t\delta_t\nabla_{\theta}\log\pi_{\theta}(a_t|s_t)$
 - end for 

## DQN及其改进

### DQN的基本改进

- 经验回放
  - 从buffer中采样一个batch的数据对DQN进行网络更新，buffer的大小通常在1w~10w之间，需要根据实际应用进行调参
  - 新采样的数据放入buffer中，并将最老的数据从buffer中删除
- 消除相关性
  - $s_t$和$s_{t+1}$通常有很强的相关性，这种相关性不利用DQN的学习。比如在超级玛丽游戏中，紧挨的两帧游戏页面的状态s是非常接近的。
- 优先经验回放
  - 不同的transition（一条训练数据）有不同的重要性，比如游戏中的第一关和boss关，重要性越高的transition，被抽到的概率也该越高
  - 可以TD error来衡量一个transition的重要性，TD error表示的是实际值和预测值的差异，TD error越大，说明模型在这个transition上的预测效果差，那么就赋予更高的采样概率
  - 由于采样概率不同，为了效果DQN的预测偏差，应该使用不同的学习率，高采样率的transition使用较低的学习率

### DQN高估的改进

- 高估如何产生
  - 回顾DQN中TD target的计算：$y_t = r_t + \gamma\cdot\max_{a}Q(s_{t + 1}, a; \mathbf{w})$，max操作使得TD target高估。假设$x_i$是真实值，$Q_i$是在$x_i$上加上了均值为0的噪声，那么$\mathbb{E}[\text{mean}_i(Q_i)] = \text{mean}_i(x_i)$，然而$\mathbb{E}[\max_i(Q_i)] \geq \max_i(x_i)$
  - DQN所使用的TD Learning算法是一种bootstrapping方法，根据DQN的梯度更新公式可知，TD target进一步用来更新$Q(s_t, a; \mathbf{w})$，最后使得$Q(s_{t+1}, a; \mathbf{w})$是高估的
- 高估的危害
  - 上述中的高估是非均匀的，均匀的高估是无害的。非均匀的高估使得错误的action被选择
- Target Network算法
  - 使用一个target network来计算TD target，target network的参数与DQN的参数不同，通常是来自于DQN，或者是target network和DQN参数的加权
- Double Network算法
  - 是对Target Network的改进，仍然使用target network计算TD target，但是计算TD target所使用的action是由DQN计算得到的最优action
  - 使用DQN选择最优action: $a^*=\underset{a}{\mathrm{argmax}}Q(s_{t + 1},a;w)$
  - 使用target network计算TD target: $y_t = r_t+\gamma\cdot Q(s_{t + 1},a^*;w^-)$
  -  $Q(s_{t + 1},a^*;w^-)\leq\underset{a}{\max}Q(s_{t + 1},a;w^-)$，其中$Q(s_{t + 1},a^*;w^-)$是Double network计算出来的，$\underset{a}{\max}Q(s_{t + 1},a;w^-)$是由Target network计算的，Double network计算的值更小，所以相比Target network缓解了高估问题

### Dueling network

- 几个定义
  - 最优动作价值函数：$Q^*(s, a)=\max_{\pi}Q_{\pi}(s, a)$
  - 最优状态价值函数：$V^*(s)=\max_{\pi}V_{\pi}(s)$
  - 最优优势函数：$A^*(s, a)=Q^*(s, a)-V^*(s)$
- 两个定理
  - 定理1：$V^*(s)=\max_{a}Q^*(s, a)$
    - 不难推理出: $max_{a} A^*(s, a)=\max_{a} Q^*(s, a)-V^*(s)=0$
  - 定理2：$Q^*(s, a)=V^*(s)+A^*(s, a)-\max_{a} A^*(s, a)$
    - 最后一项为0，这一项不能省略
    - 在Dueling network中，$V^*(s)$和$A^*(s, a)$都使用神经网络来近似，当两个神经网络波动幅度相同，当时方向相反，那么$Q^*(s, a)$将没有差别，因此无法学习。
- Dueling network
  - Dueling network主要由定理2给出
  - 使用神经网络$V(s; \mathbf{w}^V)$近似$V^*(s)$，使用神经网络$A(s, a; \mathbf{w}^A)$近似$A^*(s, a)$，两个神经网络共享同一个特征提取主干网络
  - 可以得到：$Q(s, a; \mathbf{w}^A, \mathbf{w}^V)=V(s; \mathbf{w}^V)+A(s, a; \mathbf{w}^A)-\max_{a}A(s, a; \mathbf{w}^A)$
  - Dueling network除了网络结构与DQN不同，其他都相同，因此可以使用前面提及的优化手段如优化经验回放、Double network等

## Policy gradient相关算法

### Baseline

回顾policy gradient的梯度公式：
$$
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} = \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot Q_{\pi}(s,A)\right]
$$
令b是独立于$A$的，那么：
$$
\begin{align*}
\mathbb{E}_{A\sim\pi}\left[b\cdot\frac{\partial\ln\pi(A\mid s;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right]&=b\cdot\mathbb{E}_{A\sim\pi}\left[\frac{\partial\ln\pi(A\mid s;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right]\\
&=b\cdot\sum_{a}\pi(a\mid s;\boldsymbol{\theta})\cdot\left[\frac{1}{\pi(a\mid s;\boldsymbol{\theta})}\cdot\frac{\partial\pi(a\mid s;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right]\\
&=b\cdot\sum_{a}\frac{\partial\pi(a\mid s;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\\
&=b\cdot\frac{\partial\sum_{a}\pi(a\mid s;\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\\
&=b\cdot\frac{\partial 1}{\partial\boldsymbol{\theta}} = 0.
\end{align*}
$$
那么，policy gradient的梯度公式可以写为：
$$
\begin{align*}
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} 
&= \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot Q_{\pi}(s,A)\right] \\
&= \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot Q_{\pi}(s,A)\right] - \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot b\right] \\
&= \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot (Q_{\pi}(s,A) - b)\right].
\end{align*}
$$
因此，添加baseline不会改变策略梯度，但是会减小方差，使得训练更加稳定

### REINFORCE with baseline

回顾policy gradient with baseline的梯度公式：
$$
\begin{align*}
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} 
&= \mathbb{E}_{A\sim\pi}\left[\frac{\partial \ln \pi(A\mid s;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\cdot (Q_{\pi}(s,A) - b)\right]
\end{align*}
$$
REINFORCE使用蒙特卡洛模拟计算$Q_\pi(s, A)$，即：$Q^{\pi}(s_t, a_t) \approx \sum_{t^{\prime}=t}^T\gamma^{t^{\prime}-t}r_{t^{\prime}}$

baseline的选择要求独立与$A$，而$V_\pi(s_t)$是与$A$无关的，并且$V_\pi(s_t)$是比较接近于$Q_\pi(s_t, a_t)$，因此可以使用$V_\pi(s_t)$作为baseline，实际使用时$V_\pi$使用价值网络$v(s; \boldsymbol{w})$来近似。

虽然带基线的 REINFORCE 有一个策略网络和一个价值网络，但是这种方法不是actor-critic。价值网络没有起到“评委”的作用，只是作为基线而已，目的在于降低方差，加速收敛。真正帮助策略网络（演员）改进参数 θ（演员的演技）的不是价值网络，而是实际观测到的回报$Q_\pi(s, A)$。  

### Advantage Actor-Critic（A2C）

1. A2C算法基本内容

   A2C是加入了baseline的actor-critic算法，在 A2C 中，通常使用时序差分（TD）误差来近似优势函数：

   $A(s,a)=r + \gamma V(s') - V(s)$

   价值网络的更新公式：

   $\omega \leftarrow \omega - \alpha_{v} \nabla_{\omega}L_{v}(\omega)$，其中$L_{v}(\omega)=\frac{1}{2}(r + \gamma V_{\omega}(s') - V_{\omega}(s))^2$

   策略网络的更新公式：

   $\theta \leftarrow \theta + \alpha_{a} \nabla_{\theta}\log\pi_{\theta}(a|s)A^{\pi}(s,a)$

2. 算法推导

   - 价值网络$v(s; w)$的算法从贝尔曼公式而来：$V_{\pi}(s_t)=\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\left[\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,A_t)}\left[R_t+\gamma\cdot V_{\pi}(S_{t + 1})\right]\right]$

     - 将$V_\pi(s_t)$使用神经网络$v(s; w)$来近似，TD target可以表示为$r_t+\gamma\cdot V_{\pi}(s_{t + 1})$，TD target包含真实观测到的奖励，相比$v(s_t; w)$更加可靠，因此价值网络的更新主要是让$v(s_t; w)$更加靠近TD target

   - 策略网络：根据贝尔曼公式：$Q_{\pi}(s_t,a_t)=\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}\left[R_t+\gamma\cdot V_{\pi}(S_{t + 1})\right]$，则带baseline的策略梯度公式：
     $$
     \begin{align*}
     \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) &= \left[Q_{\pi}(s,a)-V_{\pi}(s)\right]\cdot\nabla_{\boldsymbol{\theta}}\ln\pi(a|s;\boldsymbol{\theta}) \\ &=\left[\mathbb{E}_{S_{t + 1}}\left[R_t+\gamma\cdot V_{\pi}(S_{t + 1})\right]-V_{\pi}(s_t)\right]\cdot\nabla_{\boldsymbol{\theta}}\ln\pi(a_t|s_t;\boldsymbol{\theta})
     \end{align*}
     $$
     使用蒙特卡洛进行近似，进一步把状态价值函数$V_\pi(s)$替换成价值网络 $v(s; w)$，最后得到梯度：

     $\left[r_t+\gamma\cdot v(s_{t + 1};\boldsymbol{w})-v(s_t;\boldsymbol{w})\right]\cdot\nabla_{\boldsymbol{\theta}}\ln\pi(a_t|s_t;\boldsymbol{\theta})$

3. A2C和REINFORCE with baseline的区别

   - A2C的one step TD target的计算公式：$y_t = r_t + \gamma \cdot v(s_{t + 1} ; \mathbf{w})$
   - A2C的multi steps TD target的计算公式：$y_t = \sum_{i = 0}^{m - 1} \gamma^i \cdot r_{t + i} + \gamma^m \cdot v(s_{t + m} ; \mathbf{w})$
   - REINFORCE with baseline TD target的计算公式：$y_t = \sum_{i = t}^{n} \gamma^{i - t} \cdot r_i$

### Deterministic Policy Gradient（DPG）

算法流程：

- 使用策略网络选择一个动作: $a = \pi(s; \theta)$
- 更新策略网络: $\theta \leftarrow \theta+\beta\cdot\frac{\partial a}{\partial\theta}\cdot\frac{\partial q(s,a;w)}{\partial a}$
- 使用价值网络计算 $q_t = q(s,a;w)$
- 使用Target networks, $\pi(s; \theta^-)$ and $q(s,a; w^-)$, 来计算$q_{t + 1}$
- TD error: $\delta_t=q_t-(r_t + \gamma\cdot q_{t + 1})$ 
- 更新价值网络: $w \leftarrow w-\alpha\cdot\delta_t\cdot\frac{\partial q(s,a;w)}{\partial w}$



DPG是一个off-policy算法，收集transition的行为策略和优化的目标策略是不同的

- 训练策略网络
  - 给定一个状态s，目标策略输出一个动作a，然后价值网络根据状态s和动作a打一个分数$\widehat{q}=q(s, \boldsymbol{a} ; \boldsymbol{w})$。参数$\boldsymbol{\theta}$会影响动作a，从而影响$\widehat{q}$，训练策略网络的目标就是改进参数$\boldsymbol{\theta}$，是的$\widehat{q}$变大。
  - 确定策略网络用$\boldsymbol{u}(s; \boldsymbol{\theta})$表示，学习的目标可以被定义为$J(\boldsymbol{\theta}) = \mathbb{E}_S\left[q(S, \boldsymbol{\mu}(S ; \boldsymbol{\theta}) ; \boldsymbol{w})\right]$，关于状态 S 求期望消除掉了 S 的影响；不管面对什么样的状态 S，策略网络（演员）都应该做出很好的动作，使得平均分$J(\boldsymbol{\theta})$尽量高。  
  - 确定策略梯度：$\nabla_{\boldsymbol{\theta}}q(s_j, \boldsymbol{\mu}(s_j ; \boldsymbol{\theta}) ; \boldsymbol{w}) = \nabla_{\boldsymbol{\theta}}\boldsymbol{\mu}(s_j ; \boldsymbol{\theta}) \cdot \nabla_{\boldsymbol{a}}q(s_j, \widehat{\boldsymbol{a}}_j ; \boldsymbol{w})$ ，其中 $\widehat{\boldsymbol{a}}_j = \boldsymbol{\mu}(s_j ; \boldsymbol{\theta})$ 
- 训练价值网络
  - 每次从经验回放数组中取出一个四元组 $(s_j, \boldsymbol{a}_j, r_j, s_{j + 1})$，使用价值网络进行预测：$\widehat{q}_j = q(s_j, a_j; \boldsymbol{w})$ 和 $\widehat{q}_{j + 1} = q(s_{j + 1}, \boldsymbol{\mu}(s_{j + 1} ; \boldsymbol{\theta}) ; \boldsymbol{w})$
  - TD target是$\widehat{y}_j = r_j + \gamma \cdot \widehat{q}_{j + 1}$，那么损失函数$L(\boldsymbol{w})=\frac{1}{2}\left[\widehat{q}_j - \widehat{y}_j\right]^2$



## AlphaGo和AlphaGo Zero

- AlphaGo的策略网络架构

  - ![](./assets/alphago.png)

- 使用行为克隆的方法来初始化策略网络，即从人类数百万棋盘中进行策略网络的初始化学习，初始化后的策略网络能够超过业余选手的水平。该方法的局限性是：agent无法学习到奖励值；agent只能模仿专家行为，对于未见过的棋盘的泛化性效果不好

- 对经过初始化的策略网络进行强化学习训练

  - 构建两个策略网络，其中一个作为对手，从当前策略网络的先前迭代版本中进行获取
  - 策略网络的梯度更新参考策略梯度的计算公式

- 状态值网络的训练：用来评估当前局面下的胜率

  - 采样多个回合的数据，然后计算每个时间步的预期累计折扣奖励
  - 状态值网络使用神经网络模型，将状态s输入到神经网络中，计算模型预估的预期累计奖励
  - 使用MSE作为损失函数

- 推理时，使用蒙特卡洛搜索向前看，从当前节点出发进行搜索，在模拟过程中的每个状态下，计算棋面$S_t$下的最佳动作$a_t$

  - 计算每个动作的得分，$\mathrm{score}(a) = Q(a) + \eta \cdot \frac{\pi(a \mid s_t; \boldsymbol{\theta})}{1 + N(a)}$，其中$\pi(a \mid s_t; \boldsymbol{\theta})$是策略网络输出的动作概率值，$Q(a)$是通过MCTS计算的action value，$N_a$是在当前时刻动作a已经被选择的次数
  - 具体的，player做出一个action A，该action A并非实际执行的action，而是模拟思考的action；此时opponent也做出一个action
  - 使用训练好的状态值网络计算预期奖励V，持续执行下去，对每个新的状态计算预期奖励，将所有状态的预期奖励平均，作为$Q(a)$的值

- AlphaGo Zero

  - AlphaGo Zero相比AlphaGo效果更强
  - AlphaGo Zero未使用行为克隆
  - 使用了MCTS来训练策略网络
  - ![image-20250328190425737](./assets/alphago_zero.png)


## 多智能体RL

### 多智能体RL的基本概念

使用纳什均衡来衡量多智能体系统是否收敛：在保持其他agent的策略不变的情况下，任何一个agent通过改变策略无法获得更好的期望回报，此时这个多agent系统达到收敛状态

多智能体之间存在的关系：

- 合作关系：比如工业机器人，每个agent共同合作，才能完成整个任务
- 竞争关系：捕食者和被捕食者之间的关系，一个agent获取正收益的同时，另外一个agent一定获取负收益
- 合作竞争关系：足球比赛
- 利己主义：自动化股票交易系统

### 多智能体系统类型

| 系统类型                 | Policy（Actor）         | Value（Critic）  |                                                 |
| ------------------------ | ----------------------- | ---------------- | ----------------------------------------------- |
| 去中心化                 | $\pi(a^i|o^i;\theta^i)$ | $q(o^i,a^i;w^i)$ | 同单agent系统，效果不好                         |
| 中心化                   | $\pi(a^i|o; \theta^i)$  | $q(o,a;w^i)$     | 由中心系统计算每个agent的动作，比较慢           |
| 中心化训练，去中心化执行 | $\pi(a^i|o^i;\theta^i)$ | $q(o,a;w^i)$     | 训练时依赖中心系统，执行时每个agent自己计算动作 |

其中：

- i表示第i个agent，$a^i$表示第i个agent的动作，$o^i$表示第i个agent的部分观察（不同于全局观察s）
- $\theta^i$表示第i个agent的策略网络参数，$w^i$表示第i个agent的动作价值网络参数
- $o$和$a$表示全部agent的部分观测和全部agent的动作



## 一些公式的证明

### 状态-动作价值函数的贝尔曼方程

动作价值函数$Q^{\pi}(s, a)$可以通过状态价值函数$V^{\pi}(s)$来表示，即:

$Q^{\pi}(s,a)=R(s,a)+\gamma\sum_{s'\in S}P(s'|s,a)V^{\pi}(s')$。

其中，$R(s, a)$是在状态s下采取动作a所获得的即时奖励，$\gamma$是折扣因子，$P(s'|s, a)$是从状态s采取动作a后转移到状态$s'$的概率，S是状态空间。



该公式等同于下面的公式：
$Q_{\pi}(s_t, a_t) = \mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t, a_t)}[R_t + \gamma\cdot V_{\pi}(S_{t + 1})]$

两个本质相同，只是写法略有差异：等式右边，$R(s,a)$ 是当前采取动作 $a$ 得到的即时回报，$\gamma\sum_{s'\in S}P(s'|s,a)V^{\pi}(s')$ 是对所有可能下一状态 $s'$ 的价值（经折扣 $\gamma$ ），按照状态转移概率 $P(s'|s,a)$ 加权求和 ，综合起来表示从状态 $s$ 执行动作 $a$ 开始，未来累计折扣奖励的期望。 

### 贝尔曼方程：基于动作价值

$$
Q_{\pi}(s_t, a_t)=\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}\left[R_t+\gamma\cdot\mathbb{E}_{A_{t + 1}\sim\pi(\cdot|S_{t + 1};\boldsymbol{\theta})}[Q_{\pi}(S_{t + 1}, A_{t + 1})]\right]
$$

公式理解：

 - **即时奖励与后续价值**：等式右边，$R_t$ 是在状态 $s_t$ 采取动作 $a_t$ 后获得的即时奖励 。$\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}$ 表示基于状态转移概率 $p(\cdot|s_t,a_t)$ ，对采取动作 $a_t$ 后转移到的下一个状态 $S_{t + 1}$ 取期望 。
 - **递归关系**：下一个状态 $S_{t + 1}$ 的动作价值通过对基于策略 $\pi$ 选取的动作 $A_{t + 1}$ 求期望得到 。它体现了当前状态 - 动作对的价值，由即时奖励和后续状态 - 动作对价值的期望（经折扣因子 $\gamma$ 折扣后 ）组成，反映了动作价值函数的递归特性，是求解最优动作价值函数和最优策略的重要基础 。 

证明：

基于动作价值函数 $Q_{\pi}(s, a)$ 的定义及马尔可夫决策过程（MDP）的性质，从回报的定义出发，通过展开一步回报，利用期望的性质和马尔可夫性质逐步推导。

1. **明确相关定义**
    - 动作价值函数 $Q_{\pi}(s_t, a_t)$ 定义为从状态 $s_t$ 采取动作 $a_t$ 开始，遵循策略 $\pi$ 所获得的期望回报，即 $Q_{\pi}(s_t, a_t)=\mathbb{E}_{\pi}\left[\sum_{k = 0}^{\infty}\gamma^kR_{t + k + 1}|s_t, a_t\right]$ ，其中 $\gamma\in[0, 1]$ 为折扣因子，$R_{t + k + 1}$ 是 $t + k + 1$ 时刻获得的奖励 。
    - 回报 $G_t$ 定义为从时间步 $t$ 开始的累计折扣奖励，$G_t = R_t+\gamma R_{t + 1}+\gamma^2R_{t + 2}+\cdots=\sum_{k = 0}^{\infty}\gamma^kR_{t + k + 1}$ 。
2. **展开一步回报**
将回报 $G_t$ 展开一步，可得 $G_t = R_t+\gamma G_{t + 1}$ 。
对 $Q_{\pi}(s_t, a_t)$ 进行改写，$Q_{\pi}(s_t, a_t)=\mathbb{E}_{\pi}[G_t|s_t, a_t]$ ，把 $G_t = R_t+\gamma G_{t + 1}$ 代入可得：
$Q_{\pi}(s_t, a_t)=\mathbb{E}_{\pi}[R_t+\gamma G_{t + 1}|s_t, a_t]$ 
根据期望的线性性质 $\mathbb{E}[X + Y]=\mathbb{E}[X]+\mathbb{E}[Y]$ ，进一步得到：
$Q_{\pi}(s_t, a_t)=\mathbb{E}_{\pi}[R_t|s_t, a_t]+\gamma\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]$ 
3. **计算 $\mathbb{E}_{\pi}[R_t|s_t, a_t]$ 和 $\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]$**
    - 计算 $\mathbb{E}_{\pi}[R_t|s_t, a_t]$：
    已知奖励 $R_t$ 依赖于当前状态 $S_t$ 、采取的动作 $A_t$ 以及状态转移到下一个状态 $S_{t + 1}$ 的过程。根据条件期望的性质，先对下一个状态 $S_{t + 1}$ 基于状态转移概率 $p(\cdot|s_t,a_t)$ 取期望，可得 $\mathbb{E}_{\pi}[R_t|s_t, a_t]=\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}[R_t]$ 。
    - 计算 $\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]$ ：
    因为 $G_{t + 1}$ 是从 $t + 1$ 时刻开始的累计折扣奖励，而 $Q_{\pi}(S_{t + 1}, A_{t + 1})$ 是在状态 $S_{t + 1}$ 采取动作 $A_{t + 1}$ 后的动作价值函数。
    首先，在状态 $S_{t + 1}$ 要根据策略 $\pi(\cdot|S_{t + 1};\boldsymbol{\theta})$ 选择动作 $A_{t + 1}$ ，所以要先对动作 $A_{t + 1}$ 基于策略 $\pi$ 取期望，再结合状态转移等情况。即 $\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]=\mathbb{E}_{A_{t + 1}\sim\pi(\cdot|S_{t + 1};\boldsymbol{\theta})}[Q_{\pi}(S_{t + 1}, A_{t + 1})]$ ，这里取期望是考虑在状态 $S_{t + 1}$ 下按照策略 $\pi$ 选择不同动作 $A_{t + 1}$ 的所有可能情况。
4. **代入得到贝尔曼方程**
将 $\mathbb{E}_{\pi}[R_t|s_t, a_t]=\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}[R_t]$ 和 $\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]=\mathbb{E}_{A_{t + 1}\sim\pi(\cdot|S_{t + 1};\boldsymbol{\theta})}[Q_{\pi}(S_{t + 1}, A_{t + 1})]$ 代入 $Q_{\pi}(s_t, a_t)=\mathbb{E}_{\pi}[R_t|s_t, a_t]+\gamma\mathbb{E}_{\pi}[G_{t + 1}|s_t, a_t]$ ，就得到基于动作价值函数 $Q$ 的贝尔曼方程：
$Q_{\pi}(s_t, a_t)=\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,a_t)}\left[R_t+\gamma\cdot\mathbb{E}_{A_{t + 1}\sim\pi(\cdot|S_{t + 1};\boldsymbol{\theta})}[Q_{\pi}(S_{t + 1}, A_{t + 1})]\right]$ 。 

### 贝尔曼方程：基于状态价值

下面是基于状态价值的贝尔曼方程：
$$
V_{\pi}(s_t)=\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\left[\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,A_t)}\left[R_t+\gamma\cdot V_{\pi}(S_{t + 1})\right]\right]
$$


基于马尔可夫决策过程（MDP）基本概念推导贝尔曼公式

1. **马尔可夫性质**
马尔可夫决策过程满足马尔可夫性质，即系统未来的状态只取决于当前状态和当前采取的动作，与过去的状态和动作无关。用数学语言表示为：
$P(S_{t + 1}=s'|S_t = s, A_t = a, S_{t - 1}, A_{t - 1},\cdots,S_0,A_0)=P(S_{t + 1}=s'|S_t = s, A_t = a)$
这是后续推导的基础，意味着我们在计算未来状态价值时，不需要考虑历史信息，只关注当前状态和动作。

2. **状态价值函数定义**
状态价值函数 $V_{\pi}(s_t)$ 定义为从状态 $s_t$ 开始，遵循策略 $\pi$ 所获得的期望回报。回报 $G_t$ 是从时间步 $t$ 开始的累计折扣奖励，即 $G_t = R_t+\gamma R_{t + 1}+\gamma^2R_{t + 2}+\cdots$ ，其中 $\gamma\in[0,1]$ 是折扣因子。那么状态价值函数可表示为：
$V_{\pi}(s_t)=\mathbb{E}_{\pi}[G_t|S_t = s_t]=\mathbb{E}_{\pi}\left[\sum_{k = 0}^{\infty}\gamma^kR_{t + k + 1}|S_t = s_t\right]$

3. **展开一步**
我们将回报 $G_t$ 展开一步：
$G_t = R_t+\gamma R_{t + 1}+\gamma^2R_{t + 2}+\cdots=R_t+\gamma\left(R_{t + 1}+\gamma R_{t + 2}+\cdots\right)=R_t+\gamma G_{t + 1}$
对其两边取在策略 $\pi$ 下，以 $S_t = s_t$ 为条件的期望：
$V_{\pi}(s_t)=\mathbb{E}_{\pi}[G_t|S_t = s_t]=\mathbb{E}_{\pi}[R_t+\gamma G_{t + 1}|S_t = s_t]$
根据期望的线性性质 $\mathbb{E}[X + Y]=\mathbb{E}[X]+\mathbb{E}[Y]$ ，可得：
$V_{\pi}(s_t)=\mathbb{E}_{\pi}[R_t|S_t = s_t]+\gamma\mathbb{E}_{\pi}[G_{t + 1}|S_t = s_t]$

4. **计算 $\mathbb{E}_{\pi}[R_t|S_t = s_t]$ 和 $\mathbb{E}_{\pi}[G_{t + 1}|S_t = s_t]$**
    - 对于 $\mathbb{E}_{\pi}[R_t|S_t = s_t]$ ，因为奖励 $R_t$ 依赖于当前状态 $S_t$ 和采取的动作 $A_t$ 以及状态转移到下一个状态的过程，在给定策略 $\pi$ 下，先对动作 $A_t$ 基于策略 $\pi(\cdot|s_t;\boldsymbol{\theta})$ 取期望，再对下一个状态 $S_{t + 1}$ 基于状态转移概率 $p(\cdot|s_t,A_t)$ 取期望，即：
    $\mathbb{E}_{\pi}[R_t|S_t = s_t]=\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\left[\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,A_t)}[R_t]\right]$
    - 对于 $\mathbb{E}_{\pi}[G_{t + 1}|S_t = s_t]$ ，注意到 $G_{t + 1}$ 是从 $t + 1$ 时刻开始的累计折扣奖励，而 $V_{\pi}(S_{t + 1})$ 是 $t + 1$ 时刻状态 $S_{t + 1}$ 的价值函数，也就是从 $t + 1$ 时刻开始遵循策略 $\pi$ 的期望回报，所以 $\mathbb{E}_{\pi}[G_{t + 1}|S_t = s_t]=\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\left[\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,A_t)}[V_{\pi}(S_{t + 1})]\right]$

5. **代入得到贝尔曼公式**
  将上述计算结果代入 $V_{\pi}(s_t)=\mathbb{E}_{\pi}[R_t|S_t = s_t]+\gamma\mathbb{E}_{\pi}[G_{t + 1}|S_t = s_t]$ 中，就得到了贝尔曼公式：
  $$
   V_{\pi}(s_t)=\mathbb{E}_{A_t\sim\pi(\cdot|s_t;\boldsymbol{\theta})}\left[\mathbb{E}_{S_{t + 1}\sim p(\cdot|s_t,A_t)}\left[R_t+\gamma\cdot V_{\pi}(S_{t + 1})\right]\right] 
  $$
  

### $Q_{\pi}(s_t, a_t) = \mathbb{E}_{s_{t + 1}, a_{t + 1}}[R_t + \gamma \cdot Q_{\pi}(s_{t + 1}, a_{t + 1})]$

以下是对贝尔曼方程 $Q_{\pi}(s_t, a_t) = \mathbb{E}_{s_{t + 1}, a_{t + 1}}[R_t + \gamma \cdot Q_{\pi}(s_{t + 1}, a_{t + 1})]$ 的证明：

1. 定义与符号说明

- **策略**：$\pi(a|s)$ 表示在状态 $s$ 下采取动作 $a$ 的概率。
- **状态转移概率**：$p(s_{t + 1}|s_t, a_t)$ 表示在状态 $s_t$ 采取动作 $a_t$ 后转移到状态 $s_{t + 1}$ 的概率 。
- **即时奖励**：$R_t$ 是在时刻 $t$ 采取动作 $a_t$ 后从状态 $s_t$ 转移到 $s_{t + 1}$ 所获得的奖励。
- **折扣因子**：$\gamma \in [0, 1]$ ，用于权衡即时奖励和未来奖励 。
- **动作 - 价值函数**：$Q_{\pi}(s, a)$ 表示在策略 $\pi$ 下，从状态 $s$ 采取动作 $a$ 后，后续能获得的期望折扣累积奖励。

2. 期望折扣累积奖励的定义

期望折扣累积奖励 $G_t$ 从时刻 $t$ 开始定义为：
$G_t = R_t + \gamma R_{t + 1} + \gamma^2 R_{t + 2} + \cdots = \sum_{k = 0}^{\infty} \gamma^k R_{t + k}$
动作 - 价值函数 $Q_{\pi}(s_t, a_t)$ 是在策略 $\pi$ 下，从状态 $s_t$ 采取动作 $a_t$ 后 $G_t$ 的期望值，即：
$Q_{\pi}(s_t, a_t) = \mathbb{E}_{\pi}[G_t | s_t, a_t]$

3. 展开期望

将 $G_t$ 展开，根据期望的线性性质：
$$
\begin{align*}
Q_{\pi}(s_t, a_t) &= \mathbb{E}_{\pi}[R_t + \gamma R_{t + 1} + \gamma^2 R_{t + 2} + \cdots | s_t, a_t]\\
&= \mathbb{E}_{\pi}[R_t | s_t, a_t] + \gamma\mathbb{E}_{\pi}[R_{t + 1} + \gamma R_{t + 2} + \cdots | s_t, a_t]
\end{align*}
$$
由于 $R_t$ 只与 $s_t$ 和 $a_t$ 有关，所以 $\mathbb{E}_{\pi}[R_t | s_t, a_t] = R_t$ 。

对于 $\mathbb{E}_{\pi}[R_{t + 1} + \gamma R_{t + 2} + \cdots | s_t, a_t]$ ，在时刻 $t + 1$ ，系统处于状态 $s_{t + 1}$ ，此时从 $t + 1$ 时刻开始的期望折扣累积奖励就是 $Q_{\pi}(s_{t + 1}, a_{t + 1})$ 。

根据全概率公式，考虑从状态 $s_t$ 采取动作 $a_t$ 转移到不同状态 $s_{t + 1}$ 以及在 $s_{t + 1}$ 下采取不同动作 $a_{t + 1}$ 的情况：
$$
\begin{align*}
\mathbb{E}_{\pi}[R_{t + 1} + \gamma R_{t + 2} + \cdots | s_t, a_t]&=\sum_{s_{t + 1}}\sum_{a_{t + 1}}p(s_{t + 1}|s_t, a_t)\pi(a_{t + 1}|s_{t + 1})Q_{\pi}(s_{t + 1}, a_{t + 1})\\
&=\mathbb{E}_{s_{t + 1}, a_{t + 1}}[Q_{\pi}(s_{t + 1}, a_{t + 1})]
\end{align*}
$$

4. 得出贝尔曼方程

将上述结果代入 $Q_{\pi}(s_t, a_t)$ 的表达式中，可得：
$$
Q_{\pi}(s_t, a_t) = R_t + \gamma\mathbb{E}_{s_{t + 1}, a_{t + 1}}[Q_{\pi}(s_{t + 1}, a_{t + 1})]=\mathbb{E}_{s_{t + 1}, a_{t + 1}}[R_t + \gamma \cdot Q_{\pi}(s_{t + 1}, a_{t + 1})]
$$
这就完成了贝尔曼方程的证明，它描述了当前状态 - 动作对的价值与下一时刻状态 - 动作对价值之间的递归关系，是强化学习中求解最优策略的重要基础 。
