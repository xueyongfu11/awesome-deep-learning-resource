[TOC]



## Trust Region Policy Optimization (TRPO)  

### 前置知识

#### 置信域

- 置信域方法需要构造一个函数 $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$，这个函数要满足这个条件： $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$ 很接近 $J(\boldsymbol{\theta})$， $\forall \boldsymbol{\theta} \in \mathcal{N}(\boldsymbol{\theta}_{\text{now}})$ ，那么集合 $\mathcal{N}(\boldsymbol{\theta}_{\text{now}})$ 就被称作**置信域**。顾名思义，在 $\boldsymbol{\theta}_{\text{now}}$ 的邻域上，我们可以信任 $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$，可以拿 $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$ 来替代目标函数 $J(\boldsymbol{\theta})$。 
- 置信域方法一般是重复下面两个过程，知道让$J$无法继续增大。
  - **第一步——做近似**： 给定 $\boldsymbol{\theta}_{\text{now}}$，构造函数 $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$，使得 对于所有的 $\boldsymbol{\theta} \in \mathcal{N}(\boldsymbol{\theta}_{\text{now}})$，函数值 $L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$ 与 $J(\boldsymbol{\theta})$ 足够接近。
  - **第二步——最大化**：在置信 域 $\mathcal{N}(\boldsymbol{\theta}_{\text{now}})$ 中寻找变量 $\boldsymbol{\theta}$ 的值， 使得函数 $L$ 的值最大化。把找到的值记作 $\boldsymbol{\theta}_{\text{new}} = \underset{\boldsymbol{\theta} \in \mathcal{N}(\boldsymbol{\theta}_{\text{now}})}{\arg\max} L(\boldsymbol{\theta}|\boldsymbol{\theta}_{\text{now}})$
- 第二步的最大化是带有约束的最大化问题，求解这个问题需要单独的数值优化算法，可以是梯度投影算法、拉格朗日法。
- 置信域有很多选择，可以是球，也可以是KL散度

#### 重要性采样

一种用于在我们无法从目标分布 $p(x)$ 中采样时，只能从$q(x)$中进行采样，以此来近似计算期望值的方法。

我们利用概率分布的变换，将期望重新表示为：
$$
\int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx
$$
即：
$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]
$$
使用重要性采样的均值是相等的，我们进一步判定方差是否相同，并确定影响方差的因素：
$$
\text{Var}_{x \sim p}[f(x)] \quad \text{和} \quad \text{Var}_{x \sim q}\left[ f(x) \frac{p(x)}{q(x)} \right] \\
已知方差的基本公式： \text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$
用$p(x)$分布定义的方差：
$$
\text{Var}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim p}[f(x)^2] - \left( \mathbb{E}_{x \sim p}[f(x)] \right)^2
$$
用$q(x)$分布定义的方差：
$$
\text{Var}_{x \sim q}\left[ f(x) \frac{p(x)}{q(x)} \right]
= \mathbb{E}_{x \sim q} \left[ \left( f(x) \frac{p(x)}{q(x)} \right)^2 \right]
-\left( \mathbb{E}_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right] \right)^2 \\
= \mathbb{E}_{x \sim p} \left[ f(x)^2 \frac{p(x)}{q(x)} \right]
-\left( \mathbb{E}_{x \sim p} [f(x)] \right)^2
$$
比较两个方差，后项比前项多了个$\frac{p(x)}{q(x)}$，因此两个分布的差异直接影响了方差的差异。当$p(x)$和$q(x)$两个分布，接近时，定义的方差近似相等。

### TRPO算法

策略梯度算法存在问题时无法确定合适的步长，当步长不合适时，更新的参数所对应的策略是一个更不好的策略，当利用这个更不好的策略进行采样学习时，再次更新的参数会更差，因此很容易导致越学越差，最后崩溃。

TRPO找到了一块置信域，在该置信域中的参数更新可以得到安全保证，同时从理论上证明了TRPO策略学习的单调性。

#### 策略梯度目标

策略梯度的优化目标：对所有采样路径$\tau$的累计奖励求期望，使得该期望越大越好
$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]
$$
下面推导策略梯度的第二个等价目标：

已知：$V^{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$，其中$V^{\pi_\theta}(s_0)$表示起始状态$s_0$的状态价值。
$$
\begin{align*}
J(\theta) &= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] \\ 
&= \sum_{t=0}^\infty \mathbb{E}_{(s_t, a_t) \sim p_\theta(s_t, a_t)}[\gamma^t r(s_t, a_t)] \\
&= \mathbb{E}_{s_0 \sim p(s_0)} \left[ V^{\pi_\theta}(s_0) \right]
\end{align*}
$$
第二个等号根据 期望的累加等于累加的期望的性质，第三个等号根据$V^{\pi_\theta}(s_0)$推导出来

#### TRPO目标

假设当前策略为 $\pi_\theta$，参数为 $\theta$。我们考虑如何借助当前的 $\theta$ 找到一个更优的参数 $\theta'$，使得 $J(\theta') \geq J(\theta)$。具体来说，由于初始状态 $s_0$ 的分布和策略无关，因此上述策略 $\pi_\theta$ 下的优化目标 $J(\theta)$ 可以写成在新策略 $\pi_{\theta'}$ 的期望形式式：
$$
\begin{align*}
J(\theta) &= \mathbb{E}_{s_0} \left[ V^{\pi^\theta}(s_0) \right] \\
&= \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t V^{\pi^\theta}(s_t) - \sum_{t=1}^{\infty} \gamma^t V^{\pi^\theta}(s_t) \right] \\
&= - \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t \left( \gamma V^{\pi^\theta}(s_{t+1}) - V^{\pi^\theta}(s_t) \right) \right]
\end{align*}
$$
其中第三个等号，使用换元法，令t=t+1，容易推导出来

根据上述可知我们的目标是使得 $J(\theta') \geq J(\theta)$：
$$
\begin{align}
J(\theta') - J(\theta) 
&= \mathbb{E}_{s_0} \left[ V^{\pi^{\theta'}}(s_0) \right] - \mathbb{E}_{s_0} \left[ V^{\pi^{\theta}}(s_0) \right] \\
&= \mathbb{E}_{\pi^{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
+\mathbb{E}_{\pi^{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t \left( \gamma V^{\pi^{\theta}}(s_{t+1})-V^{\pi^{\theta}}(s_t) \right) \right] \\
&= \mathbb{E}_{\pi^{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \gamma V^{\pi^{\theta}}(s_{t+1}) -V^{\pi^{\theta}}(s_t) \right) \right]  \\
&= \mathbb{E}_{\pi^{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi^{\theta}}(s_t, a_t) \right] \\
&= \sum_{t=0}^{\infty} \gamma^t \, \mathbb{E}_{s_t \sim P_t^{\pi^{\theta'}}, \, a_t \sim \pi^{\theta'}(\cdot|s_t)} \left[ A^{\pi^{\theta}}(s_t, a_t) \right] \\
&= \frac{1}{1 - \gamma} \, \mathbb{E}_{s \sim \nu^{\pi^{\theta'}}, \, a \sim \pi^{\theta'}(\cdot|s)} \left[ A^{\pi^{\theta}}(s, a) \right]
\end{align}
$$
最后一个等号的成立运用到了状态访问分布的定义：$\nu^{\pi}(s)=(1 - \gamma)\sum_{t = 0}^{\infty}\gamma^{t}P_{t}^{\pi}(s)$，所以只要我们能找到一个新策略，使得 $\mathbb{E}_{s\sim\nu^{\pi_{\theta}}}\mathbb{E}_{a\sim\pi_{\theta}(\cdot|s)}[A^{\pi_{\theta}}(s, a)]\geq0$，就能保证策略性能单调递增，即 $J(\theta')\geq J(\theta)$。 

但是直接求解该式是非常困难的，因为$\pi_{\theta}$是我们需要求解的策略，但我们又要用它来收集样本。把所有可能的新策略都拿来收集数据，然后判断哪个策略满足上述条件的做法显然是不现实的。

首次近似：于是TRPO做了一步近似操作，对状态访问分布进行了相应处理。具体而言，忽略两个策略之间的状态访问分布变化，直接采用旧的策略$\pi_{\theta}$的状态分布，当新旧策略非常接近时，状态访问分布变化很小，这么近似是合理的。定义如下替代优化目标：
$$
L_{\theta}(\theta') - J(\theta) = \frac{1}{1 - \gamma}\mathbb{E}_{s\sim\nu^{\pi_{\theta}}}\mathbb{E}_{a\sim\pi_{\theta'}(\cdot|s)}[A^{\pi_{\theta}}(s,a)]
$$
二次近似：上述目标中仍然是使用新策略$\pi_\theta$采样动作，TRPO进一步使用重要性采样，使用旧策略的采样动作计算期望：
$$
L_{\theta}(\theta') = J(\theta)+\mathbb{E}_{s\sim\nu^{\pi_{\theta}}}\mathbb{E}_{a\sim\pi_{\theta}(\cdot|s)}\left[\frac{\pi_{\theta'}(a|s)}{\pi_{\theta}(a|s)}A^{\pi_{\theta}}(s,a)\right]
$$
最终我们的优化目标是：

- 使用KL散度约束梯度的更新

  - $$
    \theta' \leftarrow \underset{\theta'}{\arg\max}\sum_{t}\mathbb{E}_{s_t\sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t\sim\pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}\gamma^{t}A^{\pi_{\theta}}(s_t,a_t)\right]\right] \\
    such that  D_{KL}(\pi_{\theta'}(a_t|s_t) \parallel \pi_{\theta}(a_t|s_t))\leq\epsilon
    $$

- 使用可迭代的限制性惩罚项

  - $$
    \theta' \leftarrow \underset{\theta'}{\arg\max}\sum_{t}\mathbb{E}_{s_t\sim p_{\theta}(s_t)}\left[\mathbb{E}_{a_t\sim\pi_{\theta}(a_t|s_t)}\left[\frac{\pi_{\theta'}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}\gamma^{t}A^{\pi_{\theta}}(s_t,a_t)\right]\right]-\lambda\left(D_{KL}(\pi_{\theta'}(a_t|s_t) \parallel \pi_{\theta}(a_t|s_t)) - \epsilon\right) \\
    \lambda\leftarrow\lambda+\alpha\left(D_{KL}(\pi_{\theta'}(a_t|s_t) \parallel \pi_{\theta}(a_t|s_t)) - \epsilon\right)
    $$

    

第一种是带约束的最大化问题，无法使用梯度优化算法，可以使用数值优化算法进行计算。

## PPO

#### PPO介绍

- TRPO 算法在很多场景上的应用都很成功，但是我们也发现它的计算过程非常复杂，每一步更新的运算量非常大。PPO 基于 TRPO 的思想，但是其算法实现更加简单。并且大量的实验结果表明，与 TRPO 相比，PPO 能学习得一样好（甚至更快），这使得 PPO 成为非常流行的强化学习算法。

- PPO的一种形式是PPO-惩罚，使用拉格朗日乘数法直接将KL散度的限制直接放在了目标函数中，这就变成了无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。
  $$
  \underset{\theta }{\arg\max}\mathbb{E}_{s\sim\nu^{\pi_{\theta_{k}}}}\mathbb{E}_{a\sim\pi_{\theta_{k}}(\cdot|s)}\left[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)}A^{\pi_{\theta_{k}}}(s,a)-\beta D_{KL}[\pi_{\theta_{k}}(\cdot|s),\pi_{\theta}(\cdot|s)]\right]
  $$

  - 令 $d_{k}=D_{KL}^{\nu ^{\pi _{\theta _{k}}}}(\pi _{\theta _{k}},\pi _{\theta })$，$\beta$ 的更新规则如下：

    1. 如果 $d_{k}<\delta /1.5$，那么 $\beta _{k + 1}=\beta _{k}/2$
    2. 如果 $d_{k}>\delta\times1.5$，那么 $\beta _{k + 1}=\beta _{k}\times2$
    3. 否则 $\beta _{k + 1}=\beta _{k}$

    其中，$\delta$ 是事先设定的一个超参数，用于限制学习策略和之前一轮策略的差距。 

- PPO 的另一种形式 PPO-截断，这个方法更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大
  $$
  \underset{\theta}{\arg\max} \mathbb{E}_{s \sim \mathcal{D}^{\pi_{\theta_k}}} \mathbb{E}_{a \sim \pi_{\theta_k}(\cdot|s)} \left[ \min \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a), \text{clip} \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1 + \epsilon \right) A^{\pi_{\theta_k}}(s, a) \right) \right]
  $$

- PPO的两种形式中：

  - $$
    A^{\pi_\theta}(s_t, a_t) = r(s_t, a_t) + \gamma V^{\pi^{\theta}}(s_{t+1}) - V^{\pi^{\theta}}(s_t)
    $$

    

- 大量实验表明，PPO-截断总是比 PPO-惩罚表现得更好

#### PPO代码

PPO代码中advantage并未使用$A^{\pi_\theta}(s_t, a_t) = r(s_t, a_t) + \gamma V^{\pi^{\theta}}(s_{t+1}) - V^{\pi^{\theta}}(s_t)$，而是使用了GAE（Generalized Advantage Estimation）。

因为原始的方法只使用当前时刻来估计误差，会导致advantage噪声过大。GAE 是一种平衡 偏差-方差 的方法，它不是简单地用 TD delta，而是把多个 TD delta 叠加起来做加权平均：
$$
A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
$$
下面的代码中，函数`compute_advantage`来用计算GAE advantage。

```python

# 定义策略网络，输出是action的概率分布
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# 定义价值网络，输出是状态的价值
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        # 游戏是否结束
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        
        # 使用GAE计算advantage
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # KL
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,  1 + self.eps) * advantage  # 截断
            
            # 策略网络的损失
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            # 价值网络的损失
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
# 采样一批数据之后执行，agent执行一次update  
# 每次采样数据，都会用最新的agent执行采样
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)  # 基于采样的数据训练
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list
```



参考：

- [李宏毅强化学习](https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_&index=1)
- [code](https://github.com/boyu-ai/Hands-on-RL/blob/main/rl_utils.py#L81)

- [PPO算法介绍](https://hrl.boyuai.com/chapter/2/ppo%E7%AE%97%E6%B3%95)



[**点击查看我的更多AI学习笔记github**](https://github.com/xueyongfu11/awesome-deep-learning-resource)