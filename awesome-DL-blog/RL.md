<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [概念、基础](#%E6%A6%82%E5%BF%B5%E5%9F%BA%E7%A1%80)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## 概念、基础

- [基于模型(Model Based)和model free的强化学习](https://itpcb.com/a/162657)
- [强化学习基础](http://itpcb.com/docs/MachineLearningNotes/17%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.html)
  - model-based: 直接对环境进行建模，在状态空间和动作空间已知的情况下，对状态转移
  概率和奖励进行建模，一般可以使用监督学习的方式，如何给定状态，动作所对应的新的状态，
  和奖励，使用有监督学习中的分类和回归进行建模。  
  模型已知之后，就可以根据计算值函数或者
  策略函数来对任意策略进行评估。当模型已知时，策略的评估问题转化为一种动态规划问题。
  - model-free: 由于模型参数未知，状态值函数不能像之前那样进行全概率展开，从而
  运用动态规划法求解。一种直接的方法便是通过采样来对策略进行评估/估算其值函数
  
- [On-policy与off-policy](https://zhuanlan.zhihu.com/p/346433931)
  - on policy:行为策略和目标策略相同
  - off policy:把数据收集当作一个单独的任务，行为策略和目标策略是不同的。
  比如在Q-learning的网络更新公式中，(S, A, R, S', A'), A使用探索贪心策略,而
  A’则使用贪心策略。SARSA中，A和A‘都是使用探索贪心策略
  