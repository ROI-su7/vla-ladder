# 第 2 章 · 强化学习：从贝尔曼方程到 PPO / SAC

> 主要参考：赵世钰《强化学习的数学原理》([GitHub](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning))。
> 本章 2.01–2.10 与该书第 1–10 章一一对应，**使用相同的网格世界例子和记号**，可以对照阅读。
> 2.11 PPO 与 2.12 SAC 是书之外、但做 VLA+RL 必须掌握的两个算法（RLinf 的 PPO、RLT 的 actor-critic 都基于它们）。

## 本章路线

```text
       基础工具                        核心思想                       算法
  ┌──────────────┐
  │ 2.01 基本概念 │ ── 状态/动作/奖励/回报/策略/MDP
  └──────┬───────┘
  ┌──────▼───────┐       ┌───────────────────┐
  │2.02 贝尔曼方程│ ───▶ │2.03 贝尔曼最优方程 │ ──▶ 2.04 值迭代 / 策略迭代   （有模型）
  └──────────────┘       └───────────────────┘
                                   │ 去掉模型
                                   ▼
                    2.05 蒙特卡洛  ─▶  2.06 随机近似  ─▶ 2.07 TD / Sarsa / Q-learning   （无模型、表格）
                                                                │ 状态太多
                                                                ▼
                                                     2.08 值函数近似 / DQN
                                   ┌────────────────────────────┘ 直接优化策略
                                   ▼
                    2.09 策略梯度 ─▶ 2.10 Actor-Critic ─▶ 2.11 PPO（RLinf）
                                                     └─▶ 2.12 SAC（RLT 的 actor-critic）
```

| 课 | 内容 | 对应书中章节 |
|---|---|---|
| [2.01 基本概念](2.01_basic_concepts.md) | 状态、动作、策略、奖励、回报、MDP | 第 1 章 |
| [2.02 贝尔曼方程](2.02_bellman_equation.md) | 状态值、矩阵形式、闭式解与迭代解、动作值 | 第 2 章 |
| [2.03 贝尔曼最优方程](2.03_bellman_optimality.md) | 最优策略、压缩映射、值迭代 | 第 3 章 |
| [2.04 值迭代与策略迭代](2.04_value_policy_iteration.md) | VI、PI、截断策略迭代 | 第 4 章 |
| [2.05 蒙特卡洛方法](2.05_monte_carlo.md) | MC Basic、Exploring Starts、ε-greedy | 第 5 章 |
| [2.06 随机近似](2.06_stochastic_approximation.md) | 增量均值、Robbins-Monro、SGD | 第 6 章 |
| [2.07 时序差分学习](2.07_td_learning.md) | TD(0)、Sarsa、n 步、Q-learning | 第 7 章 |
| [2.08 值函数近似与 DQN](2.08_value_approximation_dqn.md) | 线性近似、DQN（经验回放 + 目标网络） | 第 8 章 |
| [2.09 策略梯度](2.09_policy_gradient.md) | 策略梯度定理、REINFORCE、基线 | 第 9 章 |
| [2.10 Actor-Critic](2.10_actor_critic.md) | QAC、A2C、GAE、重要性采样、DPG | 第 10 章 |
| [2.11 PPO](2.11_ppo.md) | 裁剪目标、GAE、连续动作 | — |
| [2.12 SAC](2.12_sac.md) | 最大熵 RL、双 Q、自动温度 | — |

## 学习建议

- 2.01–2.04 是**整个 RL 的地基**。一定要把"贝尔曼方程是线性方程组"这件事彻底弄懂。
- 每课的网格世界例子都可以画出来，**一定要看图**，看值和策略是怎么一步步变化的。
- 本章所有代码 CPU 即可运行；2.08 以后每个实验约 1–3 分钟。
