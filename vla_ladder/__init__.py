"""VLA 阶梯（vla-ladder）配套代码包。

- vla_ladder.envs   ：课程中用到的所有小环境（网格世界、CartPole、二维点到达、玩具桌面 VLA 任务）
- vla_ladder.models ：各阶段模型的“完整参考实现”（MLP、GPT、ACT、扩散/流匹配策略、mini-π0、RLT）
- vla_ladder.rl     ：强化学习算法的参考实现（DQN、PPO、SAC 等）
- vla_ladder.utils  ：随机种子、设备选择、画图等小工具

课程（lessons/）里会一步步带你写出这些代码；这里的文件是“写完之后应该长什么样”的标准答案。
"""

__version__ = "0.1.0"
