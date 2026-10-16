# 第 6 章 · L4 生成式策略：扩散模型与流匹配

> 阶梯 L4：**从零实现 DDPM、Diffusion Policy 与流匹配策略**，最后一课完全按照 openpi 中 π0 的约定实现流匹配，直接为第 7 章的 mini-π0 做准备。

第 3 章的结论：动作分块 + 能表示多峰分布的生成模型 = 现代 VLA 的动作头。第 5 章的 CVAE 容易坍缩。扩散模型和流匹配是目前最可靠的选择：

| 方法 | 代表工作 | 思想 |
|---|---|---|
| 扩散（DDPM/DDIM） | Diffusion Policy（Chi et al. 2023）、Octo、RDT | 学习"去噪"：从纯噪声一步步去掉噪声，得到动作 |
| 流匹配（Flow Matching） | **π0 / π0.5**、GR00T N1、SmolVLA、FluxVLA 的 flow_matching_head | 学习一个"速度场"：沿着它从噪声流动到动作 |

| 课 | 内容 |
|---|---|
| [6.01 扩散模型（DDPM）](6.01_ddpm.md) | 前向加噪、噪声预测、反向采样，在二维数据上实现 |
| [6.02 Diffusion Policy](6.02_diffusion_policy.md) | 条件去噪网络、动作块扩散、双峰任务上的表现 |
| [6.03 流匹配：π0 的动作头](6.03_flow_matching.md) | 概率路径、速度场、Euler 积分、Beta 时间采样，完全对照 openpi |

本章完整实现：[`vla_ladder/models/flow.py`](../../vla_ladder/models/flow.py)。
