# 第 5 章 · L3 ACT：CVAE + Transformer + 动作分块

> 阶梯 L3：**从零实现 ACT**（Action Chunking with Transformers，Zhao et al. 2023，ALOHA 的原配算法），并通过实验搞懂它每一个设计的作用。

你用的 ALOHA / Cobot Magic 平台，最早就是和 ACT 一起发布的。ACT 的三个核心设计：

1. **动作分块**：一次预测 $k$ 步（原版 $k=100$），缓解复合误差（第 3 章）；
2. **CVAE**：用隐变量 $z$ 表示"人类示教的风格/模式"，处理多模态；
3. **时间集成**：每一步都重新预测，把多个重叠的动作块加权平均，让动作更平滑。

| 课 | 内容 |
|---|---|
| [5.01 VAE 与 CVAE](5.01_vae_cvae.md) | 变分推断、ELBO、重参数化、条件 VAE、后验坍缩 |
| [5.02 从零实现 ACT](5.02_act.md) | ACT 架构、训练、z=0 推理、时间集成、β 的影响 |

参考代码：[原版 ACT](https://github.com/tonyzhaozh/act)（`detr/models/detr_vae.py`）、[LeRobot ACT](https://github.com/huggingface/lerobot)（`policies/act/modeling_act.py`）。本章的完整实现在 [`vla_ladder/models/act.py`](../../vla_ladder/models/act.py)。
