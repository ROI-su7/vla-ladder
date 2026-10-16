# 第 7 章 · L5 VLA：从零写一个 mini-π0

> 阶梯 L5：**写出一个结构上忠实于 openpi `PI0Pytorch` 的 VLA**，并在"看图 + 听指令 + 输出动作"的玩具桌面任务上训练、评估。这是整个项目的核心。

到这里，所有零件都已经在前面的章节写过了：

| 零件 | 来自 |
|---|---|
| ViT 视觉塔、patch embedding | 4.04 |
| Gemma 风格 Block（RMSNorm、RoPE、GeGLU） | 4.02 |
| 块状因果注意力掩码 `make_att_2d_masks` | 4.01 |
| KV cache | 4.03 |
| 流匹配（π0 约定）、Euler 采样 | 6.03 |
| 动作分块、滚动时域执行 | 3.02、5.02 |
| AdaRMSNorm（π0.5） | 1.08、6.02 的 FiLM |

这一章把它们组装起来，并解决一个新问题：**两套 Transformer（VLM 和动作专家）如何在每一层共享一次注意力**。

| 课 | 内容 |
|---|---|
| [7.01 VLM 是怎么"看图读字"的](7.01_vlm_basics.md) | PaliGemma 结构、图像预处理、提示词格式、`embed_prefix`、位置编码 |
| [7.02 π0 架构精读](7.02_pi0_architecture.md) | 动作专家、Mixture-of-Transformers 共享注意力、`embed_suffix`、π0 vs π0.5 |
| [7.03 训练 mini-π0](7.03_train_mini_pi0.md) | 两阶段训练（VLM 预训练 + VLA 微调）、评估、消融：预训练有多重要 |
| [7.04 读懂工业级代码库](7.04_codebase_reading.md) | openpi 数据流水线与部署、FluxVLA 配置系统、RTC 实时分块（动手实现） |

> 本章建议使用 GPU（Colab T4 即可）。7.03 的完整训练在 GPU 上约 3~4 分钟，CPU 上约 30~60 分钟。
> 完整参考实现：[`vla_ladder/models/mini_pi0.py`](../../vla_ladder/models/mini_pi0.py)、[`vla_ladder/pipelines.py`](../../vla_ladder/pipelines.py)。
