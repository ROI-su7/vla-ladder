# 第 4 章 · L2 Transformer：从注意力到 GPT 与 ViT

> 阶梯 L2：**从零写出一个能生成文字的 GPT、一个能看图的 ViT**，并彻底搞懂 VLA 里最关键的东西：注意力掩码。

π0 = 视觉编码器（ViT）+ 语言模型（GPT 结构的 Gemma）+ 动作专家（另一个小 Transformer）。三个部件全是 Transformer。读懂 openpi 的 `pi0_pytorch.py`，80% 的难点在于**注意力掩码怎么构造**、**两套权重怎么共享一次注意力**、**KV cache 怎么用**。这一章把这些全部从零实现一遍。

| 课 | 内容 | 对应 openpi / FluxVLA 中的代码 |
|---|---|---|
| [4.01 注意力机制与掩码](4.01_attention.md) | 缩放点积注意力、padding/因果/前缀掩码、交叉注意力 | `make_att_2d_masks` |
| [4.02 Transformer Block](4.02_transformer_block.md) | 多头注意力、位置编码（正弦/RoPE）、FFN/GeGLU、Pre-Norm | `gemma_pytorch.py`、`apply_rotary_pos_emb` |
| [4.03 从零写 GPT](4.03_gpt_from_scratch.md) | 字符级语言模型、训练、采样、KV cache | PaliGemma 的语言模型部分 |
| [4.04 Vision Transformer](4.04_vit.md) | patch embedding、位置编码、"看图找方块" | SigLIP 视觉塔 |

> 本章 4.03、4.04 有 GPU 会快很多（Colab 选 T4 即可），CPU 上每个实验约 2~5 分钟。
