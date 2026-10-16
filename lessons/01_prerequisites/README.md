# 第 1 章 · 预备知识：数学、PyTorch 与深度学习基础

> 参考资料：[《动手学深度学习》(d2l.ai)](https://zh.d2l.ai/)、[菜鸟教程 PyTorch](https://www.runoob.com/pytorch/pytorch-tutorial.html)。
> 本章按"写 VLA 真正需要什么"重新组织了这些内容，每个知识点都会说明它在 VLA 里用在哪。

## 本章路线

| 课 | 内容 | 在 VLA 里的用处 |
|---|---|---|
| [1.01 Python 与 NumPy](1.01_python_numpy.md) | 类、广播、索引、向量化 | 所有数据处理代码 |
| [1.02 PyTorch 张量](1.02_pytorch_tensor.md) | 创建、变形、索引、`einsum`、GPU | 每一行模型代码 |
| [1.03 线性代数](1.03_linear_algebra.md) | 向量、矩阵乘法、范数、投影、特征值、SVD | 注意力 = 矩阵乘法；LoRA = 低秩分解 |
| [1.04 微积分与自动求导](1.04_calculus_autograd.md) | 导数、梯度、链式法则、手写反向传播 | 训练的本质 |
| [1.05 概率与统计](1.05_probability.md) | 分布、期望、最大似然、交叉熵、KL、重参数化 | 损失函数从哪来；扩散、VAE、RL 的数学基础 |
| [1.06 线性回归与梯度下降](1.06_linear_regression.md) | 从零实现训练循环 | 所有模型训练的骨架 |
| [1.07 Softmax 回归与分类](1.07_softmax_classification.md) | logits、交叉熵、数值稳定 | 语言模型输出、离散动作 |
| [1.08 多层感知机与训练技巧](1.08_mlp_and_training.md) | 激活、初始化、正则化、归一化、优化器 | RLT 的 actor/critic 就是 MLP |
| [1.09 卷积神经网络](1.09_cnn.md) | 卷积、池化、ResNet、空间 softmax | ACT / Diffusion Policy 的视觉编码器 |
| [1.10 PyTorch 工程实践](1.10_training_pipeline.md) | Dataset/DataLoader、保存加载、调试 | 读懂 openpi / FluxVLA 训练脚本 |

## 学习建议

- 有基础的同学可以快速浏览原理，**只做练习**。练习都能在 5 分钟内写出来，就可以跳到下一课。
- 数学部分不需要一次全懂。后面遇到时（比如第 6 章扩散模型用到高斯分布）再回来复习，效果最好。
- 本章全部可以在 CPU 上运行。
