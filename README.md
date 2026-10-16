# VLA 阶梯 · 从零到自己写一个 VLA

**一个面向小白的 VLA（视觉-语言-动作模型）+ 强化学习学习项目。** 从 NumPy 和贝尔曼方程开始，一级一级往上爬，最终从零写出一个结构上与 π0 一致的 mini-VLA，并用 RLT、RECAP 让它超越示教。

- 📖 **在线阅读**：<https://roi-su7.github.io/vla-ladder/>
- 🧪 **在线练习**：每一课标题下都有 ![Colab](https://colab.research.google.com/assets/colab-badge.svg) 按钮，在浏览器里直接写代码、跑代码，不需要装任何环境
- 💡 **不会写？** 每个练习下面都有折叠的参考答案，点一下就能看完整代码；再下面是自测单元，运行后显示"通过 ✅"就说明写对了

> 每一课的原理讲解都配有可运行的代码，练习的参考实现尽量对照 [openpi](https://github.com/Physical-Intelligence/openpi)、[FluxVLA](https://github.com/FluxVLA/FluxVLA)、[RLinf](https://github.com/RLinf/RLinf)、[LeRobot](https://github.com/huggingface/lerobot)、[ACT](https://github.com/tonyzhaozh/act)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)、[nanoGPT](https://github.com/karpathy/nanoGPT) 等项目的真实实现来写。**全部代码都经过完整运行测试**（`python tools/run_lessons.py`）。

---

## 学习路线

```text
 L0 导读        如何使用本项目 · 锻炼代码逻辑的四个习惯
  │
 第1章 预备知识  Python/NumPy · PyTorch · 线性代数 · 微积分与自动求导 · 概率统计 · 线性回归 · 分类 · MLP · CNN · 工程实践
  │
 第2章 强化学习  基本概念 → 贝尔曼方程 → 最优方程 → 值/策略迭代 → MC → 随机近似 → TD/Q-learning → DQN → 策略梯度 → Actor-Critic → PPO → SAC
  │
 L1 第3章 模仿学习      行为克隆 · 分布偏移与 DAgger · 多模态与动作分块
 L2 第4章 Transformer   注意力与掩码 · Transformer Block · 从零写 GPT（KV cache）· ViT
 L3 第5章 ACT           VAE/CVAE · ACT 完整实现 · 时间集成
 L4 第6章 生成式策略     DDPM · Diffusion Policy · 流匹配（π0 的动作头）
 L5 第7章 VLA           VLM 基础 · π0 架构精读 · 训练 mini-π0 · 读懂 openpi/FluxVLA · RTC 实时分块
 L6 第8章 VLA + RL      全景 · RLT（RL Token）· RECAP（优势条件化）· 真机落地清单
```

| 章 | 课程 | 你会写出的东西 |
|---|---|---|
| [第 0 章 导读](lessons/00_guide/README.md) | 0.1 ~ 0.2 | 学习方法、形状调试、动作分块函数 |
| [第 1 章 预备知识](lessons/01_prerequisites/README.md) | 1.01 ~ 1.10 | 迷你自动求导引擎、Adam、LayerNorm/RMSNorm、空间 softmax、训练模板 |
| [第 2 章 强化学习](lessons/02_rl/README.md) | 2.01 ~ 2.12 | 网格世界上的全部表格算法、DQN、REINFORCE、A2C+GAE、PPO、SAC |
| [第 3 章 模仿学习（L1）](lessons/03_imitation/README.md) | 3.01 ~ 3.02 | BC、HG-DAgger（人工接管）、混合密度网络 |
| [第 4 章 Transformer（L2）](lessons/04_transformer/README.md) | 4.01 ~ 4.04 | openpi 的注意力掩码、RoPE、GeGLU、GPT + KV cache、ViT |
| [第 5 章 ACT（L3）](lessons/05_act/README.md) | 5.01 ~ 5.02 | CVAE、ACT、时间集成 |
| [第 6 章 生成式策略（L4）](lessons/06_generative_policy/README.md) | 6.01 ~ 6.03 | DDPM、Diffusion Policy、π0 约定的流匹配 |
| [第 7 章 VLA（L5）](lessons/07_vla/README.md) | 7.01 ~ 7.04 | **mini-π0**（VLM + 动作专家共享注意力 + KV cache + 流匹配）、引导式 RTC |
| [第 8 章 VLA + RL（L6）](lessons/08_vla_rl/README.md) | 8.01 ~ 8.04 | 流 SDE 的 log π、**RLT**、**RECAP**、评测统计工具 |

**建议节奏**（每天 1~2 小时）：第 0~1 章 2 周 · 第 2 章 3 周 · 第 3~6 章 3 周 · 第 7~8 章 3 周。

---

## 怎么用

**方式一：Colab（推荐）**：点任意一课的 Colab 按钮 → 运行最上面的"环境准备"单元 → 从上往下做。第 4 章以后建议在 Colab 里选择 GPU 运行时。

**方式二：本地**

```bash
git clone https://github.com/ROI-su7/vla-ladder.git
cd vla-ladder
pip install -e .              # 只依赖 numpy / torch / matplotlib
pip install jupyterlab
jupyter lab notebooks/
```

**方式三：GitHub Codespaces**：仓库页面 → Code → Codespaces → Create codespace。

### 做练习的"三遍法"

1. 先独立写，卡住超过 20 分钟再看答案；
2. 看答案时只看思路，关掉答案自己写出来；
3. 第二天在空白文件里从零再写一遍。

详见 [0.1 如何使用本项目](lessons/00_guide/0.1_how_to_use.md) 与 [0.2 锻炼代码逻辑思维](lessons/00_guide/0.2_code_thinking.md)。

---

## 项目结构

```text
vla-ladder/
├── lessons/        📖 课程源文件（原理 + 代码 + 练习 + 折叠答案），GitHub 上可直接阅读
├── notebooks/      🧪 由 lessons 自动生成的练习本（Colab / Jupyter）
├── vla_ladder/     📦 配套代码包（每个阶段"写完之后应该长什么样"的完整参考实现）
│   ├── envs/       网格世界、CartPole/Pendulum、二维点到达、双峰避障、玩具桌面（视觉+语言+动作）
│   ├── rl/         表格方法、DQN、PPO、SAC
│   ├── models/     transformer（GPT/ViT）、act、flow（DDPM/流匹配）、mini_pi0、rlt
│   └── pipelines.py  mini-π0 的两阶段训练与评估流水线
└── tools/
    ├── build.py        lessons -> notebooks + 网站
    └── run_lessons.py  把所有课程当脚本完整跑一遍（练习用参考答案替换），保证代码可运行
```

## 一些实验结果（全部可以在课程里复现）

| 课 | 实验 | 结果 |
|---|---|---|
| 3.01 | 普通 BC vs 人工接管式 DAgger（3cm 精度到达任务） | BC：255 帧 0.31、717 帧 0.94；HG-DAgger：585 个标注 0.89、714 个标注 1.00 |
| 3.02 | 双峰任务：MSE 动作块 vs 混合密度网络动作块 | 0.0 → 0.9 |
| 5.02 | ACT 默认 β=10 | KL → 0（后验坍缩），时间集成的闭环重规划是它仍能工作的关键 |
| 6.03 | 流匹配 Euler 步数 1 / 3 / 10 | 成功率 ≈ 0.01 / 0.92 / 0.99（1 步 = 取平均） |
| 7.03 | mini-π0：有 / 无 VLM 预训练（同样 2000 条示教） | 0.99 vs 0.27 |
| 7.04 | 双峰起点处的 RTC：独立采样 / 替换式 / 引导式（d=3） | 新块延续旧模式的比例 0.38 / 0.56 / 0.98 |
| 8.02 | RLT：冻结 VLA，修正"插不到底"的示教偏差 | 0.31 → 0.87 |
| 8.03 | RECAP：混合质量示教 | BC < RECAP(+) ≈ 只用好示教的上限 |

---

## 参考资料

- 《动手学深度学习》：<https://zh.d2l.ai/>
- 菜鸟教程 PyTorch：<https://www.runoob.com/pytorch/pytorch-tutorial.html>
- 赵世钰《强化学习的数学原理》：<https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning>
- openpi：<https://github.com/Physical-Intelligence/openpi> · FluxVLA：<https://github.com/FluxVLA/FluxVLA> · RLinf：<https://github.com/RLinf/RLinf>

## 本地构建与测试

```bash
python tools/build.py           # 生成 notebooks/ 与 build/docs/（网站源文件），刷新 mkdocs 导航
python tools/run_lessons.py     # 完整运行所有课程（GPU 约 40 分钟）
pip install mkdocs-material && mkdocs serve   # 本地预览网站
```

推送到 `main` 分支后，GitHub Actions 会自动构建并发布网站（`.github/workflows/pages.yml`）。

## License

MIT
