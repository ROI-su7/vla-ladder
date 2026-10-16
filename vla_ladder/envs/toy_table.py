"""玩具桌面（ToyTable）：给 mini-VLA（第 7、8 章）用的“视觉 + 语言 + 动作”任务。

场景：桌面上随机摆放 3 个不同颜色的方块，还有一个“夹爪”（空心圆环）。
指令：随机一句中文，例如“去红色方块”“把夹爪移到蓝色方块上”。
任务：把夹爪移动到指令所说颜色的方块中心。

为什么它是一个合格的 VLA 任务？
  * 方块位置只能从图像中看到        -> 必须用视觉（Vision）
  * 去哪个方块只能从指令中知道      -> 必须用语言（Language）
  * 输出连续的二维速度              -> 必须输出动作（Action）
  * 可以把成功半径调小来模拟“线头对插”这类精细任务（第 8 章用 RL 提升精度）

观测（全部是批量的，第 0 维为环境编号）：
    obs["image"]       float32 (N, 3, H, W)，取值 [0, 1]
    obs["state"]       float32 (N, 2)，夹爪位置（本体感知 proprioception）
    obs["instruction"] list[str]，长度 N
动作：float32 (N, 2) ∈ [-1, 1]，位置更新 p += step_size * a
"""

from __future__ import annotations

import numpy as np

COLOR_NAMES = ["红", "绿", "蓝", "黄"]
COLOR_EN = ["red", "green", "blue", "yellow"]  # 画图标题用英文（Colab 默认没有中文字体）
COLOR_RGB = np.array(
    [[0.90, 0.15, 0.15], [0.15, 0.70, 0.20], [0.20, 0.35, 0.95], [0.95, 0.80, 0.10]], dtype=np.float32
)
TEMPLATES = ["去{c}色方块", "移动到{c}色的方块", "把夹爪移到{c}色方块上", "请靠近{c}色方块", "找到{c}色方块"]
TABLE_RGB = np.array([0.85, 0.80, 0.70], dtype=np.float32)
GRIPPER_RGB = np.array([0.10, 0.10, 0.10], dtype=np.float32)


class CharTokenizer:
    """最简单的分词器：一个汉字一个 token。

    真实 VLA（如 PaliGemma）用的是 SentencePiece 子词分词器，词表有 25 万个；
    原理完全一样：文本 -> 整数 id 序列 -> 查 embedding 表。
    """

    PAD, BOS = 0, 1

    def __init__(self):
        chars = sorted({ch for t in TEMPLATES for ch in t.replace("{c}", "")} | set(COLOR_NAMES))
        self.itos = ["<pad>", "<bos>"] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, texts, max_len: int = 16):
        """返回 (ids, mask)：ids int64 (N, max_len)，mask bool (N, max_len)，True 表示真实 token。"""
        if isinstance(texts, str):
            texts = [texts]
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        mask = np.zeros((len(texts), max_len), dtype=bool)
        for i, t in enumerate(texts):
            toks = [self.BOS] + [self.stoi[ch] for ch in t][: max_len - 1]
            ids[i, : len(toks)] = toks
            mask[i, : len(toks)] = True
        return ids, mask

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > self.BOS)


class ToyTableEnv:
    def __init__(
        self,
        num_envs: int = 1,
        image_size: int = 48,
        n_blocks: int = 3,
        block_half: float = 0.12,
        step_size: float = 0.05,
        max_steps: int = 50,
        success_radius: float = 0.05,
        expert_offset: float = 0.0,
        seed=None,
    ):
        self.num_envs, self.image_size, self.n_blocks = num_envs, image_size, n_blocks
        # expert_offset > 0：模拟"有系统性偏差的示教者"（例如插线总是插不到底）：
        # 专家会停在离方块中心 expert_offset 远的地方（沿着它出发时的来向）
        self.expert_offset = expert_offset
        self.detour = np.zeros(num_envs, dtype=bool)  # 为 True 的回合：专家先走向一个错误的方块，再改去正确的（RECAP 的"低质量示教"）
        self.block_half, self.step_size, self.max_steps = block_half, step_size, max_steps
        self.success_radius = success_radius
        self.rng = np.random.default_rng(seed)
        # 像素中心坐标：列 -> x ∈ [-1, 1]，行 -> y（第 0 行在最上方，对应 y = +1）
        c = (np.arange(image_size) + 0.5) / image_size * 2 - 1
        self.px = c[None, :].repeat(image_size, 0)  # (H, W) 每个像素的 x
        self.py = -c[:, None].repeat(image_size, 1)  # (H, W) 每个像素的 y
        self.soft = 1.0 / image_size  # 边缘柔化宽度（约半个像素），让亚像素位置也能体现在图像里

    # ------------------------------------------------------------------ 回合控制
    def reset(self):
        n, nb = self.num_envs, self.n_blocks
        self.blocks = np.zeros((n, nb, 2))
        for i in range(n):  # 拒绝采样，保证方块之间不重叠
            pts = []
            while len(pts) < nb:
                p = self.rng.uniform(-0.75, 0.75, size=2)
                if all(np.linalg.norm(p - q) > 0.4 for q in pts):
                    pts.append(p)
            self.blocks[i] = np.array(pts)
        self.colors = np.stack([self.rng.permutation(len(COLOR_NAMES))[:nb] for _ in range(n)])  # (n, nb)
        self.target_idx = self.rng.integers(0, nb, size=n)
        self.instructions = [
            self.rng.choice(TEMPLATES).format(c=COLOR_NAMES[self.colors[i, self.target_idx[i]]]) for i in range(n)
        ]
        self.pos = self.rng.uniform(-0.85, 0.85, size=(n, 2))
        self.start_pos = self.pos.copy()
        self.wrong_idx = (self.target_idx + 1 + self.rng.integers(0, self.n_blocks - 1, size=n)) % self.n_blocks
        self.detour_steps = self.rng.integers(8, 16, size=n)
        self.t = 0
        self.done = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        return self.obs()

    @property
    def target(self):
        return self.blocks[np.arange(self.num_envs), self.target_idx]

    def obs(self):
        return {"image": self.render(), "state": self.pos.astype(np.float32).copy(), "instruction": list(self.instructions)}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(self.num_envs, 2), -1, 1)
        active = ~self.done
        self.pos[active] = np.clip(self.pos[active] + self.step_size * action[active], -1, 1)
        self.t += 1
        dist = np.linalg.norm(self.pos - self.target, axis=1)
        reached = (dist < self.success_radius) & active
        self.success |= reached
        reward = reached.astype(np.float32)
        self.done |= reached | (self.t >= self.max_steps)
        return self.obs(), reward, self.done.copy(), {"success": self.success.copy(), "dist": dist}

    def expert_action(self, noise: float = 0.0):
        """脚本专家：朝目标方块中心直线前进（速度上限为 1）。noise 模拟遥操作的手抖；expert_offset 见 __init__。"""
        goal = self.target
        if self.detour.any():
            wrong = self.blocks[np.arange(self.num_envs), self.wrong_idx]
            goal = np.where((self.detour & (self.t < self.detour_steps))[:, None], wrong, goal)
        if self.expert_offset > 0:
            d = self.target - self.start_pos
            goal = self.target - self.expert_offset * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        a = (goal - self.pos) / self.step_size
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0)
        if noise > 0:
            a = a + noise * self.rng.standard_normal(a.shape)
        return np.clip(a, -1, 1).astype(np.float32)

    # ------------------------------------------------------------------ 渲染
    def render(self):
        n, H = self.num_envs, self.image_size
        img = np.broadcast_to(TABLE_RGB[None, :, None, None], (n, 3, H, H)).copy()
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        for b in range(self.n_blocks):
            cx = self.blocks[:, b, 0][:, None, None]
            cy = self.blocks[:, b, 1][:, None, None]
            m = sig((self.block_half - np.abs(self.px - cx)) / self.soft) * sig((self.block_half - np.abs(self.py - cy)) / self.soft)
            rgb = COLOR_RGB[self.colors[:, b]][:, :, None, None]  # (n, 3, 1, 1)
            img = img * (1 - m[:, None]) + rgb * m[:, None]
        # 夹爪：空心圆环（不会完全挡住下面的方块）
        d = np.sqrt((self.px - self.pos[:, 0][:, None, None]) ** 2 + (self.py - self.pos[:, 1][:, None, None]) ** 2)
        ring = sig((0.025 - np.abs(d - 0.07)) / self.soft)
        img = img * (1 - ring[:, None]) + GRIPPER_RGB[None, :, None, None] * ring[:, None]
        return img.astype(np.float32)

    def show(self, i: int = 0, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(self.render()[i].transpose(1, 2, 0), extent=(-1, 1, -1, 1))
        ax.set_title(f"goal: {COLOR_EN[self.colors[i, self.target_idx[i]]]} block")
        ax.axis("off")
        return ax


def collect_table_demos(env: ToyTableEnv, n_episodes: int, noise: float = 0.0, seed: int = 0):
    """用脚本专家采集示教数据，返回按帧展开的数据字典（图像存 uint8 节省内存）。

    返回字段（F 为总帧数）：
        image (F, 3, H, W) uint8 | state (F, 2) | action (F, 2) | instruction list[str] | episode (F,) | t (F,)
    """
    env.rng = np.random.default_rng(seed)
    data = {"image": [], "state": [], "action": [], "instruction": [], "episode": [], "t": [], "detour": []}
    ep_offset = 0
    n_success = 0
    while ep_offset < n_episodes:
        obs = env.reset()
        alive_hist = []
        frames = []
        while not env.done.all():
            act = env.expert_action(noise=noise)
            alive = ~env.done
            frames.append((obs["image"], obs["state"], act, alive, env.t))
            obs, _, _, _ = env.step(act)
            alive_hist.append(alive)
        for i in range(env.num_envs):
            if ep_offset >= n_episodes:
                break
            for img, st, act, alive, t in frames:
                if alive[i]:
                    data["image"].append((img[i] * 255).round().astype(np.uint8))
                    data["state"].append(st[i])
                    data["action"].append(act[i])
                    data["instruction"].append(env.instructions[i])
                    data["episode"].append(ep_offset)
                    data["t"].append(t)
                    data["detour"].append(bool(env.detour[i]))
            n_success += int(env.success[i])
            ep_offset += 1
    out = {k: (np.stack(v) if k != "instruction" else v) for k, v in data.items()}
    out["success_rate"] = n_success / n_episodes
    return out
