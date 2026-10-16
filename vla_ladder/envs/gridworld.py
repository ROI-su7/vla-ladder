"""网格世界（Grid World）——《强化学习的数学原理》（赵世钰）全书使用的例子。

坐标约定：状态用 (行, 列) 表示，从 0 开始；状态编号 s = 行 * 列数 + 列。
动作约定（与书中 a1..a5 对应）：
    0: 上 (a1)   1: 右 (a2)   2: 下 (a3)   3: 左 (a4)   4: 原地不动 (a5)
奖励约定（与书中一致）：
    撞墙（试图走出边界）：r_boundary = -1，并且留在原地
    进入禁区：r_forbidden = -1（可以进入，但会被惩罚）
    到达/停留在目标：r_target = +1
    其它：0
这是一个“确定性”环境：p(s'|s,a) 只有一个 s' 的概率为 1。
"""

from __future__ import annotations

import numpy as np

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
ACTION_NAMES = ["↑", "→", "↓", "←", "○"]


class GridWorld:
    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        target=(3, 2),
        forbidden=((1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)),
        r_boundary: float = -1.0,
        r_forbidden: float = -1.0,
        r_target: float = 1.0,
        r_other: float = 0.0,
        start=(0, 0),
    ):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.target = tuple(target)
        self.forbidden = {tuple(f) for f in forbidden}
        self.r_boundary, self.r_forbidden = r_boundary, r_forbidden
        self.r_target, self.r_other = r_target, r_other
        self.start = tuple(start)
        self.n_states = n_rows * n_cols
        self.n_actions = len(ACTIONS)
        self._build_model()
        self.state = self.to_index(self.start)

    # ------------------------------------------------------------------ 编号转换
    def to_index(self, rc) -> int:
        return rc[0] * self.n_cols + rc[1]

    def to_rc(self, s: int):
        return divmod(int(s), self.n_cols)

    # ------------------------------------------------------------------ 环境模型
    def _next(self, s: int, a: int):
        """返回 (下一个状态, 奖励)。"""
        r, c = self.to_rc(s)
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            return s, self.r_boundary  # 撞墙：留在原地，受惩罚
        ns = self.to_index((nr, nc))
        if (nr, nc) == self.target:
            return ns, self.r_target
        if (nr, nc) in self.forbidden:
            return ns, self.r_forbidden
        return ns, self.r_other

    def _build_model(self) -> None:
        """构造完整的模型：P[s, a, s'] 与 R[s, a]（期望即时奖励）。"""
        S, A = self.n_states, self.n_actions
        self.P = np.zeros((S, A, S))
        self.R = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                ns, r = self._next(s, a)
                self.P[s, a, ns] = 1.0
                self.R[s, a] = r

    # ------------------------------------------------------------------ 交互接口（无模型方法使用）
    def reset(self, start=None) -> int:
        if start is None:
            start = self.start
        self.state = start if isinstance(start, (int, np.integer)) else self.to_index(start)
        return self.state

    def step(self, a: int):
        """与环境交互一步：返回 (下一个状态, 奖励)。网格世界是持续性任务，没有 done。"""
        ns, r = self._next(self.state, a)
        self.state = ns
        return ns, r

    # ------------------------------------------------------------------ 可视化
    def _draw_grid(self, ax):
        import matplotlib.patches as patches

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                color = "white"
                if (r, c) in self.forbidden:
                    color = "#f6c344"
                if (r, c) == self.target:
                    color = "#5ab4e5"
                ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="gray"))
        ax.set_xlim(0, self.n_cols)
        ax.set_ylim(self.n_rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(self.n_cols) + 0.5, [str(i) for i in range(self.n_cols)])
        ax.set_yticks(np.arange(self.n_rows) + 0.5, [str(i) for i in range(self.n_rows)])

    def plot(self, V=None, policy=None, title: str = "", ax=None):
        """画出网格、状态值 V（数字）与策略（箭头）。

        policy 可以是：形状 (S,) 的确定性动作编号，或形状 (S, A) 的概率矩阵。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        self._draw_grid(ax)
        for s in range(self.n_states):
            r, c = self.to_rc(s)
            if V is not None:
                ax.text(c + 0.5, r + 0.78, f"{V[s]:.1f}", ha="center", va="center", fontsize=8, color="black")
            if policy is not None:
                probs = np.asarray(policy[s], dtype=float)
                if probs.ndim == 0:
                    probs = np.eye(self.n_actions)[int(probs)]
                for a, p in enumerate(probs):
                    if p < 1e-3:
                        continue
                    if a == 4:
                        ax.add_patch(plt.Circle((c + 0.5, r + 0.4), 0.08 * p + 0.04, fill=False, color="green"))
                    else:
                        dr, dc = ACTIONS[a]
                        ax.arrow(
                            c + 0.5, r + 0.4, 0.3 * dc * p, 0.3 * dr * p,
                            head_width=0.1, head_length=0.08, fc="green", ec="green", length_includes_head=True,
                        )
        ax.set_title(title)
        return ax


def policy_to_matrix(policy, n_actions: int = 5) -> np.ndarray:
    """把确定性策略（动作编号数组）转成 (S, A) 概率矩阵。"""
    policy = np.asarray(policy)
    if policy.ndim == 2:
        return policy
    return np.eye(n_actions)[policy]
