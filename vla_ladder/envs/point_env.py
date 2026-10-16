"""二维点到达环境：模仿学习（第 3~6 章）用的最小“机器人”。

为什么用二维点？——真机任务的难点（数据、多模态、精度）在这里都能复现，
但训练只要几十秒，非常适合理解算法本质。

两个环境都是“批量环境”：一次同时模拟 num_envs 个独立的回合，所有数组第 0 维是环境编号。

PointReachEnv     ：从随机起点走到随机目标点（单峰任务，行为克隆就能学好）
ObstacleReachEnv  ：中间有一个圆形障碍物，专家随机从左边或右边绕过去（双峰任务，
                    用 MSE 行为克隆会“取平均”直接撞上障碍物——这正是 ACT / 扩散策略要解决的问题）
"""

from __future__ import annotations

import numpy as np


class PointReachEnv:
    """观测 obs = [x, y, goal_x, goal_y]，动作 a ∈ [-1, 1]^2，位置更新 p += step_size * a。"""

    obs_dim = 4
    act_dim = 2

    def __init__(self, num_envs: int = 1, step_size: float = 0.05, max_steps: int = 60, success_radius: float = 0.05, seed=None):
        self.num_envs, self.step_size, self.max_steps = num_envs, step_size, max_steps
        self.success_radius = success_radius
        self.rng = np.random.default_rng(seed)

    def reset(self):
        n = self.num_envs
        self.pos = self.rng.uniform(-0.9, 0.9, size=(n, 2))
        self.goal = self.rng.uniform(-0.9, 0.9, size=(n, 2))
        self.t = 0
        self.done = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        return self.obs()

    def obs(self):
        return np.concatenate([self.pos, self.goal], axis=1).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(self.num_envs, 2), -1, 1)
        active = ~self.done
        self.pos[active] = np.clip(self.pos[active] + self.step_size * action[active], -1, 1)
        self.t += 1
        dist = np.linalg.norm(self.pos - self.goal, axis=1)
        reached = dist < self.success_radius
        self.success |= reached & active
        reward = -dist * active
        self.done |= reached | (self.t >= self.max_steps)
        return self.obs(), reward.astype(np.float32), self.done.copy(), {"success": self.success.copy()}

    def expert_action(self, noise: float = 0.0):
        """专家：直接朝目标走（带一点噪声模拟人类示教的不完美）。"""
        a = (self.goal - self.pos) / self.step_size
        a = np.clip(a, -1, 1)
        if noise > 0:
            a = a + noise * self.rng.standard_normal(a.shape)
        return np.clip(a, -1, 1).astype(np.float32)


class ObstacleReachEnv:
    """起点在下方、目标在上方、中间有圆形障碍物。

    观测 obs = [x, y]（目标位置固定，所以不需要放进观测）；include_velocity=True 时再加上一步的速度 [vx, vy]
    碰到障碍物 -> 失败并结束回合；进入目标圈 -> 成功。
    专家：每个回合开始时以 50% 概率选择“从左绕”或“从右绕”，这让同一个观测对应两种截然不同的动作（双峰）。
    """

    obs_dim = 2
    act_dim = 2

    def __init__(
        self,
        num_envs: int = 1,
        step_size: float = 0.05,
        max_steps: int = 80,
        obstacle=(0.0, 0.0),
        obstacle_radius: float = 0.3,
        goal=(0.0, 0.8),
        success_radius: float = 0.1,
        start_noise: float = 0.0,
        include_velocity: bool = False,
        seed=None,
    ):
        self.num_envs, self.step_size, self.max_steps = num_envs, step_size, max_steps
        self.start_noise = start_noise
        self.include_velocity = include_velocity  # True 时观测为 [x, y, vx, vy]（上一步的位移/step_size），相当于"两帧观测历史"
        self.obs_dim = 4 if include_velocity else 2
        self.obstacle = np.array(obstacle, dtype=float)
        self.obstacle_radius = obstacle_radius
        self.goal_pos = np.array(goal, dtype=float)
        self.success_radius = success_radius
        self.rng = np.random.default_rng(seed)

    def reset(self):
        n = self.num_envs
        w = self.start_noise
        self.pos = np.stack([self.rng.uniform(-w, w, n), self.rng.uniform(-0.8 - w / 2, -0.8 + w / 2, n)], axis=1)
        self.prev_pos = self.pos.copy()
        self.side = self.rng.choice([-1.0, 1.0], size=n)  # 专家的绕行方向（策略看不到）
        self.t = 0
        self.done = np.zeros(n, dtype=bool)
        self.success = np.zeros(n, dtype=bool)
        self.collided = np.zeros(n, dtype=bool)
        return self.obs()

    def obs(self):
        if self.include_velocity:
            vel = (self.pos - self.prev_pos) / self.step_size
            return np.concatenate([self.pos, vel], axis=1).astype(np.float32)
        return self.pos.astype(np.float32).copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float).reshape(self.num_envs, 2), -1, 1)
        active = ~self.done
        self.prev_pos = self.pos.copy()
        self.pos[active] = np.clip(self.pos[active] + self.step_size * action[active], -1, 1)
        self.t += 1
        hit = (np.linalg.norm(self.pos - self.obstacle, axis=1) < self.obstacle_radius) & active
        reached = (np.linalg.norm(self.pos - self.goal_pos, axis=1) < self.success_radius) & active & ~hit
        self.collided |= hit
        self.success |= reached
        reward = reached.astype(np.float32) - hit.astype(np.float32)
        self.done |= hit | reached | (self.t >= self.max_steps)
        return self.obs(), reward, self.done.copy(), {"success": self.success.copy(), "collided": self.collided.copy()}

    def expert_action(self, noise: float = 0.0):
        """先走到障碍物侧面的路点，再走向目标。"""
        waypoint = np.stack([self.side * (self.obstacle_radius + 0.25), np.full(self.num_envs, self.obstacle[1])], axis=1)
        below = self.pos[:, 1] < self.obstacle[1] - 0.02
        far_side = np.abs(self.pos[:, 0]) < self.obstacle_radius + 0.2
        target = np.where((below & far_side)[:, None], waypoint, self.goal_pos[None, :])
        a = np.clip((target - self.pos) / self.step_size, -1, 1)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0)  # 限制速度大小
        if noise > 0:
            a = a + noise * self.rng.standard_normal(a.shape)
        return np.clip(a, -1, 1).astype(np.float32)

    def plot(self, trajs=None, ax=None, title=""):
        """画出障碍物、目标和若干条轨迹（trajs: list of (T, 2) 数组）。"""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        ax.add_patch(plt.Circle(self.obstacle, self.obstacle_radius, color="#e57373", alpha=0.6))
        ax.add_patch(plt.Circle(self.goal_pos, self.success_radius, color="#81c784", alpha=0.6))
        if trajs is not None:
            for tr in trajs:
                tr = np.asarray(tr)
                ax.plot(tr[:, 0], tr[:, 1], "-", lw=1, alpha=0.7)
                ax.plot(tr[0, 0], tr[0, 1], "k.", ms=3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
        return ax


def collect_demos(env, n_episodes: int, noise: float = 0.0, seed: int = 0):
    """用专家策略收集示教数据。

    返回 list[dict]，每条轨迹包含 obs (T, obs_dim) 与 actions (T, act_dim)。
    （真机上这一步就是“遥操作采集数据”。）
    """
    env.rng = np.random.default_rng(seed)
    episodes = []
    n_batches = int(np.ceil(n_episodes / env.num_envs))
    for _ in range(n_batches):
        obs = env.reset()
        obs_buf, act_buf, alive_buf = [], [], []
        while not env.done.all():
            act = env.expert_action(noise=noise)
            alive_buf.append(~env.done)
            obs_buf.append(obs)
            act_buf.append(act)
            obs, _, _, _ = env.step(act)
        obs_arr, act_arr, alive_arr = np.stack(obs_buf, 1), np.stack(act_buf, 1), np.stack(alive_buf, 1)
        for i in range(env.num_envs):
            T = int(alive_arr[i].sum())
            episodes.append({"obs": obs_arr[i, :T], "actions": act_arr[i, :T], "success": bool(env.success[i])})
    return episodes[:n_episodes]
