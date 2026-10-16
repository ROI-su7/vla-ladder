"""经典控制环境：CartPole（倒立摆小车）与 Pendulum（单摆）。

为了让课程不依赖 gym/gymnasium 的版本，这里用纯 numpy 重写，动力学公式与
gymnasium 的 classic_control 实现保持一致，接口也模仿 gymnasium：
    obs, info = env.reset(seed=...)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math

import numpy as np


class CartPole:
    """离散动作：0 向左推，1 向右推。每存活一步奖励 +1，最多 500 步。"""

    obs_dim = 4
    n_actions = 2

    def __init__(self, max_steps: int = 500):
        self.gravity, self.masscart, self.masspole = 9.8, 1.0, 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 摆杆长度的一半
        self.polemass_length = self.masspole * self.length
        self.force_mag, self.tau = 10.0, 0.02
        self.theta_threshold = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.state = None
        self.t = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.astype(np.float32), {}

    def step(self, action: int):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.t += 1
        terminated = bool(
            x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold or theta > self.theta_threshold
        )
        truncated = self.t >= self.max_steps
        return self.state.astype(np.float32), 1.0, terminated, truncated, {}


class Pendulum:
    """连续动作：力矩 u ∈ [-2, 2]。目标是把摆甩到竖直向上并保持住。每局 200 步。"""

    obs_dim = 3
    act_dim = 1
    act_high = 2.0

    def __init__(self, max_steps: int = 200):
        self.max_speed, self.max_torque = 8.0, 2.0
        self.dt, self.g, self.m, self.l = 0.05, 10.0, 1.0, 1.0
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.state = None
        self.t = 0

    def _obs(self):
        th, thdot = self.state
        return np.array([math.cos(th), math.sin(th), thdot], dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform([-math.pi, -1.0], [math.pi, 1.0])
        self.t = 0
        return self._obs(), {}

    def step(self, u):
        th, thdot = self.state
        u = float(np.clip(np.asarray(u).reshape(-1)[0], -self.max_torque, self.max_torque))
        angle_norm = ((th + math.pi) % (2 * math.pi)) - math.pi
        cost = angle_norm**2 + 0.1 * thdot**2 + 0.001 * u**2
        newthdot = thdot + (3 * self.g / (2 * self.l) * math.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        newth = th + newthdot * self.dt
        self.state = np.array([newth, newthdot])
        self.t += 1
        return self._obs(), -cost, False, self.t >= self.max_steps, {}


class SyncVecEnv:
    """把多个单环境打包成一个"向量环境"（同步、串行执行），并在回合结束时自动 reset。

    PPO / A2C 常用：同时跑 N 个环境，一次前向得到 N 个动作。
        obs = venv.reset(seed)                       -> (N, obs_dim)
        obs, r, term, trunc, info = venv.step(acts)  -> 每项第 0 维为 N
    回合结束的环境会自动重置，此时返回的 obs 是新回合的第一个观测，
    结束前的最后一个观测放在 info["final_obs"][i]；完成回合的总回报放在 info["episode_returns"]。
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._ret = np.zeros(self.num_envs)

    def reset(self, seed: int = 0):
        self._ret[:] = 0
        return np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])

    def step(self, actions):
        obs, rews, terms, truncs = [], [], [], []
        info = {"final_obs": [None] * self.num_envs, "episode_returns": []}
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, te, tr, _ = e.step(a)
            self._ret[i] += r
            if te or tr:
                info["final_obs"][i] = o
                info["episode_returns"].append(self._ret[i])
                self._ret[i] = 0
                o, _ = e.reset()
            obs.append(o)
            rews.append(r)
            terms.append(te)
            truncs.append(tr)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(terms), np.array(truncs), info
