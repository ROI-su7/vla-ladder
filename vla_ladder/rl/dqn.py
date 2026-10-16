"""DQN（第 2.08 课）：回放缓冲区、DQN 损失（目标网络 + terminated 处理）、CartPole 训练循环。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_ladder.envs import CartPole
from vla_ladder.utils import set_seed


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, o, a, r, no, term):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i], self.next_obs[i], self.terminated[i] = o, a, r, no, term
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ["obs", "actions", "rewards", "next_obs", "terminated"]}


def dqn_loss(q_net, target_net, batch, gamma):
    q_sa = q_net(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)
    with torch.no_grad():
        target = batch["rewards"] + gamma * (1 - batch["terminated"]) * target_net(batch["next_obs"]).max(1).values
    return F.mse_loss(q_sa, target)


def make_q_net(obs_dim=4, n_actions=2, hidden=128):
    return nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))


def train_dqn(total_steps=30_000, seed=0, gamma=0.99, lr=5e-4, batch_size=64, buffer_size=20_000, learning_starts=1000,
              tau=0.005, eps_start=1.0, eps_end=0.05, eps_decay_steps=5_000):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPole()
    q_net, target_net = make_q_net(), make_q_net()
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, 4)
    obs, _ = env.reset(seed=seed)
    ep_ret, returns = 0.0, []
    for step in range(total_steps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if rng.random() < eps:
            a = int(rng.integers(2))
        else:
            with torch.no_grad():
                a = int(q_net(torch.as_tensor(obs)).argmax())
        nxt, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, nxt, float(term))
        obs, ep_ret = nxt, ep_ret + r
        if term or trunc:
            returns.append(ep_ret)
            obs, _ = env.reset()
            ep_ret = 0.0
        if step >= learning_starts:
            loss = dqn_loss(q_net, target_net, buf.sample(batch_size, rng), gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p_t, p in zip(target_net.parameters(), q_net.parameters()):
                    p_t.mul_(1 - tau).add_(p, alpha=tau)
    return q_net, returns
