"""PPO（第 2.10~2.11 课）：GAE、裁剪损失、近似 KL、离散 / 高斯 actor-critic、通用训练循环。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, Normal

from vla_ladder.envs import SyncVecEnv
from vla_ladder.utils import set_seed


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """rewards/values/dones: (T, N)；last_values: (N,)。返回 (advantages, returns)"""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        nv = last_values if t == T - 1 else values[t + 1]
        nd = 1.0 - dones[t]
        last = rewards[t] + gamma * nv * nd - values[t] + gamma * lam * nd * last
        adv[t] = last
    return adv, adv + values


def ppo_policy_loss(new_logp, old_logp, adv, clip_eps=0.2):
    ratio = (new_logp - old_logp).exp()
    loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv).mean()
    return loss, ((ratio - 1).abs() > clip_eps).float().mean()


def approx_kl(new_logp, old_logp):
    log_ratio = new_logp - old_logp
    return ((log_ratio.exp() - 1) - log_ratio).mean()


class DiscreteAC(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def dist(self, obs):
        return Categorical(logits=self.actor(obs))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class GaussianAC(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, act_scale=1.0):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.act_scale = act_scale

    def dist(self, obs):
        mu = self.mu(obs) * self.act_scale
        return Independent(Normal(mu, self.log_std.exp().expand_as(mu) * self.act_scale), 1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def ppo(make_env, model, total_steps, n_envs=8, n_steps=128, epochs=10, n_minibatch=4, lr=7e-4, gamma=0.99, lam=0.95,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, reward_scale=1.0, seed=0):
    """返回 (回合回报列表, 对应的环境步数)"""
    set_seed(seed)
    venv = SyncVecEnv([make_env for _ in range(n_envs)])
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    obs = torch.as_tensor(venv.reset(seed=seed), dtype=torch.float32)
    ep_returns, ep_steps, step = [], [], 0
    while step < total_steps:
        buf = {k: [] for k in ["obs", "actions", "logp", "rewards", "dones", "values"]}
        for _ in range(n_steps):
            with torch.no_grad():
                d = model.dist(obs); a = d.sample(); logp = d.log_prob(a); v = model.value(obs)
            nxt, r, term, trunc, info = venv.step(a.numpy())
            r = torch.as_tensor(r) * reward_scale
            for i in np.nonzero(trunc & ~term)[0]:
                with torch.no_grad():
                    r[i] += gamma * model.value(torch.as_tensor(info["final_obs"][i], dtype=torch.float32))
            for k, val in zip(buf, [obs, a, logp, r, torch.as_tensor(term | trunc, dtype=torch.float32), v]):
                buf[k].append(val)
            obs = torch.as_tensor(nxt, dtype=torch.float32)
            step += n_envs
            ep_returns += info["episode_returns"]
            ep_steps += [step] * len(info["episode_returns"])
        data = {k: torch.stack(v) for k, v in buf.items()}
        with torch.no_grad():
            adv, ret = compute_gae(data["rewards"], data["values"], data["dones"], model.value(obs), gamma, lam)
        flat = {k: v.reshape(n_steps * n_envs, *v.shape[2:]) for k, v in data.items()}
        flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
        B = n_steps * n_envs
        mb = B // n_minibatch
        for _ in range(epochs):
            perm = torch.randperm(B)
            for s in range(0, B, mb):
                idx = perm[s : s + mb]
                d = model.dist(flat["obs"][idx])
                a_ = flat["adv"][idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg, _ = ppo_policy_loss(d.log_prob(flat["actions"][idx]), flat["logp"][idx], a_, clip_eps)
                vl = F.mse_loss(model.value(flat["obs"][idx]), flat["ret"][idx])
                loss = pg + vf_coef * vl - ent_coef * d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm); opt.step()
    return ep_returns, ep_steps
