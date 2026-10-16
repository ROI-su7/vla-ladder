"""SAC（第 2.12 课）：tanh 压缩高斯 actor、twin Q、三个损失（先更新 critic 再算 actor 损失）、Pendulum 训练循环。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from vla_ladder.envs import Pendulum
from vla_ladder.utils import set_seed

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_high, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.act_high = act_high

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mu = self.mu(h)
        std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        logp = dist.log_prob(u).sum(-1) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return torch.tanh(u) * self.act_high, logp


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


def sac_critic_loss(batch, actor, q1, q2, q1_targ, q2_targ, alpha, gamma):
    with torch.no_grad():
        a2, logp2 = actor(batch["next_obs"])
        q_next = torch.min(q1_targ(batch["next_obs"], a2), q2_targ(batch["next_obs"], a2))
        y = batch["rew"] + gamma * (1 - batch["done"]) * (q_next - alpha * logp2)
    return F.mse_loss(q1(batch["obs"], batch["act"]), y) + F.mse_loss(q2(batch["obs"], batch["act"]), y)


def sac_actor_alpha_loss(batch, actor, q1, q2, log_alpha, target_entropy):
    alpha = log_alpha.exp().detach()
    a_new, logp_new = actor(batch["obs"])
    actor_loss = (alpha * logp_new - torch.min(q1(batch["obs"], a_new), q2(batch["obs"], a_new))).mean()
    alpha_loss = -(log_alpha * (logp_new.detach() + target_entropy)).mean()
    return actor_loss, alpha_loss


def train_sac(total_steps=12_000, start_steps=1000, batch_size=256, gamma=0.99, tau=0.005, lr=3e-4, seed=0):
    set_seed(seed)
    env = Pendulum()
    od, ad, high = 3, 1, 2.0
    actor = SquashedGaussianActor(od, ad, high)
    q1, q2, q1t, q2t = QNet(od, ad), QNet(od, ad), QNet(od, ad), QNet(od, ad)
    q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_pi = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    opt_a = torch.optim.Adam([log_alpha], lr=lr)
    cap = 100_000
    B = {"obs": np.zeros((cap, od), np.float32), "act": np.zeros((cap, ad), np.float32), "rew": np.zeros(cap, np.float32),
         "next_obs": np.zeros((cap, od), np.float32), "done": np.zeros(cap, np.float32)}
    ptr = size = 0
    obs, _ = env.reset(seed=seed)
    ep_ret, rets = 0.0, []
    for step in range(total_steps):
        if step < start_steps:
            act = np.random.uniform(-high, high, size=ad).astype(np.float32)
        else:
            with torch.no_grad():
                act = actor(torch.as_tensor(obs)[None])[0][0].numpy()
        obs2, r, term, trunc, _ = env.step(act)
        for k, v in zip(B, [obs, act, r, obs2, float(term)]):
            B[k][ptr] = v
        ptr, size = (ptr + 1) % cap, min(size + 1, cap)
        obs, ep_ret = obs2, ep_ret + r
        if term or trunc:
            rets.append(ep_ret); obs, _ = env.reset(); ep_ret = 0.0
        if step >= start_steps:
            i = np.random.randint(0, size, batch_size)
            batch = {k: torch.as_tensor(v[i]) for k, v in B.items()}
            ql = sac_critic_loss(batch, actor, q1, q2, q1t, q2t, log_alpha.exp().item(), gamma)
            opt_q.zero_grad(); ql.backward(); opt_q.step()
            al, alp = sac_actor_alpha_loss(batch, actor, q1, q2, log_alpha, -float(ad))
            opt_pi.zero_grad(); al.backward(); opt_pi.step()
            opt_a.zero_grad(); alp.backward(); opt_a.step()
            with torch.no_grad():
                for tgt, src in [(q1t, q1), (q2t, q2)]:
                    for pt, p in zip(tgt.parameters(), src.parameters()):
                        pt.mul_(1 - tau).add_(p, alpha=tau)
    return actor, rets
