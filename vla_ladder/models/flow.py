"""生成式动作头：DDPM 扩散与流匹配（第 6 章的完整参考实现）。

参考：
  * DDPM：Ho et al. 2020；Diffusion Policy：Chi et al. 2023（https://github.com/real-stanford/diffusion_policy）
  * 流匹配：Lipman et al. 2023；Rectified Flow：Liu et al. 2023
  * π0 的约定（openpi/src/openpi/models_pytorch/pi0_pytorch.py）：
        x_t = t * noise + (1 - t) * actions        t=1 是纯噪声，t=0 是数据
        u_t = noise - actions                       网络预测的"速度"目标
        推理：从 t=1 积分到 t=0，Euler 步长 dt = -1/num_steps
        训练时间采样：t = Beta(1.5, 1) * 0.999 + 0.001
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from vla_ladder.models.transformer import sinusoidal_time_embedding


# ----------------------------------------------------------------------------- 去噪网络
class FiLMResBlock(nn.Module):
    """带 FiLM 条件调制的残差 MLP 块：h = h + MLP(LN(h) * (1 + scale) + shift)。"""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class ChunkDenoiser(nn.Module):
    """预测"噪声"（DDPM）或"速度"（流匹配）的网络。

    输入：带噪动作块 x (B, H, A)、时间 t (B,)、观测条件 obs (B, obs_dim)
    输出：与 x 同形状 (B, H, A)
    结构：把整个动作块展平 -> 残差 MLP，每一块用 FiLM 注入 (时间编码 + 观测编码)。
    （Diffusion Policy 原版用 1D 卷积 UNet；这里用 MLP，原理相同、代码更短。）
    """

    def __init__(self, obs_dim: int, act_dim: int, horizon: int, dim: int = 256, depth: int = 4, time_dim: int = 64):
        super().__init__()
        self.horizon, self.act_dim, self.time_dim = horizon, act_dim, time_dim
        cond_dim = dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.obs_mlp = nn.Sequential(nn.Linear(obs_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.in_proj = nn.Linear(horizon * act_dim, dim)
        self.blocks = nn.ModuleList([FiLMResBlock(dim, cond_dim) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, horizon * act_dim))

    def forward(self, x, t, obs):
        B = x.shape[0]
        cond = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim)) + self.obs_mlp(obs)
        h = self.in_proj(x.reshape(B, -1))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h).view(B, self.horizon, self.act_dim)


# ----------------------------------------------------------------------------- DDPM
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """改进的余弦噪声表（Nichol & Dhariwal 2021；Diffusion Policy 的 squaredcos_cap_v2）。"""
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_bar = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=0.999).float()


class DDPM:
    """离散时间扩散：预测噪声 ε。

    前向加噪：x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
    训练：    min ||ε - ε_θ(x_t, t, obs)||²
    采样：    x_{t-1} = 1/sqrt(α_t) (x_t - β_t/sqrt(1-ᾱ_t) ε_θ) + σ_t z
    """

    def __init__(self, T: int = 100, device="cpu"):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t, noise):
        ab = self.alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0, obs):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t.float() / self.T, obs)
        return ((pred - noise) ** 2).mean()

    @torch.no_grad()
    def sample(self, model, obs, shape, generator=None):
        x = torch.randn(shape, device=obs.device, generator=generator)
        for t in reversed(range(self.T)):
            tt = torch.full((shape[0],), t, device=obs.device)
            eps = model(x, tt.float() / self.T, obs)
            a, ab, b = self.alphas[t], self.alphas_bar[t], self.betas[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if t > 0:
                var = b * (1 - self.alphas_bar[t - 1]) / (1 - ab)  # 后验方差 β̃_t
                x = mean + var.sqrt() * torch.randn(shape, device=obs.device, generator=generator)
            else:
                x = mean
        return x


# ----------------------------------------------------------------------------- 流匹配（π0 约定）
def sample_flow_time(B: int, device, alpha: float = 1.5, beta: float = 1.0) -> torch.Tensor:
    """π0 的训练时间采样：Beta(1.5, 1) * 0.999 + 0.001（偏向 t≈1，即噪声更多的一端）。"""
    t = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta)).sample((B,)).to(device)
    return t * 0.999 + 0.001


def flow_matching_loss(model, actions, obs, time=None, noise=None):
    """π0 的训练目标（逐元素 MSE 的均值）。actions: (B, H, A)"""
    B = actions.shape[0]
    noise = torch.randn_like(actions) if noise is None else noise
    t = sample_flow_time(B, actions.device) if time is None else time
    te = t[:, None, None]
    x_t = te * noise + (1 - te) * actions
    u_t = noise - actions
    v_t = model(x_t, t, obs)
    return ((v_t - u_t) ** 2).mean()


@torch.no_grad()
def flow_sample(model, obs, shape, num_steps: int = 10, noise=None):
    """π0 的推理：从 t=1 的纯噪声出发，Euler 积分到 t=0。"""
    x = torch.randn(shape, device=obs.device) if noise is None else noise
    dt = -1.0 / num_steps
    t = 1.0
    for _ in range(num_steps):
        v = model(x, torch.full((shape[0],), t, device=obs.device), obs)
        x = x + dt * v
        t += dt
    return x
