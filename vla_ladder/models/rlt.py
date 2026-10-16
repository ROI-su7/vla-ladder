"""RLT（RL Token）的精简参考实现（第 8 章）。

对照 RLinf：
  rlinf/models/embodiment/modules/rlt_token_transformer.py   -> RLTokenTransformer（编码器 + 自回归解码器）
  rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py       -> RLTActorCritic（参考动作作为输入、固定标准差、twin Q）
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py           -> rlt_losses（critic TD 损失；actor 损失 = -Q + β·BC）

两个阶段：
  Stage 1：冻结 VLA，用重建损失训练 RL token 编码器：把 VLA 的前缀 embedding（图像+文字 token）压缩成 1 个 token z_rl
  Stage 2：冻结 VLA 和编码器，在线训练小 actor-critic：
           actor  π(a | z_rl, proprio, a_ref) = N(a_ref + Δ, σ² I)（残差输出、Δ 头零初始化、σ 固定且很小）
           critic Q(z_rl, proprio, a)（twin Q，取 min）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_ladder.models.transformer import sinusoidal_pe


class RLTLayer(nn.Module):
    """与 RLinf 的 RLTSelfAttentionLayer 相同：Pre-LN 自注意力 + MLP。注意 nn.MultiheadAttention 的掩码 True = 禁止。"""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class RLTokenTransformer(nn.Module):
    """编码器：[prefix tokens..., RL token] -> 取最后一个位置 = z_rl
    解码器：[z_rl, prefix_1, ..., prefix_{L-1}] 因果注意力 -> 重建 [prefix_1, ..., prefix_L]（teacher forcing）"""

    def __init__(self, input_dim: int, embed_dim: int = 64, seq_len: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.rl_token = nn.Parameter(sinusoidal_pe(1, embed_dim).clone())
        self.enc_pos = nn.Parameter(sinusoidal_pe(seq_len + 1, embed_dim).clone())
        self.encoder = nn.ModuleList([RLTLayer(embed_dim, n_heads) for _ in range(n_layers)])
        self.teacher_proj = nn.Linear(input_dim, embed_dim)
        self.dec_pos = nn.Parameter(sinusoidal_pe(seq_len, embed_dim).clone())
        self.decoder = nn.ModuleList([RLTLayer(embed_dim, n_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(embed_dim, input_dim)

    @property
    def z_dim(self):
        return self.rl_token.shape[-1]

    def encode(self, prefix, mask=None):
        """prefix: (B, L, input_dim)，mask: bool (B, L) True = 有效。返回 z_rl (B, embed_dim)"""
        B, L, _ = prefix.shape
        x = torch.cat([self.in_proj(prefix), self.rl_token.expand(B, -1, -1)], 1) + self.enc_pos[: L + 1]
        kpm = None
        if mask is not None:
            kpm = ~torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], 1)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=kpm)
        return x[:, -1]

    def decode(self, z, prefix, mask=None):
        """自回归重建：第 i 个输出只能看 z 与 prefix[:i]。返回 (B, L, input_dim)"""
        B, L, _ = prefix.shape
        shifted = prefix[:, :-1]
        if mask is not None:
            shifted = shifted.masked_fill(~mask[:, :-1, None], 0.0)
        x = torch.cat([z[:, None], self.teacher_proj(shifted)], 1) + self.dec_pos[:L]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=prefix.device), 1)  # True = 禁止看未来
        for layer in self.decoder:
            x = layer(x, attn_mask=causal)
        return self.out_proj(x)

    def loss(self, prefix, mask=None):
        prefix = prefix.detach()  # VLA 是冻结的：梯度不会传回 VLA
        z = self.encode(prefix, mask)
        recon = self.decode(z, prefix, mask)
        err = (recon - prefix).pow(2).mean(-1)
        if mask is not None:
            return (err * mask).sum() / mask.sum()
        return err.mean()


def mlp(i, o, h=256, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class RLTActorCritic(nn.Module):
    def __init__(self, z_dim: int, proprio_dim: int, chunk_dim: int, hidden: int = 256, fixed_std: float = 0.05):
        super().__init__()
        self.actor = mlp(z_dim + proprio_dim + chunk_dim, chunk_dim, hidden)
        nn.init.zeros_(self.actor[-1].weight)  # 残差头零初始化：一开始 actor 完全复现 VLA 的参考动作
        nn.init.zeros_(self.actor[-1].bias)
        self.q1 = mlp(z_dim + proprio_dim + chunk_dim, 1, hidden)
        self.q2 = mlp(z_dim + proprio_dim + chunk_dim, 1, hidden)
        self.fixed_std = fixed_std

    def actor_mean(self, z, proprio, a_ref, ref_dropout: float = 0.0):
        ref_in = a_ref
        if ref_dropout > 0 and self.training:  # 参考动作 dropout：只在输入端丢弃，输出的残差基准不丢
            keep = (torch.rand(a_ref.shape[0], 1, device=a_ref.device) > ref_dropout).float()
            ref_in = a_ref * keep
        return a_ref + self.actor(torch.cat([z, proprio, ref_in], -1))

    def act(self, z, proprio, a_ref, deterministic=False):
        mean = self.actor_mean(z, proprio, a_ref)
        if deterministic:
            return mean
        return mean + self.fixed_std * torch.randn_like(mean)

    def q(self, z, proprio, a):
        x = torch.cat([z, proprio, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def rlt_losses(ac: RLTActorCritic, ac_targ: RLTActorCritic, batch, gamma: float = 0.95, bc_weight: float = 1.0, ref_dropout: float = 0.2):
    """返回 (critic_loss, actor_loss_fn)。actor 损失需在 critic 更新之后再计算（2.12 课讲过的 inplace 问题）。

    batch: z, proprio, a_ref, action, reward, next_z, next_proprio, next_a_ref, done
    """
    with torch.no_grad():
        a2 = ac_targ.act(batch["next_z"], batch["next_proprio"], batch["next_a_ref"])  # 目标策略加噪（TD3 风格的目标平滑）
        q1t, q2t = ac_targ.q(batch["next_z"], batch["next_proprio"], a2)
        y = batch["reward"] + gamma * (1 - batch["done"]) * torch.min(q1t, q2t)
    q1, q2 = ac.q(batch["z"], batch["proprio"], batch["action"])
    critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

    def actor_loss_fn():
        mean = ac.actor_mean(batch["z"], batch["proprio"], batch["a_ref"], ref_dropout)
        q1p, q2p = ac.q(batch["z"], batch["proprio"], mean)
        bc = (mean - batch["a_ref"]).pow(2).sum(-1).mean()
        return -torch.min(q1p, q2p).mean() + bc_weight * bc, bc

    return critic_loss, actor_loss_fn
