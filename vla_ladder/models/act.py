"""ACT（Action Chunking with Transformers）的精简参考实现（第 5 章）。

参考：
  * 原版：https://github.com/tonyzhaozh/act （detr/models/detr_vae.py）
  * LeRobot：lerobot/common/policies/act/modeling_act.py

与原版的对应关系（原版 -> 本实现）：
  ResNet18 图像特征 + 2D 正弦位置编码   -> 可选的 image_encoder（第 5 章的点环境只用状态，不需要图像）
  CVAE 编码器：[CLS, qpos, a_1..a_k]   -> self.cvae_encoder，输出隐变量 z 的均值与 log 方差
  Transformer 编码器：[z, qpos, 图像特征] -> self.encoder
  Transformer 解码器：k 个可学习 query   -> self.decoder（DETR 风格，交叉注意力读取编码器输出）
  损失：L1 重建 + β·KL                   -> loss()
  推理：z = 0（先验均值）                 -> predict(..., sample_z=False)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_ladder.models.transformer import sinusoidal_pe


class ACT(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int = 16,
        dim: int = 128,
        n_heads: int = 4,
        enc_layers: int = 2,
        dec_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim, self.action_dim, self.chunk_size, self.latent_dim = state_dim, action_dim, chunk_size, latent_dim

        def enc_layer():
            return nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout, batch_first=True, norm_first=True)

        # ---------------- CVAE 编码器 q(z | s, a_{1:k})：只在训练时使用 ----------------
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.cvae_state_proj = nn.Linear(state_dim, dim)
        self.cvae_action_proj = nn.Linear(action_dim, dim)
        self.register_buffer("cvae_pos", sinusoidal_pe(chunk_size + 2, dim)[None], persistent=False)
        self.cvae_encoder = nn.TransformerEncoder(enc_layer(), enc_layers)
        self.latent_proj = nn.Linear(dim, 2 * latent_dim)  # -> (mu, logvar)

        # ---------------- 策略（CVAE 解码器）p(a_{1:k} | s, z) ----------------
        self.latent_out_proj = nn.Linear(latent_dim, dim)
        self.state_proj = nn.Linear(state_dim, dim)
        self.enc_pos = nn.Parameter(torch.randn(1, 2, dim) * 0.02)  # [z, state] 两个 token 的位置编码
        self.encoder = nn.TransformerEncoder(enc_layer(), enc_layers)
        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, dim) * 0.02)  # DETR 风格的 k 个动作 query
        dec_layer = nn.TransformerDecoderLayer(dim, n_heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.action_head = nn.Linear(dim, action_dim)

    # ------------------------------------------------------------------ CVAE 编码器
    def encode_latent(self, state, actions, is_pad=None):
        """state (B, S), actions (B, k, A) -> mu, logvar (B, latent)"""
        B = state.shape[0]
        tokens = torch.cat(
            [self.cls_embed.expand(B, -1, -1), self.cvae_state_proj(state)[:, None], self.cvae_action_proj(actions)], dim=1
        ) + self.cvae_pos
        pad = None
        if is_pad is not None:  # nn.Transformer 的 key_padding_mask：True = 忽略（与 openpi 相反！）
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=state.device), is_pad], dim=1)
        h = self.cvae_encoder(tokens, src_key_padding_mask=pad)
        mu, logvar = self.latent_proj(h[:, 0]).chunk(2, dim=-1)
        return mu, logvar

    # ------------------------------------------------------------------ 策略
    def decode(self, state, z):
        """state (B, S), z (B, latent) -> 动作块 (B, k, A)"""
        B = state.shape[0]
        memory = torch.cat([self.latent_out_proj(z)[:, None], self.state_proj(state)[:, None]], dim=1) + self.enc_pos
        memory = self.encoder(memory)
        h = self.decoder(self.query_embed.expand(B, -1, -1), memory)
        return self.action_head(h)

    def loss(self, state, actions, is_pad=None, kl_weight: float = 10.0):
        """返回 (总损失, 信息字典)。actions: (B, k, A)；is_pad: (B, k) True 表示补齐的位置"""
        mu, logvar = self.encode_latent(state, actions, is_pad)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()  # 重参数化
        pred = self.decode(state, z)
        l1 = (pred - actions).abs().mean(-1)  # (B, k)
        if is_pad is not None:
            l1 = (l1 * ~is_pad).sum() / (~is_pad).sum().clamp(min=1)
        else:
            l1 = l1.mean()
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}

    @torch.no_grad()
    def predict(self, state, sample_z: bool = False):
        """推理：原版 ACT 取 z = 0（先验均值）；sample_z=True 则从先验 N(0, I) 采样。"""
        B = state.shape[0]
        z = torch.randn(B, self.latent_dim, device=state.device) if sample_z else torch.zeros(B, self.latent_dim, device=state.device)
        return self.decode(state, z)


class TemporalEnsembler:
    """ACT 的时间集成：每一步都查询策略，把所有"覆盖当前时刻"的历史预测按指数权重平均。

    w_i = exp(-m * i)，i = 0 是**最早**的那次预测（原版 ACT 的约定：越早的预测权重越大）。
    """

    def __init__(self, chunk_size: int, m: float = 0.01):
        self.k, self.m = chunk_size, m
        self.buffer = []  # 每个元素：(预测时刻 t0, 动作块 (N, k, A))

    def reset(self):
        self.buffer = []

    def step(self, t: int, chunk):
        """t: 当前时刻；chunk: 当前时刻预测的动作块 (N, k, A)。返回当前要执行的动作 (N, A)"""
        self.buffer.append((t, chunk))
        self.buffer = [(t0, c) for (t0, c) in self.buffer if t - t0 < self.k]
        preds = torch.stack([c[:, t - t0] for (t0, c) in self.buffer], dim=0)  # (n, N, A)，按预测时间从早到晚
        w = torch.exp(-self.m * torch.arange(len(self.buffer), dtype=preds.dtype, device=preds.device))
        w = w / w.sum()
        return (w[:, None, None] * preds).sum(0)
