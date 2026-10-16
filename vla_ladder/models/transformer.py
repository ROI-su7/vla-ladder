"""Transformer 组件（第 4 章的完整参考实现）。

全部按 Gemma（π0 的语言模型/动作专家）的风格实现：
    Pre-RMSNorm（Gemma 风格 1+weight） + RoPE + 多头/多查询注意力 + GeGLU
掩码约定与 openpi 相同：bool 张量，True = 允许关注。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- 掩码
def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """openpi / big_vision 的掩码构造（4.01 课）。

    pad_masks: bool (B, N)，True = 真实 token
    att_masks: int/bool (B, N)，1 = 从这里开始一个新块（前面的 token 看不到它）
    返回 bool (B, N, N)：[b, i, j] = token i 能否看 token j
    """
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] & pad_masks[:, :, None]
    return att_2d & pad_2d


# ----------------------------------------------------------------------------- 位置编码
def sinusoidal_pe(seq_len: int, dim: int) -> torch.Tensor:
    """经典正弦位置编码 (seq_len, dim)。"""
    position = torch.arange(seq_len, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    """把标量时间 t ∈ [0,1] 编码成向量（与 openpi 的 create_sinusoidal_pos_embedding 相同）。t: (B,) -> (B, dim)"""
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = (2 * math.pi / period)[None, :] * t[:, None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float = 10000.0):
    """positions: (B, L) -> cos, sin: (B, L, head_dim)"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = positions[..., None].float() * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, h, L, hd)；cos, sin: (B, L, hd)"""
    cos, sin = cos[:, None].to(q.dtype), sin[:, None].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ----------------------------------------------------------------------------- 基础层
class RMSNorm(nn.Module):
    """Gemma 风格 RMSNorm：x / rms(x) * (1 + weight)，weight 初始化为 0。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * (1.0 + self.weight)


class AdaRMSNorm(nn.Module):
    """π0.5 动作专家用的自适应 RMSNorm：由条件向量 cond（流匹配时间 t 的编码）生成 scale/shift/gate。

    返回 (归一化后的 x, gate)。gate 用于"门控残差"：x + gate * f(x)。
    对应 openpi transformers_replace 里的 GemmaRMSNorm(cond=...) 与 _gated_residual：
      normed * (1 + scale) + shift，残差为 x + y * gate。
    区别：openpi 直接用 dense 输出的 gate（它加载预训练权重）；这里返回 1 + gate 并把 dense 全部零初始化，
    这样从零训练时一开始等价于"普通 RMSNorm + 普通残差"，更稳定。
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)  # 零初始化：一开始等价于普通 RMSNorm、gate=1 的残差

    def forward(self, x, cond):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        scale, shift, gate = self.dense(cond)[:, None, :].chunk(3, dim=-1)
        return x * (1 + scale) + shift, 1 + gate


class GeGLU(nn.Module):
    """Gemma 的 MLP：down(gelu(gate(x)) * up(x))"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """多头注意力，支持 RoPE 与 KV cache。"""

    def __init__(self, dim: int, n_heads: int, head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or dim // n_heads
        inner = n_heads * self.head_dim
        self.q_proj = nn.Linear(dim, inner, bias=False)
        self.k_proj = nn.Linear(dim, inner, bias=False)
        self.v_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)

    def qkv(self, x):
        B, L, _ = x.shape
        split = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        return split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))

    def forward(self, x, mask=None, cos_sin=None, kv_cache=None):
        """x: (B, L, dim)；mask: bool 可广播到 (B, 1, L, L_total)；cos_sin: RoPE 的 (cos, sin)；
        kv_cache: None 或 (k_past, v_past)。返回 (输出, 新的 (k, v))"""
        q, k, v = self.qkv(x)
        if cos_sin is not None:
            q, k = apply_rope(q, k, *cos_sin)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, _, L, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1)), (k, v)


class Block(nn.Module):
    """Gemma 风格 Transformer 层：x += attn(norm(x))；x += mlp(norm(x))"""

    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, n_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(dim)
        self.mlp = GeGLU(dim, mlp_hidden)

    def forward(self, x, mask=None, cos_sin=None, kv_cache=None):
        h, new_cache = self.self_attn(self.input_layernorm(x), mask, cos_sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


# ----------------------------------------------------------------------------- GPT
class GPT(nn.Module):
    """最小的 decoder-only 语言模型（4.03 课）。"""

    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 4, n_heads: int = 4, head_dim: int = 64, mlp_hidden: int = 1024):
        super().__init__()
        self.dim, self.head_dim = dim, head_dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, head_dim, mlp_hidden) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        nn.init.normal_(self.embed.weight, std=0.02)  # 共享输出层时 embedding 必须初始化得小，否则初始 logits 过大

    def forward(self, idx, kv_caches=None, start_pos: int = 0):
        """idx: (B, L) -> logits (B, L, V), 新的 kv_caches"""
        B, L = idx.shape
        x = self.embed(idx) * math.sqrt(self.dim)  # Gemma：embedding 乘 sqrt(dim)
        pos = torch.arange(start_pos, start_pos + L, device=idx.device)[None].expand(B, -1)
        cos_sin = rope_cos_sin(pos, self.head_dim)
        total = start_pos + L
        # 新 token i（绝对位置 start_pos+i）可以看所有绝对位置 <= start_pos+i 的 token
        mask = torch.arange(total, device=idx.device)[None, :] <= (start_pos + torch.arange(L, device=idx.device))[:, None]
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, mask, cos_sin, None if kv_caches is None else kv_caches[i])
            new_caches.append(c)
        logits = self.norm(x) @ self.embed.weight.T  # 输出层与输入 embedding 共享权重（weight tying）
        return logits, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, use_cache: bool = True):
        caches, pos = None, 0
        inp = idx
        for _ in range(max_new_tokens):
            if use_cache:
                logits, caches = self(inp, caches, pos)
                pos += inp.shape[1]
            else:
                logits, _ = self(idx)
            logits = logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            inp = nxt
        return idx


# ----------------------------------------------------------------------------- ViT
class ViT(nn.Module):
    """最小的 Vision Transformer（4.04 课）：patch embedding + 可学习位置编码 + 双向 Transformer。

    输出所有 patch 的 token (B, N, dim)（与 SigLIP 一样，不使用 CLS token）。
    """

    def __init__(self, image_size: int = 48, patch: int = 8, dim: int = 128, depth: int = 4, n_heads: int = 4, mlp_hidden: int = 512, in_ch: int = 3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.n_patches = (image_size // patch) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)
        self.blocks = nn.ModuleList([Block(dim, n_heads, dim // n_heads, mlp_hidden) for _ in range(depth)])
        self.norm = RMSNorm(dim)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed  # (B, N, dim)
        for blk in self.blocks:
            x, _ = blk(x)
        return self.norm(x)
