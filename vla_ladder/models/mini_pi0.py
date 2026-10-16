"""mini-π0：一个结构上忠实于 openpi PI0Pytorch 的迷你 VLA（第 7 章的完整参考实现）。

对照 openpi/src/openpi/models_pytorch/pi0_pytorch.py 与 gemma_pytorch.py：

    PI0Pytorch.embed_prefix      -> MiniPi0.embed_prefix     图像 token + 文字 token（+ π0.5 的离散状态 token）
    PI0Pytorch.embed_suffix      -> MiniPi0.embed_suffix     状态 token（π0）+ 带噪动作 token + 时间编码
    PaliGemmaWithExpertModel     -> MixtureOfTransformers    两套权重（VLM / 动作专家），每层拼接 q/k/v 共享一次注意力
    make_att_2d_masks            -> vla_ladder.models.transformer.make_att_2d_masks
    PI0Pytorch.forward           -> MiniPi0.forward          流匹配训练损失
    PI0Pytorch.sample_actions    -> MiniPi0.sample_actions   前缀 KV cache + 10 步 Euler 去噪
    PI0Pytorch.denoise_step      -> MiniPi0.denoise_step

规模（默认）：VLM 宽 128、动作专家宽 64、4 层、4 个头、head_dim 32，约 2M 参数（π0 约 3.3B）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_ladder.models.transformer import (
    AdaRMSNorm,
    Block,
    GeGLU,
    RMSNorm,
    apply_rope,
    make_att_2d_masks,
    rope_cos_sin,
    sinusoidal_time_embedding,
)


@dataclass
class MiniPi0Config:
    vocab_size: int = 32
    image_size: int = 48
    patch: int = 8
    vision_width: int = 128
    vision_depth: int = 2
    vlm_width: int = 128
    expert_width: int = 64
    depth: int = 4  # VLM 与动作专家层数相同（π0 都是 18 层）
    n_heads: int = 4  # 两个专家的头数、head_dim 必须相同，才能拼接 q/k/v
    head_dim: int = 32
    state_dim: int = 2
    action_dim: int = 2
    action_horizon: int = 10
    max_text_len: int = 16
    pi05: bool = False  # True：π0.5 风格（时间用 adaRMSNorm 注入、状态离散化后放进前缀）
    state_bins: int = 64  # π0.5 中把每一维状态离散成多少个 bin


# ----------------------------------------------------------------------------- 视觉塔（SigLIP 的迷你版）
class VisionTower(nn.Module):
    def __init__(self, cfg: MiniPi0Config):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, cfg.vision_width, kernel_size=cfg.patch, stride=cfg.patch)
        self.n_patches = (cfg.image_size // cfg.patch) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, cfg.vision_width) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(cfg.vision_width, 4, cfg.vision_width // 4, 4 * cfg.vision_width) for _ in range(cfg.vision_depth)]
        )
        self.norm = RMSNorm(cfg.vision_width)
        # PaliGemma 的 multi_modal_projector：把视觉特征投影到语言模型的宽度
        self.projector = nn.Linear(cfg.vision_width, cfg.vlm_width)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x, _ = blk(x)
        return self.projector(self.norm(x))


# ----------------------------------------------------------------------------- 一个专家的一层
class ExpertLayer(nn.Module):
    """Gemma 风格的一层（只有权重，注意力计算由 MixtureOfTransformers 统一完成）。"""

    def __init__(self, width: int, n_heads: int, head_dim: int, adarms_cond_dim: int | None = None):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.use_adarms = adarms_cond_dim is not None
        if self.use_adarms:
            self.input_layernorm = AdaRMSNorm(width, adarms_cond_dim)
            self.post_attention_layernorm = AdaRMSNorm(width, adarms_cond_dim)
        else:
            self.input_layernorm = RMSNorm(width)
            self.post_attention_layernorm = RMSNorm(width)
        self.q_proj = nn.Linear(width, inner, bias=False)
        self.k_proj = nn.Linear(width, inner, bias=False)
        self.v_proj = nn.Linear(width, inner, bias=False)
        self.o_proj = nn.Linear(inner, width, bias=False)
        self.mlp = GeGLU(width, 4 * width)

    def norm(self, which, x, cond):
        """返回 (归一化后的 x, gate)；普通 RMSNorm 的 gate 为 None"""
        layer = self.input_layernorm if which == "input" else self.post_attention_layernorm
        if self.use_adarms:
            return layer(x, cond)
        return layer(x), None

    def qkv(self, h):
        B, L, _ = h.shape
        split = lambda t: t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # noqa: E731
        return split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))


def gated_residual(x, y, gate):
    return x + y if gate is None else x + y * gate


class MixtureOfTransformers(nn.Module):
    """openpi 的 PaliGemmaWithExpertModel：VLM（处理前缀）与动作专家（处理后缀）在每一层共享一次注意力。

    关键：两套权重各自计算自己 token 的 q/k/v，然后**沿序列维拼接**，做一次注意力（掩码控制谁能看谁），
    再把输出切回两部分，各自过自己的 o_proj、残差、MLP。
    """

    def __init__(self, cfg: MiniPi0Config):
        super().__init__()
        self.cfg = cfg
        cond_dim = cfg.expert_width if cfg.pi05 else None
        self.vlm_layers = nn.ModuleList([ExpertLayer(cfg.vlm_width, cfg.n_heads, cfg.head_dim) for _ in range(cfg.depth)])
        self.expert_layers = nn.ModuleList(
            [ExpertLayer(cfg.expert_width, cfg.n_heads, cfg.head_dim, cond_dim) for _ in range(cfg.depth)]
        )
        self.vlm_norm = RMSNorm(cfg.vlm_width)
        self.expert_norm = AdaRMSNorm(cfg.expert_width, cond_dim) if cfg.pi05 else RMSNorm(cfg.expert_width)

    def final_norm(self, i, x, cond):
        if i == 0:
            return self.vlm_norm(x)
        return self.expert_norm(x, cond)[0] if self.cfg.pi05 else self.expert_norm(x)

    def forward(self, inputs_embeds, attention_mask, position_ids, adarms_cond=None, past_key_values=None, use_cache=False):
        """
        inputs_embeds: [prefix_embs 或 None, suffix_embs 或 None]
        attention_mask: bool (B, L_query, L_key)，True = 可以看
        position_ids: (B, L_query)（RoPE 用；padding 不占位置号）
        past_key_values: 推理时缓存的前缀 [(k, v)] * depth
        返回 ([prefix_out, suffix_out], 新的 kv 缓存或 None)
        """
        layers_per_expert = [self.vlm_layers, self.expert_layers]
        conds = [None, adarms_cond]
        active = [i for i, e in enumerate(inputs_embeds) if e is not None]
        hs = list(inputs_embeds)
        cos_sin = rope_cos_sin(position_ids, self.cfg.head_dim)
        mask4d = attention_mask[:, None]  # (B, 1, Lq, Lk)，广播到所有头
        new_cache = []
        for li in range(self.cfg.depth):
            qs, ks, vs, gates = [], [], [], {}
            for i in active:
                layer = layers_per_expert[i][li]
                h, gates[i] = layer.norm("input", hs[i], conds[i])
                q, k, v = layer.qkv(h)
                qs.append(q), ks.append(k), vs.append(v)
            q, k, v = torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2)  # 沿序列维拼接
            q, k = apply_rope(q, k, *cos_sin)
            if use_cache:
                new_cache.append((k, v))
            if past_key_values is not None:  # 推理：拼上缓存的前缀 K/V
                pk, pv = past_key_values[li]
                k, v = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask4d)  # 一次注意力
            att = att.transpose(1, 2).reshape(att.shape[0], att.shape[2], -1)
            start = 0
            for i in active:  # 切回各自的部分
                layer = layers_per_expert[i][li]
                L = hs[i].shape[1]
                out = layer.o_proj(att[:, start : start + L])
                start += L
                x = gated_residual(hs[i], out, gates[i])
                h, gate2 = layer.norm("post", x, conds[i])
                hs[i] = gated_residual(x, layer.mlp(h), gate2)
        outs = [None, None]
        for i in active:
            outs[i] = self.final_norm(i, hs[i], conds[i])
        return outs, (new_cache if use_cache else None)


# ----------------------------------------------------------------------------- mini-π0
class MiniPi0(nn.Module):
    def __init__(self, cfg: MiniPi0Config):
        super().__init__()
        self.cfg = cfg
        self.vision = VisionTower(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.vlm_width)
        nn.init.normal_(self.embed_tokens.weight, std=cfg.vlm_width**-0.5)
        self.moe = MixtureOfTransformers(cfg)
        W = cfg.expert_width
        self.action_in_proj = nn.Linear(cfg.action_dim, W)
        self.action_out_proj = nn.Linear(W, cfg.action_dim)
        if cfg.pi05:
            self.state_embed = nn.Embedding(cfg.state_dim * cfg.state_bins, cfg.vlm_width)  # 离散状态 token
            self.time_mlp_in = nn.Linear(W, W)
            self.time_mlp_out = nn.Linear(W, W)
        else:
            self.state_proj = nn.Linear(cfg.state_dim, W)
            self.action_time_mlp_in = nn.Linear(2 * W, W)
            self.action_time_mlp_out = nn.Linear(W, W)

    # ------------------------------------------------------------------ 前缀：图像 + 文字（+ π0.5 的状态）
    def discretize_state(self, state):
        """π0.5：把每一维状态（[-1,1]）离散成 state_bins 个 bin，每一维用不同的 id 段，返回 (B, state_dim) 的 id"""
        bins = ((state.clamp(-1, 1) + 1) / 2 * self.cfg.state_bins).long().clamp(max=self.cfg.state_bins - 1)
        offset = torch.arange(self.cfg.state_dim, device=state.device) * self.cfg.state_bins
        return bins + offset

    def embed_prefix(self, images, lang_tokens, lang_masks, state=None):
        img = self.vision(images)  # (B, N_img, Wv)
        B, n_img = img.shape[:2]
        lang = self.embed_tokens(lang_tokens) * math.sqrt(self.cfg.vlm_width)  # Gemma：乘 sqrt(width)
        embs = [img, lang]
        pad = [torch.ones(B, n_img, dtype=torch.bool, device=img.device), lang_masks]
        if self.cfg.pi05 and state is not None:  # VLM 预训练阶段没有状态输入
            st = self.state_embed(self.discretize_state(state)) * math.sqrt(self.cfg.vlm_width)
            embs.append(st)
            pad.append(torch.ones(B, st.shape[1], dtype=torch.bool, device=img.device))
        embs, pad = torch.cat(embs, 1), torch.cat(pad, 1)
        att = torch.zeros_like(pad, dtype=torch.long)  # 前缀内部全双向
        return embs, pad, att

    # ------------------------------------------------------------------ 后缀：（状态）+ 带噪动作 + 时间
    def embed_suffix(self, state, noisy_actions, timestep):
        B, H, _ = noisy_actions.shape
        W = self.cfg.expert_width
        time_emb = sinusoidal_time_embedding(timestep, W)  # 周期 4e-3 ~ 4.0，与 openpi 相同
        action_emb = self.action_in_proj(noisy_actions)
        embs, att = [], []
        if not self.cfg.pi05:
            embs.append(self.state_proj(state)[:, None])
            att += [1]  # 状态 token：新的一块（前缀看不到它）
            time_emb = time_emb[:, None].expand(B, H, W)
            x = torch.cat([action_emb, time_emb], -1)
            x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
            adarms_cond = None
        else:
            adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
            x = action_emb
        embs.append(x)
        att += [1] + [0] * (H - 1)  # 动作块：新的一块，块内双向
        embs = torch.cat(embs, 1)
        pad = torch.ones(B, embs.shape[1], dtype=torch.bool, device=embs.device)
        att = torch.tensor(att, device=embs.device)[None].expand(B, -1)
        return embs, pad, att, adarms_cond

    # ------------------------------------------------------------------ 训练
    def forward(self, images, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        """返回逐元素的流匹配损失 (B, H, action_dim)，与 openpi 一样 reduction='none'"""
        if noise is None:
            noise = torch.randn_like(actions)
        if time is None:
            time = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0)).sample((actions.shape[0],)).to(actions.device)
            time = time * 0.999 + 0.001
        te = time[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, lang_tokens, lang_masks, state)
        suffix_embs, suffix_pad, suffix_att, cond = self.embed_suffix(state, x_t, time)
        pad = torch.cat([prefix_pad, suffix_pad], 1)
        att = torch.cat([prefix_att, suffix_att], 1)
        att_2d = make_att_2d_masks(pad, att)
        position_ids = torch.cumsum(pad.long(), 1) - 1  # padding 不占位置号
        (_, suffix_out), _ = self.moe([prefix_embs, suffix_embs], att_2d, position_ids, adarms_cond=cond)
        v_t = self.action_out_proj(suffix_out[:, -self.cfg.action_horizon :])
        return F.mse_loss(u_t, v_t, reduction="none")

    # ------------------------------------------------------------------ 推理
    @torch.no_grad()
    def prefix_cache(self, images, lang_tokens, lang_masks, state=None):
        """只跑一次前缀，返回 (KV 缓存, prefix_pad, prefix 最后一层输出)"""
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, lang_tokens, lang_masks, state)
        att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad.long(), 1) - 1
        (prefix_out, _), cache = self.moe([prefix_embs, None], att_2d, pos, use_cache=True)
        return cache, prefix_pad, prefix_out

    def denoise_step(self, state, prefix_pad, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad, suffix_att, cond = self.embed_suffix(state, x_t, timestep)
        B, Ls = suffix_pad.shape
        Lp = prefix_pad.shape[1]
        prefix_2d = prefix_pad[:, None, :].expand(B, Ls, Lp)  # 后缀可以看所有有效的前缀 token
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full = torch.cat([prefix_2d, suffix_2d], 2)  # (B, Ls, Lp + Ls)
        pos = prefix_pad.long().sum(-1, keepdim=True) + torch.cumsum(suffix_pad.long(), 1) - 1
        (_, suffix_out), _ = self.moe([None, suffix_embs], full, pos, adarms_cond=cond, past_key_values=past_key_values)
        return self.action_out_proj(suffix_out[:, -self.cfg.action_horizon :])

    @torch.no_grad()
    def sample_actions(self, images, lang_tokens, lang_masks, state, num_steps: int = 10, noise=None):
        B = images.shape[0]
        if noise is None:
            noise = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, device=images.device)
        cache, prefix_pad, _ = self.prefix_cache(images, lang_tokens, lang_masks, state)
        dt = -1.0 / num_steps
        x_t, time = noise, 1.0
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad, cache, x_t, torch.full((B,), time, device=images.device))
            x_t = x_t + dt * v_t
            time += dt
        return x_t


def _rtc_prefix_weights(H: int, d: int, decay_end: int | None, device):
    """前 d 个位置为 1，之后按 exp(-3·frac) 衰减（与 FluxVLA compute_prefix_weights 的 'exp' 相同）"""
    decay_end = decay_end or 2 * d
    w = torch.zeros(H, device=device)
    w[:d] = 1.0
    L = decay_end - d
    for i in range(max(L, 0)):
        if d + i >= H:
            break
        w[d + i] = math.exp(-3.0 * i / max(L - 1, 1))
    return w


def _rtc_guidance_weight(tau: float, max_g: float) -> float:
    tau, omt = max(tau, 1e-6), max(1 - tau, 1e-6)
    return min((omt / tau) * (tau**2 + omt**2) / omt**2, max_g)


@torch.no_grad()
def sample_actions_rtc(model: MiniPi0, images, lang_tokens, lang_masks, state, prev_actions, d: int, num_steps: int = 10,
                       method: str = "guidance", max_guidance: float = 5.0, decay_end: int | None = None, noise=None):
    """实时分块（RTC, Real-Time Chunking），第 7.04 课。

    场景：推理需要 d 个控制周期，这期间机器人还在执行上一个块里的动作，所以新块的前 d 个动作
    "已经注定"等于 prev_actions[:, :d]。让新块与之衔接：
      method="replace"：每步把前 d 个位置替换成按当前时间加噪后的旧动作（简单，但约束力弱）
      method="guidance"：RTC 论文的引导式修补（FluxVLA rtc_guidance.py 的 openpi 约定版）：
            x0_hat = x_t - t·v；  v <- v - g(1-t) · W ⊙ (prev - x0_hat)
    """
    B = images.shape[0]
    H = model.cfg.action_horizon
    if noise is None:
        noise = torch.randn(B, H, model.cfg.action_dim, device=images.device)
    cache, prefix_pad, _ = model.prefix_cache(images, lang_tokens, lang_masks, state)
    W = _rtc_prefix_weights(H, d, decay_end, images.device)[None, :, None]
    dt = -1.0 / num_steps
    x_t, time = noise.clone(), 1.0
    while time >= -dt / 2:
        if method == "replace":
            x_t[:, :d] = time * noise[:, :d] + (1 - time) * prev_actions[:, :d]
        v_t = model.denoise_step(state, prefix_pad, cache, x_t, torch.full((B,), time, device=images.device))
        if method == "guidance":
            x0_hat = x_t - time * v_t
            v_t = v_t - _rtc_guidance_weight(1 - time, max_guidance) * W * (prev_actions - x0_hat)
        x_t = x_t + dt * v_t
        time += dt
    if method == "replace":
        x_t[:, :d] = prev_actions[:, :d]
    return x_t
