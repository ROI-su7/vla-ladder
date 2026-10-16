"""mini-π0 在玩具桌面任务上的完整训练流水线（第 7 章；第 8 章直接调用它得到"基础 VLA"）。

    data = build_vla_dataset(n_episodes, horizon)       采集示教并切成动作块
    pretrain_vlm(model, ...)                            阶段 1：VLM 预训练（看图 + 读指令 -> 定位目标方块）
    train_vla(model, data, ...)                         阶段 2：VLA 微调（流匹配学动作块）
    evaluate_vla(model, ...)                            在新场景上评估成功率
    get_base_vla(...)                                   以上全部，并把结果缓存到磁盘（第 8 章用）
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vla_ladder.envs import CharTokenizer, ToyTableEnv, collect_table_demos
from vla_ladder.models.mini_pi0 import MiniPi0, MiniPi0Config
from vla_ladder.models.transformer import make_att_2d_masks


# ----------------------------------------------------------------------------- 数据
def build_vla_dataset(n_episodes: int = 2000, horizon: int = 10, noise: float = 0.1, seed: int = 0, device="cpu",
                      expert_offset: float = 0.0, demo_steps: int = 50, demo_radius: float = 0.05):
    """采集示教并构造 (图像, 指令 id, 指令 mask, 状态, 动作块, 动作块 padding) 张量。

    expert_offset > 0 时模拟有系统性偏差的示教者（停在离目标 expert_offset 处），此时一般令 demo_radius=0，
    让示教一直持续 demo_steps 步（专家到达偏差点后停在那里），这样数据里包含"停在偏差点"的行为。"""
    tok = CharTokenizer()
    env = ToyTableEnv(num_envs=200, seed=seed, expert_offset=expert_offset, max_steps=demo_steps, success_radius=demo_radius)
    raw = collect_table_demos(env, n_episodes, noise=noise, seed=seed)
    ep = raw["episode"]
    last = np.zeros_like(ep)
    for i in range(len(ep) - 1, -1, -1):  # 每一帧所在回合的最后一帧下标
        last[i] = i if (i == len(ep) - 1 or ep[i + 1] != ep[i]) else last[i + 1]
    idx_raw = np.arange(len(ep))[:, None] + np.arange(horizon)[None]
    idx = np.minimum(idx_raw, last[:, None])
    ids, msk = tok.encode(raw["instruction"])
    return {
        "image": torch.tensor(raw["image"]).to(device),  # uint8，使用时再 /255
        "lang": torch.tensor(ids).to(device),
        "lang_mask": torch.tensor(msk).to(device),
        "state": torch.tensor(raw["state"]).to(device),
        "actions": torch.tensor(raw["action"])[torch.tensor(idx)].to(device),
        "action_pad": torch.tensor(idx_raw > last[:, None]).to(device),
        "episode": torch.tensor(ep).to(device),
        "tokenizer": tok,
    }


def obs_to_tensors(obs, tok: CharTokenizer, device):
    ids, msk = tok.encode(obs["instruction"])
    return (
        torch.as_tensor(obs["image"], device=device),
        torch.as_tensor(ids, device=device),
        torch.as_tensor(msk, device=device),
        torch.as_tensor(obs["state"], device=device),
    )


# ----------------------------------------------------------------------------- 阶段 1：VLM 预训练
def prefix_text_feature(model: MiniPi0, images, lang, lang_mask):
    """跑一遍前缀（只用 VLM 那一套权重），对文字 token 的输出做平均池化 -> (B, vlm_width)"""
    embs, pad, att = model.embed_prefix(images, lang, lang_mask)
    pos = torch.cumsum(pad.long(), 1) - 1
    (out, _), _ = model.moe([embs, None], make_att_2d_masks(pad, att), pos)
    txt = out[:, model.vision.n_patches : model.vision.n_patches + lang.shape[1]]
    w = lang_mask.float()[..., None]
    return (txt * w).sum(1) / w.sum(1)


def pretrain_vlm(model: MiniPi0, n_images: int = 20000, steps: int = 4000, batch_size: int = 128, lr: float = 1e-3, seed: int = 100, device="cpu", log=print):
    """视觉-语言对齐预训练：给定桌面图像与指令，回归目标方块的位置。
    只训练视觉塔、词嵌入与 VLM 层（就像 PaliGemma 在互联网图文数据上预训练过一样）。"""
    tok = CharTokenizer()
    env = ToyTableEnv(num_envs=n_images, seed=seed)
    obs = env.reset()
    images = torch.tensor(obs["image"]).to(device)
    ids, msk = tok.encode(obs["instruction"])
    lang, lang_mask = torch.tensor(ids).to(device), torch.tensor(msk).to(device)
    target = torch.tensor(env.target, dtype=torch.float32).to(device)
    head = nn.Linear(model.cfg.vlm_width, 2).to(device)
    params = [p for mod in (model.vision, model.embed_tokens, model.moe.vlm_layers, model.moe.vlm_norm) for p in mod.parameters()]
    opt = torch.optim.AdamW(params + list(head.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, n_images, (batch_size,), device=device)
        loss = ((head(prefix_text_feature(model, images[i], lang[i], lang_mask[i])) - target[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (step % 1000 == 0 or step == steps - 1):
            log(f"  [VLM 预训练] step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    return head


# ----------------------------------------------------------------------------- 阶段 2：VLA 微调
def train_vla(model: MiniPi0, data, steps: int = 8000, batch_size: int = 128, lr: float = 1e-3, device="cpu", log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    N = len(data["state"])
    t0 = time.time()
    model.train()
    for step in range(steps):
        i = torch.randint(0, N, (batch_size,), device=device)
        loss = model(data["image"][i].float() / 255, data["lang"][i], data["lang_mask"][i], data["state"][i], data["actions"][i])
        valid = (~data["action_pad"][i]).float()[..., None]  # 补齐的动作不计入损失
        loss = (loss * valid).sum() / (valid.sum() * loss.shape[-1])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if log and (step % 1000 == 0 or step == steps - 1):
            log(f"  [VLA 微调] step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    model.eval()
    return model


@torch.no_grad()
def evaluate_vla(model: MiniPi0, n: int = 200, radius: float = 0.05, exec_steps: int = 5, seed: int = 123, device="cpu", return_env=False):
    """在 n 个新场景上评估。返回 dict(success, median_dist)"""
    tok = CharTokenizer()
    env = ToyTableEnv(num_envs=n, seed=seed, success_radius=radius)
    obs = env.reset()
    model.eval()
    while not env.done.all():
        chunk = model.sample_actions(*obs_to_tensors(obs, tok, device)).clamp(-1, 1).cpu().numpy()
        for k in range(exec_steps):
            obs, _, _, info = env.step(chunk[:, k])
    out = {"success": float(info["success"].mean()), "median_dist": float(np.median(info["dist"]))}
    return (out, env) if return_env else out


# ----------------------------------------------------------------------------- 一键得到基础 VLA（缓存到磁盘）
def default_cache_dir() -> Path:
    return Path(os.environ.get("VLA_LADDER_CACHE", Path.home() / ".cache" / "vla_ladder"))


def get_base_vla(pi05: bool = False, n_episodes: int = 2000, horizon: int = 10, device="cpu", cache_dir=None, log=print,
                 expert_offset: float = 0.0, demo_steps: int = 50):
    """返回在玩具桌面上训练好的 mini-π0（预训练 VLM + 示教微调）。第一次调用会训练并保存，之后直接加载。

    expert_offset > 0：用"有系统性偏差"的示教训练（第 8 章 RLT / RECAP 用）。"""
    cache_dir = Path(cache_dir or default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_off{expert_offset:g}_T{demo_steps}" if expert_offset > 0 else ""
    path = cache_dir / f"mini_pi0_{'pi05' if pi05 else 'pi0'}_ep{n_episodes}_h{horizon}{tag}.pt"
    tok = CharTokenizer()
    cfg = MiniPi0Config(vocab_size=tok.vocab_size, action_horizon=horizon, pi05=pi05)
    torch.manual_seed(0)
    model = MiniPi0(cfg).to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        if log:
            log(f"已从 {path} 加载基础 VLA")
        return model.eval()
    if log:
        log("没有找到缓存的基础 VLA，开始训练（GPU 约 3~4 分钟）……")
    pretrain_vlm(model, device=device, log=log)
    if expert_offset > 0:
        data = build_vla_dataset(n_episodes, horizon, device=device, expert_offset=expert_offset, demo_steps=demo_steps, demo_radius=0.0)
    else:
        data = build_vla_dataset(n_episodes, horizon, device=device)
    # π0.5 变体（实验性）：adaRMSNorm 对学习率更敏感，1e-3 会不稳定，这里用 3e-4 并多训练一倍步数
    train_vla(model, data, steps=16000 if pi05 else 8000, lr=3e-4 if pi05 else 1e-3, device=device, log=log)
    torch.save(model.state_dict(), path)
    if log:
        log(f"已保存到 {path}")
    return model.eval()


def get_pretrained_vlm_state(device="cpu", horizon: int = 10, cache_dir=None, log=print):
    """返回"只做过阶段 1（VLM 预训练）"的 mini-π0 的 state_dict，并缓存到磁盘（第 8.03 课 RECAP 用作初始化）。"""
    cache_dir = Path(cache_dir or default_cache_dir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"mini_pi0_vlm_pretrained_h{horizon}.pt"
    if path.exists():
        if log:
            log(f"已从 {path} 加载预训练 VLM")
        return torch.load(path, map_location=device)
    tok = CharTokenizer()
    torch.manual_seed(0)
    model = MiniPi0(MiniPi0Config(vocab_size=tok.vocab_size, action_horizon=horizon)).to(device)
    pretrain_vlm(model, device=device, log=log)
    torch.save(model.state_dict(), path)
    return model.state_dict()
