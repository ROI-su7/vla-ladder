"""课程通用小工具。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """固定所有随机数种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(verbose: bool = True) -> torch.device:
    """有 GPU 用 GPU，否则用 CPU。"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"使用设备：{dev}")
    return dev


def count_params(module: torch.nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_close(name: str, got, want, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    """练习自测用：比较两个张量/数组，失败时给出清晰的报错。"""
    got_t = torch.as_tensor(np.asarray(got) if not torch.is_tensor(got) else got).detach().cpu().double()
    want_t = torch.as_tensor(np.asarray(want) if not torch.is_tensor(want) else want).detach().cpu().double()
    if got_t.shape != want_t.shape:
        raise AssertionError(f"❌ {name}: 形状不对，得到 {tuple(got_t.shape)}，期望 {tuple(want_t.shape)}")
    if not torch.allclose(got_t, want_t, atol=atol, rtol=rtol):
        err = (got_t - want_t).abs().max().item()
        raise AssertionError(f"❌ {name}: 数值不对，最大误差 {err:.3e}")
    print(f"✅ {name} 通过")


def moving_average(x, k: int = 10) -> np.ndarray:
    """画训练曲线时做平滑。"""
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")
