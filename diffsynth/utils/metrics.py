"""PSNR / SSIM / LPIPS for NVS evaluation.

All metrics expect float tensors in [0, 1] with shape ``[B, 3, H, W]`` and
return a per-batch tensor of shape ``[B]`` on the input device.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def compute_psnr(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt = ground_truth.clamp(0.0, 1.0).float()
    pred = predicted.clamp(0.0, 1.0).float()
    mse = ((gt - pred) ** 2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d


@torch.no_grad()
def compute_ssim(ground_truth: Tensor, predicted: Tensor, window_size: int = 11) -> Tensor:
    gt = ground_truth.clamp(0.0, 1.0).float()
    pred = predicted.clamp(0.0, 1.0).float()
    C = gt.shape[1]

    window_2d = _gaussian_window(window_size, sigma=1.5, device=gt.device, dtype=gt.dtype)
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = F.conv2d(gt, window, padding=pad, groups=C)
    mu_y = F.conv2d(pred, window, padding=pad, groups=C)
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x_sq = F.conv2d(gt * gt, window, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(gt * pred, window, padding=pad, groups=C) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.flatten(1).mean(dim=1)


_lpips_model = None


def _get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "compute_lpips requires the `lpips` package. Install with: pip install lpips"
            ) from e
        _lpips_model = lpips.LPIPS(net="vgg").to(device)
        _lpips_model.eval()
        for p in _lpips_model.parameters():
            p.requires_grad_(False)
    elif next(_lpips_model.parameters()).device != device:
        _lpips_model = _lpips_model.to(device)
    return _lpips_model


@torch.no_grad()
def compute_lpips(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt = ground_truth.clamp(0.0, 1.0).float()
    pred = predicted.clamp(0.0, 1.0).float()
    model = _get_lpips_model(gt.device)
    # LPIPS expects [-1, 1]
    out = model(gt * 2.0 - 1.0, pred * 2.0 - 1.0)
    return out.view(out.shape[0])
