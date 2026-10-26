"""AdamW construction and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_adamw(
    model: nn.Module, *, learning_rate: float, weight_decay: float, fused: bool = False
) -> torch.optim.AdamW:
    """Decay matrix weights only; vectors (norm scales) get no weight decay."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )


def cosine_lr(
    step: int, *, max_steps: int, peak: float, warmup_steps: int, min_fraction: float
) -> float:
    """Linear warmup to ``peak`` then cosine decay to ``peak * min_fraction``."""

    if warmup_steps and step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    return peak * (min_fraction + (1 - min_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))
