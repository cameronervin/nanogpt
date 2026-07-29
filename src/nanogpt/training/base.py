"""Base causal language-model training entry points."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .common import Batch, TrainLoopConfig, build_adamw, train_updates


def train_base(
    model: nn.Module,
    batches: Iterable[Batch],
    config: TrainLoopConfig,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.optim.AdamW, list[dict[str, float]]]:
    """Train from random initialization with the documented AdamW defaults."""

    model.to(device)
    optimizer = build_adamw(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device != "cpu" and torch.cuda.is_available(),
    )
    return optimizer, train_updates(model, batches, optimizer, config, device=device)
