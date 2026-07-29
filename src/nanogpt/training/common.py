"""Shared optimizer, schedule, batching, and update primitives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class Batch(Protocol):
    @property
    def input_ids(self) -> Tensor: ...

    @property
    def targets(self) -> Tensor: ...

    @property
    def loss_mask(self) -> Tensor | None: ...


@dataclass(frozen=True)
class CausalBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor | None = None


@dataclass(frozen=True)
class TrainLoopConfig:
    max_steps: int
    learning_rate: float
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.learning_rate <= 0 or self.accumulation_steps <= 0:
            raise ValueError("steps, learning rate, and accumulation must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be between zero and max_steps")
        if not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("invalid schedule, decay, or clipping value")


def causal_batches(tokens: Tensor, *, batch_size: int, seq_len: int) -> Iterator[CausalBatch]:
    """Yield packed next-token batches from one contiguous token stream."""

    if tokens.ndim != 1 or batch_size <= 0 or seq_len <= 0:
        raise ValueError("tokens must be 1D and batch/sequence sizes positive")
    stride = batch_size * seq_len
    for offset in range(0, tokens.numel() - stride, stride):
        window = tokens[offset : offset + stride + 1]
        yield CausalBatch(
            window[:-1].view(batch_size, seq_len),
            window[1:].view(batch_size, seq_len),
        )


def build_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    """Exclude vectors (norm scales) from weight decay; decay matrix weights."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs: dict[str, object] = {}
    if fused is not None:
        kwargs["fused"] = fused
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        **kwargs,  # type: ignore[arg-type]
    )


def cosine_lr(step: int, config: TrainLoopConfig) -> float:
    if step < config.warmup_steps and config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
    coefficient = config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )
    return config.learning_rate * coefficient


def train_updates(
    model: nn.Module,
    batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    config: TrainLoopConfig,
    *,
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    """Run finite guarded updates; callers own checkpoint/evaluation cadence."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics: list[dict[str, float]] = []
    micro_step = 0
    for batch in batches:
        input_ids = batch.input_ids.to(device)
        targets = batch.targets.to(device)
        loss_mask = None if batch.loss_mask is None else batch.loss_mask.to(device)
        output = model(input_ids, targets=targets, loss_mask=loss_mask)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("training loss is missing or non-finite")
        (loss / config.accumulation_steps).backward()
        micro_step += 1
        if micro_step % config.accumulation_steps:
            continue
        update = micro_step // config.accumulation_steps - 1
        learning_rate = cosine_lr(update, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("gradient norm is non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.append(
            {
                "step": float(update + 1),
                "loss": float(loss.detach()),
                "lr": learning_rate,
                "grad_norm": float(grad_norm),
            }
        )
        if update + 1 >= config.max_steps:
            break
    return metrics
