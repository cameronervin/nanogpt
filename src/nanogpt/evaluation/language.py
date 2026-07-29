"""Held-out causal language-model loss, perplexity, and bits per byte."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class EvalBatch(Protocol):
    input_ids: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor | None


@dataclass(frozen=True)
class LanguageMetrics:
    loss: float
    perplexity: float
    bits_per_byte: float | None
    active_tokens: int


@torch.inference_mode()
def evaluate_language_model(
    model: nn.Module,
    batches: Iterable[EvalBatch],
    *,
    device: torch.device | str = "cpu",
    byte_count: int | None = None,
) -> LanguageMetrics:
    """Compute token-weighted held-out metrics without mutating train mode."""

    was_training = model.training
    model.eval()
    loss_sum = 0.0
    active_tokens = 0
    try:
        for batch in batches:
            targets = batch.targets.to(device)
            mask = targets.ne(-100)
            if batch.loss_mask is not None:
                mask &= batch.loss_mask.to(device).bool()
            count = int(mask.sum())
            if count == 0:
                continue
            output = model(
                batch.input_ids.to(device),
                targets=targets,
                loss_mask=None if batch.loss_mask is None else batch.loss_mask.to(device),
            )
            if output.loss is None or not torch.isfinite(output.loss):
                raise FloatingPointError("evaluation loss is missing or non-finite")
            loss_sum += float(output.loss) * count
            active_tokens += count
    finally:
        model.train(was_training)
    if active_tokens == 0:
        raise ValueError("evaluation contained no active target tokens")
    loss = loss_sum / active_tokens
    perplexity = math.exp(loss) if loss < 80 else float("inf")
    bpb = None
    if byte_count is not None:
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        bpb = loss_sum / (math.log(2) * byte_count)
    return LanguageMetrics(loss, perplexity, bpb, active_tokens)
