"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Precompute and apply rotary position embeddings to query/key heads.

    The cache is non-persistent because it is fully determined by the model
    configuration. Excluding it from checkpoints keeps weights portable.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")

        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        # Adjacent features form complex-number pairs: (x[0], x[1]), ...
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", angles.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer("sin", angles.sin().repeat_interleave(2, dim=-1), persistent=False)
        self.max_seq_len = max_seq_len

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)

    def forward(self, q: Tensor, k: Tensor, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        """Rotate ``[batch, heads, time, head_dim]`` query and key tensors."""

        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must have shape [batch, heads, time, head_dim]")
        if q.shape[-2:] != k.shape[-2:]:
            raise ValueError("q and k must have matching time and head dimensions")
        if q.size(-1) != self.cos.size(-1):
            raise ValueError("q and k head dimensions do not match this RoPE instance")
        if q.size(0) != k.size(0):
            raise ValueError("q and k must have matching batch dimensions")
        if q.device != k.device or q.dtype != k.dtype:
            raise ValueError("q and k must have matching devices and dtypes")
        if isinstance(start_pos, bool) or not isinstance(start_pos, int):
            raise TypeError("start_pos must be an integer")
        if start_pos < 0 or start_pos + q.size(-2) > self.max_seq_len:
            raise ValueError("requested RoPE positions exceed max_seq_len")

        stop = start_pos + q.size(-2)
        # [time, head_dim] broadcasts over batch and heads.
        cos = self.cos[start_pos:stop].to(device=q.device, dtype=q.dtype)
        sin = self.sin[start_pos:stop].to(device=q.device, dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin
