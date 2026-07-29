"""Normalization layers used by the decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale.

    The variance reduction is performed in FP32 even during mixed-precision
    training. This small cast avoids overflow/underflow in a numerically
    sensitive operation without forcing the surrounding block to FP32.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        # Promote low-precision reductions, but never demote an FP64 caller.
        reduction_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        inverse_rms = torch.rsqrt(reduction_input.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (reduction_input * inverse_rms).to(input_dtype)
        return normalized * self.weight.to(input_dtype)


def rms_normalize_heads(x: Tensor, eps: float) -> Tensor:
    """Apply parameter-free RMS normalization across each attention head."""

    input_dtype = x.dtype
    reduction_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    inverse_rms = torch.rsqrt(reduction_input.square().mean(dim=-1, keepdim=True) + eps)
    return (reduction_input * inverse_rms).to(input_dtype)
