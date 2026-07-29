"""Feed-forward network for decoder blocks."""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """Bias-free gated MLP using the SiLU activation."""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        if d_model <= 0 or hidden_size <= 0:
            raise ValueError("d_model and hidden_size must be positive")
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return cast(Tensor, self.down_proj(gated))
