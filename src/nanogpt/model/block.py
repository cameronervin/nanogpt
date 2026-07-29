"""Pre-normalized decoder block."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from nanogpt.config import ModelConfig
from nanogpt.inference.cache import LayerKVCache
from nanogpt.model.attention import GroupedQueryAttention
from nanogpt.model.mlp import SwiGLU
from nanogpt.model.norm import RMSNorm


class DecoderBlock(nn.Module):
    """One residual attention/MLP block using pre-RMSNorm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            n_query_heads=config.n_query_heads,
            n_kv_heads=config.n_kv_heads,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            norm_eps=config.rms_norm_eps,
            qk_norm=config.qk_norm,
        )
        self.mlp_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.mlp = SwiGLU(config.d_model, config.ffn_hidden_size)

    def forward(
        self,
        x: Tensor,
        *,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> Tensor:
        x = x + self.attention(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache)
        return cast(Tensor, x + self.mlp(self.mlp_norm(x)))
