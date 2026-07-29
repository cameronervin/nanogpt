"""Grouped-query causal self-attention."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt.inference.cache import LayerKVCache
from nanogpt.model.norm import rms_normalize_heads
from nanogpt.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Causal attention with fewer KV heads than query heads.

    PyTorch SDPA selects the best available fused kernel. MPS does not support
    SDPA's native ``enable_gqa`` on all supported PyTorch versions, so that
    backend receives a compatibility view with repeated KV heads. The cache
    itself always remains in the smaller KV-head representation.
    """

    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_theta: float,
        norm_eps: float,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_model % n_query_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_query_heads")
        if n_kv_heads <= 0 or n_query_heads % n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")

        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads
        self.groups = n_query_heads // n_kv_heads
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm

        self.q_proj = nn.Linear(d_model, n_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_query_heads * self.head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    def _shape_heads(self, x: Tensor, heads: int) -> Tensor:
        batch, time, _ = x.shape
        return x.view(batch, time, heads, self.head_dim).transpose(1, 2)

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        start_pos: int,
        *,
        key_start_pos: int,
    ) -> Tensor:
        query_len, key_len = q.size(2), k.size(2)
        use_native_gqa = self.groups > 1 and q.device.type != "mps"
        if self.groups > 1 and not use_native_gqa:
            # Repeat only the temporary attention operands, never cached tensors.
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if start_pos == key_start_pos and query_len == key_len:
            attention_mask = None
            is_causal = True
        else:
            # SDPA's default rectangular causal mask is upper-left aligned. A
            # cached query is offset by start_pos, so construct the correct mask.
            query_positions = start_pos + torch.arange(query_len, device=q.device)
            key_positions = key_start_pos + torch.arange(key_len, device=q.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            is_causal = False

        kwargs: dict[str, object] = {
            "attn_mask": attention_mask,
            "dropout_p": 0.0,
            "is_causal": is_causal,
        }
        if use_native_gqa:
            kwargs["enable_gqa"] = True
        try:
            return F.scaled_dot_product_attention(q, k, v, **kwargs)  # type: ignore[arg-type]
        except (RuntimeError, TypeError):
            if not use_native_gqa:
                raise
            # Older PyTorch builds or a backend without native GQA support.
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            kwargs.pop("enable_gqa")
            return F.scaled_dot_product_attention(q, k, v, **kwargs)  # type: ignore[arg-type]

    def forward(
        self,
        x: Tensor,
        *,
        start_pos: int = 0,
        kv_cache: LayerKVCache | None = None,
    ) -> Tensor:
        batch, time, width = x.shape
        q = self._shape_heads(self.q_proj(x), self.n_query_heads)
        k = self._shape_heads(self.k_proj(x), self.n_kv_heads)
        v = self._shape_heads(self.v_proj(x), self.n_kv_heads)

        if self.qk_norm:
            q = rms_normalize_heads(q, self.norm_eps)
            k = rms_normalize_heads(k, self.norm_eps)
        q, k = self.rope(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)

        # Cached K/V always begin at absolute position zero. Without a cache,
        # callers may apply RoPE at an offset but the local keys start there too.
        key_start_pos = 0 if kv_cache is not None else start_pos
        attended = self._attention(q, k, v, start_pos, key_start_pos=key_start_pos)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        return cast(Tensor, self.o_proj(attended))
