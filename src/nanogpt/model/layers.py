"""Transformer building blocks: RMSNorm, RoPE, grouped-query attention, SwiGLU.

Architecture invariants (divisibility, even head width, ...) are validated once
by ``ModelConfig``; these modules assume they receive consistent dimensions.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt.model.kv_cache import LayerKVCache


def _rms_normalize(x: Tensor, eps: float) -> Tensor:
    # Promote low-precision reductions to FP32, but never demote an FP64 caller.
    reduction_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    inverse_rms = torch.rsqrt(reduction_input.square().mean(dim=-1, keepdim=True) + eps)
    return (reduction_input * inverse_rms).to(x.dtype)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale.

    The variance reduction runs in FP32 even under mixed precision, avoiding
    overflow/underflow without forcing the surrounding block to FP32.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return _rms_normalize(x, self.eps) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Precompute and apply rotary position embeddings to query/key heads.

    The sin/cos tables are non-persistent buffers: they are fully determined by
    the configuration, so excluding them keeps checkpoints portable.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
        # Adjacent features form complex-number pairs: (x[0], x[1]), (x[2], x[3]), ...
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

        stop = start_pos + q.size(-2)
        if start_pos < 0 or stop > self.max_seq_len:
            raise ValueError("requested RoPE positions exceed max_seq_len")
        # [time, head_dim] broadcasts over batch and heads.
        cos = self.cos[start_pos:stop].to(dtype=q.dtype)
        sin = self.sin[start_pos:stop].to(dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class GroupedQueryAttention(nn.Module):
    """Causal attention with fewer KV heads than query heads.

    PyTorch SDPA selects the best fused kernel. MPS does not support SDPA's
    native ``enable_gqa`` on every supported PyTorch version, so that backend
    receives repeated KV heads. The cache always stores the compact KV heads.
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

    def _attention(self, q: Tensor, k: Tensor, v: Tensor, start_pos: int, key_start: int) -> Tensor:
        query_len, key_len = q.size(2), k.size(2)
        use_native_gqa = self.groups > 1 and q.device.type != "mps"
        if self.groups > 1 and not use_native_gqa:
            # Repeat only the temporary attention operands, never cached tensors.
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if start_pos == key_start and query_len == key_len:
            attention_mask, is_causal = None, True
        else:
            # SDPA's built-in rectangular causal mask is upper-left aligned. A
            # cached query is offset by start_pos, so build the correct mask.
            query_positions = start_pos + torch.arange(query_len, device=q.device)
            key_positions = key_start + torch.arange(key_len, device=q.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            is_causal = False

        kwargs: dict[str, object] = {"attn_mask": attention_mask, "is_causal": is_causal}
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
        self, x: Tensor, *, start_pos: int = 0, kv_cache: LayerKVCache | None = None
    ) -> Tensor:
        batch, time, width = x.shape
        q = self._shape_heads(self.q_proj(x), self.n_query_heads)
        k = self._shape_heads(self.k_proj(x), self.n_kv_heads)
        v = self._shape_heads(self.v_proj(x), self.n_kv_heads)

        if self.qk_norm:
            # Parameter-free per-head RMS normalization bounds attention logits.
            q = _rms_normalize(q, self.norm_eps)
            k = _rms_normalize(k, self.norm_eps)
        q, k = self.rope(q, k, start_pos=start_pos)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)

        # Cached K/V always begin at absolute position zero. Without a cache,
        # callers may apply RoPE at an offset, and the local keys start there too.
        key_start = 0 if kv_cache is not None else start_pos
        attended = self._attention(q, k, v, start_pos, key_start)
        attended = attended.transpose(1, 2).contiguous().view(batch, time, width)
        return cast(Tensor, self.o_proj(attended))


class SwiGLU(nn.Module):
    """Bias-free gated MLP: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
