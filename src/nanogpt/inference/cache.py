"""Preallocated key/value cache for autoregressive decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nanogpt.config import ModelConfig


@dataclass
class LayerKVCache:
    """Storage for one layer in compact ``[B, H_kv, T, D]`` form."""

    key: Tensor
    value: Tensor

    def update(self, key: Tensor, value: Tensor, start_pos: int) -> tuple[Tensor, Tensor]:
        """Write a time slice and return the populated prefix without copying."""

        if self.key.shape != self.value.shape:
            raise ValueError("cache key and value buffers must have identical shapes")
        if self.key.ndim != 4:
            raise ValueError("cache buffers must have shape [batch, heads, time, head_dim]")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("cache key and value buffers must share device and dtype")
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        incompatible_shape = (
            key.ndim != 4
            or key.shape[:2] != self.key.shape[:2]
            or key.shape[3] != self.key.shape[3]
        )
        if incompatible_shape:
            raise ValueError("key/value shape is incompatible with this cache")
        if key.device != self.key.device or value.device != self.value.device:
            raise ValueError("key/value device is incompatible with this cache")
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            # Silent copy_ conversion would corrupt cache-equivalence guarantees.
            raise ValueError("key/value dtype is incompatible with this cache")
        if isinstance(start_pos, bool) or not isinstance(start_pos, int):
            raise TypeError("start_pos must be an integer")
        stop = start_pos + key.size(2)
        if start_pos < 0 or stop > self.key.size(2):
            raise ValueError("key/value cache capacity exceeded")
        self.key[:, :, start_pos:stop].copy_(key)
        self.value[:, :, start_pos:stop].copy_(value)
        return self.key[:, :, :stop], self.value[:, :, :stop]


class KVCache:
    """A model-wide cache with a shared sequence cursor for every layer."""

    def __init__(self, layers: list[LayerKVCache], max_seq_len: int) -> None:
        if not layers:
            raise ValueError("a KV cache needs at least one layer")
        if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        reference_shape = layers[0].key.shape
        if not layers[0].key.dtype.is_floating_point:
            raise ValueError("KV cache buffers must use a floating-point dtype")
        for layer in layers:
            if layer.key.shape != reference_shape or layer.value.shape != reference_shape:
                raise ValueError("all KV cache layers must have matching key/value shapes")
            if layer.key.ndim != 4 or layer.key.size(2) != max_seq_len:
                raise ValueError("KV cache layer capacity does not match max_seq_len")
            if layer.key.device != layers[0].key.device or layer.value.device != layer.key.device:
                raise ValueError("all KV cache buffers must share one device")
            if layer.key.dtype != layers[0].key.dtype or layer.value.dtype != layer.key.dtype:
                raise ValueError("all KV cache buffers must share one dtype")
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.sequence_length = 0

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> KVCache:
        """Allocate all layer buffers once to avoid per-token concatenation."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        if config.d_model % config.n_query_heads != 0:
            raise ValueError("d_model must be divisible by n_query_heads")
        head_dim = config.d_model // config.n_query_heads
        shape = (batch_size, config.n_kv_heads, config.max_seq_len, head_dim)
        layers = [
            LayerKVCache(
                key=torch.empty(shape, device=device, dtype=dtype),
                value=torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in range(config.n_layers)
        ]
        return cls(layers, config.max_seq_len)

    @property
    def batch_size(self) -> int:
        return self.layers[0].key.size(0)

    def reset(self) -> None:
        """Make the cache reusable; old values are ignored and need not be zeroed."""

        self.sequence_length = 0

    def validate_step(self, *, batch_size: int, token_count: int, n_layers: int) -> None:
        if batch_size <= 0 or token_count <= 0 or n_layers <= 0:
            raise ValueError("batch_size, token_count, and n_layers must be positive")
        if len(self.layers) != n_layers:
            raise ValueError("KV cache layer count does not match the model")
        if self.batch_size != batch_size:
            raise ValueError("KV cache batch size does not match input_ids")
        if self.sequence_length < 0 or self.sequence_length + token_count > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded")
