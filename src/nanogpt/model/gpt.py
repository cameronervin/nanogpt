"""Decoder-only GPT language model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt.config import ModelConfig
from nanogpt.inference.cache import KVCache
from nanogpt.model.block import DecoderBlock
from nanogpt.model.norm import RMSNorm


@dataclass
class ModelOutput:
    """Typed output shared by training and incremental inference."""

    logits: Tensor | None
    loss: Tensor | None
    kv_cache: KVCache | None


class GPT(nn.Module):
    """A compact modern GPT built directly from PyTorch primitives."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # Assignment makes the two modules share one Parameter and its gradient.
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_module)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            typed_block = cast(DecoderBlock, block)
            # Residual branches accumulate with depth, so initialize their
            # output projections more conservatively than other matrices.
            nn.init.normal_(typed_block.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(typed_block.mlp.down_proj.weight, mean=0.0, std=residual_std)

    def _validate_config(self, config: ModelConfig) -> None:
        integer_fields = (
            "vocab_size",
            "max_seq_len",
            "n_layers",
            "d_model",
            "n_query_heads",
            "n_kv_heads",
            "ffn_hidden_size",
        )
        if any(getattr(config, name) <= 0 for name in integer_fields):
            raise ValueError("all model dimensions must be positive")
        if config.d_model % config.n_query_heads != 0:
            raise ValueError("d_model must be divisible by n_query_heads")
        if config.n_query_heads % config.n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if (config.d_model // config.n_query_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        """Compute aligned next-token logits/loss and optionally update a cache.

        ``targets[b, t]`` is the desired token for ``input_ids[b, t]``; callers
        perform the one-token shift while packing batches. ``loss_mask`` can
        additionally suppress prompt/padding positions during SFT.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token IDs")
        batch, time = input_ids.shape
        if time <= 0:
            raise ValueError("input_ids must contain at least one token")
        if loss_mask is not None and targets is None:
            raise ValueError("loss_mask requires targets")

        start_pos = 0
        if kv_cache is not None:
            kv_cache.validate_step(batch_size=batch, token_count=time, n_layers=len(self.blocks))
            start_pos = kv_cache.sequence_length
        elif time > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache.layers[layer_index]
            x = block(x, start_pos=start_pos, kv_cache=layer_cache)
        x = self.final_norm(x)
        # FP32 logits keep cross-entropy stable under BF16 autocast.
        logits = self.lm_head(x).float()

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            if targets.dtype not in (torch.int32, torch.int64):
                raise TypeError("targets must contain integer token IDs")
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=-100,
                reduction="none",
            ).view_as(targets)
            active = targets.ne(-100)
            weights = active.to(token_losses.dtype)
            if loss_mask is not None:
                if loss_mask.shape != targets.shape:
                    raise ValueError("loss_mask must have the same shape as targets")
                if loss_mask.dtype == torch.bool:
                    valid_mask = torch.ones_like(loss_mask)
                elif loss_mask.is_floating_point() or loss_mask.dtype in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ):
                    valid_mask = torch.isfinite(loss_mask) & ((loss_mask == 0) | (loss_mask == 1))
                else:
                    raise TypeError("loss_mask must be boolean or contain numeric zeros and ones")
                if not bool(valid_mask.all()):
                    raise ValueError("loss_mask values must be finite zeros or ones")
                weights = weights * loss_mask.to(device=weights.device, dtype=weights.dtype)
            denominator = weights.sum()
            # An all-masked microbatch contributes a differentiable zero.
            loss = (token_losses * weights).sum() / denominator.clamp_min(1.0)

        if kv_cache is not None:
            # Advance exactly once after every layer has written the same slice.
            kv_cache.sequence_length += time
        return ModelOutput(logits=logits, loss=loss, kv_cache=kv_cache)

    def parameter_count(self, *, exclude_embeddings: bool = False) -> int:
        """Return the number of unique trainable parameters."""

        count = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            count -= self.token_embedding.weight.numel()
        return count
