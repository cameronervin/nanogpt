"""Decoder-only GPT language model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn

from nanogpt.config import ModelConfig
from nanogpt.model.kv_cache import KVCache, LayerKVCache
from nanogpt.model.layers import GroupedQueryAttention, RMSNorm, SwiGLU

IGNORE_INDEX = -100


@dataclass
class ModelOutput:
    logits: Tensor
    loss: Tensor | None


class DecoderBlock(nn.Module):
    """One residual block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``."""

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
        self, x: Tensor, *, start_pos: int = 0, kv_cache: LayerKVCache | None = None
    ) -> Tensor:
        x = x + self.attention(self.attn_norm(x), start_pos=start_pos, kv_cache=kv_cache)
        return cast(Tensor, x + self.mlp(self.mlp_norm(x)))


class GPT(nn.Module):
    """A compact modern GPT built directly from PyTorch primitives."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
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
            typed = cast(DecoderBlock, block)
            # Residual branches accumulate with depth, so initialize their
            # output projections more conservatively than other matrices.
            nn.init.normal_(typed.attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(typed.mlp.down_proj.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        """Compute next-token logits and, given targets, the mean token loss.

        ``targets[b, t]`` is the desired token after ``input_ids[b, t]``; callers
        do the one-token shift when building batches. Positions whose target is
        ``-100`` (prompt, padding) are ignored, so the loss is a mean over
        supervised tokens only. A fully ignored batch yields a differentiable 0.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, time = input_ids.shape
        start_pos = 0
        if kv_cache is not None:
            kv_cache.validate_step(batch_size=batch, token_count=time, n_layers=len(self.blocks))
            start_pos = kv_cache.sequence_length
        elif time > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")

        x = self.token_embedding(input_ids)
        for index, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache.layers[index]
            x = block(x, start_pos=start_pos, kv_cache=layer_cache)
        # FP32 logits keep cross-entropy stable under BF16 autocast.
        logits = self.lm_head(self.final_norm(x)).float()

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1).long(),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            # Dividing by clamp(count, 1) rather than using reduction="mean"
            # keeps an all-ignored micro-batch a finite zero instead of NaN.
            count = targets.ne(IGNORE_INDEX).sum().clamp_min(1)
            loss = token_losses / count

        if kv_cache is not None:
            # Advance exactly once after every layer has written the same slice.
            kv_cache.sequence_length += time
        return ModelOutput(logits=logits, loss=loss)

    def parameter_count(self, *, exclude_embeddings: bool = False) -> int:
        """Return the number of unique trainable parameters."""

        count = sum(parameter.numel() for parameter in self.parameters())
        if exclude_embeddings:
            count -= self.token_embedding.weight.numel()
        return count
