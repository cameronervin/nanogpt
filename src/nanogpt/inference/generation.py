"""Autoregressive token generation utilities."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from nanogpt.inference.cache import KVCache
from nanogpt.model.gpt import GPT


def _validate_sampling_options(temperature: float, top_k: int | None) -> None:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a finite non-negative number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
        raise ValueError("top_k must be a positive integer")


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per row, or take argmax when temperature is zero."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if logits.size(0) == 0 or logits.size(1) == 0:
        raise ValueError("logits must have nonzero batch and vocabulary dimensions")
    _validate_sampling_options(temperature, top_k)
    if temperature == 0:
        return logits.argmax(dim=-1)

    scaled = logits.float() / temperature
    if top_k is not None:
        k = min(top_k, scaled.size(-1))
        cutoff = torch.topk(scaled, k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


@contextmanager
def _evaluation_mode(model: GPT) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    # CUDA supports device-local generators. CPU logits are used below for MPS
    # so seeded sampling remains available on an M1 development machine.
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def _active_compute_dtype(device: torch.device, fallback: torch.dtype) -> torch.dtype:
    """Choose the projection dtype, including an enclosing autocast context."""

    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return fallback


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> Tensor:
    """Generate continuations for a batch of equal-length tokenized prompts.

    The output contains the original prompts followed by generated tokens. If
    some rows finish early, their remaining batch positions stay at EOS until
    every row finishes. Variable-length text prompts should be bucketed by
    tokenized length before calling this low-level function.
    """

    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape [batch, nonzero time]")
    if input_ids.size(0) == 0:
        raise ValueError("input_ids batch dimension cannot be empty")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer")
    if input_ids.size(1) > model.config.max_seq_len:
        raise ValueError("prompt exceeds max_seq_len")
    if eos_token_id is not None:
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise ValueError("eos_token_id must be an integer")
        if not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    _validate_sampling_options(temperature, top_k)
    if input_ids.device != model.token_embedding.weight.device:
        raise ValueError("input_ids and model must be on the same device")

    # Context capacity is a hard bound for both cached and uncached decoding.
    steps = min(max_new_tokens, model.config.max_seq_len - input_ids.size(1))
    if steps == 0:
        return input_ids.clone()

    output_ids = input_ids.clone()
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    sampling_device = input_ids.device if input_ids.device.type == "cuda" else torch.device("cpu")
    generator = _make_generator(input_ids.device, seed)

    with _evaluation_mode(model):
        cache = None
        if use_cache:
            cache = KVCache.allocate(
                model.config,
                input_ids.size(0),
                device=input_ids.device,
                dtype=_active_compute_dtype(
                    input_ids.device,
                    model.token_embedding.weight.dtype,
                ),
            )
        logits = model(input_ids, kv_cache=cache).logits
        assert logits is not None

        for step in range(steps):
            next_logits = logits[:, -1]
            # On MPS, seeded multinomial is performed on CPU for generator support.
            sampled = sample_next_token(
                next_logits.to(sampling_device),
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            ).to(input_ids.device)
            if eos_token_id is not None:
                sampled = torch.where(finished, eos_token_id, sampled)
                finished |= sampled.eq(eos_token_id)
            output_ids = torch.cat((output_ids, sampled[:, None]), dim=1)
            if finished.all() or step + 1 == steps:
                break
            if use_cache:
                logits = model(sampled[:, None], kv_cache=cache).logits
            else:
                logits = model(output_ids).logits
            assert logits is not None

    return output_ids
