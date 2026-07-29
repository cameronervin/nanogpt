"""Tests for deterministic sampling and incremental generation."""

from __future__ import annotations

import pytest
import torch

from nanogpt.config import ModelConfig
from nanogpt.inference import generate, sample_next_token
from nanogpt.model import GPT


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(11)
    config = ModelConfig(
        vocab_size=29,
        max_seq_len=10,
        n_layers=2,
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
    )
    return GPT(config)


def test_sample_next_token_greedy_and_top_k() -> None:
    logits = torch.tensor([[0.0, 10.0, 1.0], [5.0, 2.0, 3.0]])

    assert sample_next_token(logits, temperature=0).tolist() == [1, 0]
    assert sample_next_token(logits, top_k=1).tolist() == [1, 0]


def test_seeded_sampling_is_deterministic(model: GPT) -> None:
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])

    first = generate(model, prompt, max_new_tokens=5, temperature=0.8, top_k=10, seed=7)
    second = generate(model, prompt, max_new_tokens=5, temperature=0.8, top_k=10, seed=7)

    assert torch.equal(first, second)
    assert model.training  # generate restores the caller's previous mode.


def test_cached_and_uncached_greedy_generation_match(model: GPT) -> None:
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])

    cached = generate(model, prompt, max_new_tokens=5, temperature=0, use_cache=True)
    uncached = generate(model, prompt, max_new_tokens=5, temperature=0, use_cache=False)

    assert torch.equal(cached, uncached)


def test_cached_generation_uses_active_autocast_dtype(model: GPT) -> None:
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        cached = generate(model, prompt, max_new_tokens=3, temperature=0, use_cache=True)
        uncached = generate(model, prompt, max_new_tokens=3, temperature=0, use_cache=False)

    assert torch.equal(cached, uncached)


def test_generation_respects_context_capacity(model: GPT) -> None:
    prompt = torch.ones((1, 8), dtype=torch.long)

    output = generate(model, prompt, max_new_tokens=99, temperature=0)

    assert output.shape == (1, model.config.max_seq_len)


def test_generation_rejects_overlong_prompt(model: GPT) -> None:
    prompt = torch.ones((1, model.config.max_seq_len + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="prompt exceeds"):
        generate(model, prompt, max_new_tokens=1)


def test_generation_stops_the_entire_batch_after_eos(model: GPT) -> None:
    prompt = torch.tensor([[1, 2], [3, 4]])
    with torch.no_grad():
        model.token_embedding.weight.zero_()

    output = generate(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0,
        eos_token_id=0,
    )

    assert output.tolist() == [[1, 2, 0], [3, 4, 0]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_new_tokens": 1.5}, "non-negative integer"),
        ({"max_new_tokens": 1, "temperature": float("nan")}, "finite non-negative"),
        ({"max_new_tokens": 1, "top_k": 1.5}, "positive integer"),
        ({"max_new_tokens": 1, "seed": -1}, "non-negative integer"),
        ({"max_new_tokens": 1, "eos_token_id": True}, "must be an integer"),
    ],
)
def test_generation_rejects_invalid_sampling_contract(
    model: GPT, kwargs: dict[str, object], message: str
) -> None:
    prompt = torch.tensor([[1, 2]])
    with pytest.raises((TypeError, ValueError), match=message):
        generate(model, prompt, **kwargs)  # type: ignore[arg-type]


def test_generation_validates_sampling_contract_even_when_no_tokens_requested(
    model: GPT,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        generate(model, torch.tensor([[1]]), max_new_tokens=0, top_k=0)
