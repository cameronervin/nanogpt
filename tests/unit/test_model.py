"""Focused correctness tests for the decoder architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nanogpt.config import ModelConfig
from nanogpt.model import GPT, GroupedQueryAttention, KVCache, RMSNorm, RotaryEmbedding


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=37,
        max_seq_len=12,
        n_layers=2,
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
    )


def test_rms_norm_normalizes_last_dimension() -> None:
    norm = RMSNorm(8, eps=1e-6)
    x = torch.randn(2, 3, 8)

    output = norm(x)

    assert output.shape == x.shape
    assert torch.allclose(output.square().mean(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_rms_norm_preserves_float64_reduction_precision() -> None:
    norm = RMSNorm(2, eps=1e-12).double()
    x = torch.tensor([[1.0e8, 1.0]], dtype=torch.float64)

    output = norm(x)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)

    assert output.dtype == torch.float64
    torch.testing.assert_close(output, expected, rtol=1e-14, atol=0)


def test_rope_zero_position_is_identity() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    q = torch.randn(2, 4, 1, 8)
    k = torch.randn(2, 2, 1, 8)

    rotated_q, rotated_k = rope(q, k)

    assert torch.equal(rotated_q, q)
    assert torch.equal(rotated_k, k)


def test_rope_applies_the_documented_adjacent_pair_rotation() -> None:
    rope = RotaryEmbedding(head_dim=4, max_seq_len=3, theta=10_000.0)
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    k = q.clone()

    rotated_q, _ = rope(q, k, start_pos=1)
    angles = torch.tensor([1.0, 0.01])
    expected = torch.tensor(
        [
            [
                [
                    [
                        1.0 * angles[0].cos() - 2.0 * angles[0].sin(),
                        2.0 * angles[0].cos() + 1.0 * angles[0].sin(),
                        3.0 * angles[1].cos() - 4.0 * angles[1].sin(),
                        4.0 * angles[1].cos() + 3.0 * angles[1].sin(),
                    ]
                ]
            ]
        ]
    )

    torch.testing.assert_close(rotated_q, expected)


def test_forward_shapes_weight_tying_and_ignored_targets(model_config: ModelConfig) -> None:
    torch.manual_seed(1)
    model = GPT(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (2, 6))
    targets = torch.randint(0, model_config.vocab_size, (2, 6))
    targets[:, :2] = -100

    output = model(input_ids, targets=targets)

    assert output.logits.shape == (2, 6, model_config.vocab_size)
    assert output.logits.dtype == torch.float32
    assert output.loss is not None and output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert model.lm_head.weight is model.token_embedding.weight


def test_bias_free_modules_and_scaled_residual_initialization(
    model_config: ModelConfig,
) -> None:
    torch.manual_seed(13)
    model = GPT(model_config)
    linear_modules = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert all(module.bias is None for module in linear_modules)

    ordinary = torch.cat([block.attention.q_proj.weight.flatten() for block in model.blocks])
    residual = torch.cat([block.attention.o_proj.weight.flatten() for block in model.blocks])
    assert ordinary.std().item() == pytest.approx(0.02, rel=0.1)
    expected_residual_std = 0.02 / (2 * model_config.n_layers) ** 0.5
    assert residual.std().item() == pytest.approx(expected_residual_std, rel=0.1)


def test_all_masked_loss_is_differentiable_zero(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (2, 4))
    targets = torch.full_like(input_ids, -100)

    loss = model(input_ids, targets=targets).loss

    assert loss is not None and loss.item() == 0.0
    loss.backward()


def test_loss_is_mean_over_only_nonignored_tokens(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    input_ids = torch.randint(0, model_config.vocab_size, (1, 4))
    targets = torch.tensor([[2, -100, -100, 4]])

    output = model(input_ids, targets=targets)
    assert output.loss is not None
    per_token = F.cross_entropy(
        output.logits.view(-1, model_config.vocab_size),
        targets.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    expected = (per_token[0, 0] + per_token[0, 3]) / 2
    torch.testing.assert_close(output.loss, expected)


def test_targets_must_match_input_shape(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    tokens = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="same shape"):
        model(tokens, targets=tokens[:, :1])


def test_logits_are_causal(model_config: ModelConfig) -> None:
    torch.manual_seed(2)
    model = GPT(model_config).eval()
    first = torch.randint(0, model_config.vocab_size, (1, 7))
    second = first.clone()
    second[:, 4:] = torch.randint(0, model_config.vocab_size, (1, 3))

    first_logits = model(first).logits
    second_logits = model(second).logits

    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6, rtol=1e-5)


def test_cached_multi_token_logits_match_uncached(model_config: ModelConfig) -> None:
    """Exercise the offset causal mask, not just the single-token all-true case."""

    torch.manual_seed(3)
    model = GPT(model_config).eval()
    tokens = torch.randint(0, model_config.vocab_size, (2, 8))
    full_logits = model(tokens).logits
    cache = KVCache.allocate(model_config, 2, device="cpu", dtype=torch.float32)

    prefix = model(tokens[:, :3], kv_cache=cache)
    suffix = model(tokens[:, 3:], kv_cache=cache)

    cached_logits = torch.cat((prefix.logits, suffix.logits), dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-4)
    assert cache.sequence_length == tokens.size(1)
    assert cache.layers[0].key.shape == (
        2,
        model_config.n_kv_heads,
        model_config.max_seq_len,
        model_config.head_dim,
    )


def test_offset_attention_without_cache_remains_causal() -> None:
    torch.manual_seed(4)
    attention = GroupedQueryAttention(
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        max_seq_len=12,
        rope_theta=10_000.0,
        norm_eps=1e-6,
    ).eval()
    first = torch.randn(1, 5, 32)
    second = first.clone()
    second[:, 3:] = torch.randn(1, 2, 32)

    first_output = attention(first, start_pos=4)
    second_output = attention(second, start_pos=4)

    torch.testing.assert_close(first_output[:, :3], second_output[:, :3])


def test_native_gqa_failure_falls_back_to_repeated_kv_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5)
    attention = GroupedQueryAttention(
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        rope_theta=10_000.0,
        norm_eps=1e-6,
    ).eval()
    x = torch.randn(2, 6, 32)
    expected = attention(x)
    real_sdpa = F.scaled_dot_product_attention

    def reject_native_gqa(*args: object, **kwargs: object) -> torch.Tensor:
        if kwargs.get("enable_gqa"):
            raise TypeError("simulated backend without native GQA")
        return real_sdpa(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(F, "scaled_dot_product_attention", reject_native_gqa)
    actual = attention(x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_cache_rejects_overflow(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    cache = KVCache.allocate(model_config, 1, device="cpu", dtype=torch.float32)
    tokens = torch.zeros((1, model_config.max_seq_len), dtype=torch.long)
    model(tokens, kv_cache=cache)

    with pytest.raises(ValueError, match="capacity"):
        model(tokens[:, :1], kv_cache=cache)


def test_cache_rejects_silent_dtype_conversion(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    cache = KVCache.allocate(model_config, 1, device="cpu", dtype=torch.float16)
    tokens = torch.ones((1, 1), dtype=torch.long)

    with pytest.raises(ValueError, match="dtype is incompatible"):
        model(tokens, kv_cache=cache)


def test_cache_allocation_requires_floating_point_dtype(model_config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="floating point"):
        KVCache.allocate(model_config, 1, device="cpu", dtype=torch.int64)
