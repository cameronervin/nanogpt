from __future__ import annotations

import copy
from fractions import Fraction

import pytest
import torch

from nanogpt.config import ModelConfig
from nanogpt.data.sources import arithmetic_prompts
from nanogpt.model import GPT
from nanogpt.rewards import numeric_exact_reward, parse_numeric_answer
from nanogpt.tokenizer import ByteBPETokenizer
from nanogpt.train.grpo import (
    RewardDensity,
    collect_rollout,
    group_advantages,
    grpo_loss,
    grpo_update,
    pilot_gate_error,
    reward_density,
    rollout_log_probs,
    sequence_log_probs,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work\n<final>-1,024</final>", Fraction(-1024)),
        ("<final>0.125</final>", Fraction(1, 8)),
        ("#### 3/4", Fraction(3, 4)),
        ("<final>NaN</final>", None),
        ("<final>1</final><final>2</final>", None),
        ("The answer is 4", None),
        ("<final>__import__('os')</final>", None),
    ],
)
def test_safe_numeric_parser(text: str, expected: Fraction | None) -> None:
    assert parse_numeric_answer(text) == expected


def test_numeric_exact_reward() -> None:
    assert numeric_exact_reward("<final>0.50</final>", "1/2") == 1.0
    assert numeric_exact_reward("<final>4</final>", "5") == 0.0


def test_group_advantages_skip_zero_variance() -> None:
    rewards = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    advantages, valid = group_advantages(rewards)
    assert valid.tolist() == [True, False]
    assert advantages[1].tolist() == [0.0, 0.0]
    assert reward_density(rewards).mixed_group_fraction == 0.5


def test_grpo_objective_matches_identity_ratio_and_freezes_reference() -> None:
    new = torch.zeros(1, 2, 2, requires_grad=True)
    old = torch.zeros_like(new)
    reference = torch.zeros_like(new, requires_grad=True)
    rewards = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(new, dtype=torch.bool)
    result = grpo_loss(new, old, reference, rewards, mask)
    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.kl.item() == pytest.approx(0.0)
    result.loss.backward()
    assert new.grad is not None
    assert reference.grad is None


def test_grpo_all_constant_groups_is_differentiable_skip() -> None:
    new = torch.randn(2, 3, 4, requires_grad=True)
    result = grpo_loss(
        new,
        new.detach(),
        new.detach(),
        torch.ones(2, 3),
        torch.ones_like(new, dtype=torch.bool),
    )
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert new.grad is not None


def test_grpo_update_does_not_apply_weight_decay_to_skipped_groups() -> None:
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 8, bias=False))

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = model

        def forward(self, values: torch.Tensor) -> object:
            logits = self.layers(values)
            return type("Output", (), {"logits": logits})()

    policy = Policy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.1)
    inputs = torch.tensor([[[1, 2], [1, 3]]])
    targets = torch.tensor([[[2, 3], [3, 4]]])
    with torch.no_grad():
        logits = policy(inputs.flatten(0, 1)).logits
        old = logits.log_softmax(-1).gather(-1, targets.flatten(0, 1)[..., None]).squeeze(-1)
        old = old.reshape_as(inputs)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    result = grpo_update(
        model=policy,
        optimizer=optimizer,
        input_ids=inputs,
        target_ids=targets,
        old_log_probs=old,
        reference_log_probs=old,
        rewards=torch.ones(1, 2),
        completion_mask=torch.ones_like(inputs, dtype=torch.bool),
    )
    assert result.valid_group_fraction.item() == 0.0
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(policy.parameters(), before, strict=True)
    )


def test_rollout_masks_only_completion_tokens_and_reference_stays_frozen() -> None:
    torch.manual_seed(0)
    tokenizer = ByteBPETokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        n_layers=1,
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
    )
    policy = GPT(config)
    reference = GPT(config).requires_grad_(False)
    prompts = list(arithmetic_prompts(2))
    rollout = collect_rollout(
        policy,
        tokenizer,
        prompts,
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=None,
        seed=0,
    )
    assert rollout.input_ids.shape[:2] == (2, 3)
    assert rollout.rewards.shape == (2, 3)
    assert rollout.completion_mask.shape == rollout.input_ids.shape
    # Each row supervises between 1 and max_new_tokens completion targets.
    per_row = rollout.completion_mask.sum(-1)
    assert bool((per_row >= 1).all()) and bool((per_row <= 5).all())

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    old = rollout_log_probs(policy, rollout)
    ref = rollout_log_probs(reference, rollout)
    grpo_update(
        model=policy,
        optimizer=optimizer,
        input_ids=rollout.input_ids,
        target_ids=rollout.target_ids,
        old_log_probs=old,
        reference_log_probs=ref,
        rewards=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
        completion_mask=rollout.completion_mask,
    )
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_chunked_grpo_update_matches_single_pass() -> None:
    torch.manual_seed(0)
    tokenizer = ByteBPETokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        n_layers=1,
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
    )
    policy = GPT(config)
    reference = copy.deepcopy(policy).requires_grad_(False)
    rollout = collect_rollout(
        policy,
        tokenizer,
        list(arithmetic_prompts(3)),
        group_size=3,
        max_new_tokens=5,
        temperature=1.0,
        top_k=None,
        seed=0,
    )
    # Group 0 is uniform (skipped); groups 1 and 2 are mixed.
    rewards = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    old = rollout_log_probs(policy, rollout)
    assert torch.allclose(rollout_log_probs(policy, rollout, chunk_prompts=2), old)
    ref = rollout_log_probs(reference, rollout)

    results, states = [], []
    for chunk in (None, 1):
        model = copy.deepcopy(policy)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        results.append(
            grpo_update(
                model=model,
                optimizer=optimizer,
                input_ids=rollout.input_ids,
                target_ids=rollout.target_ids,
                old_log_probs=old,
                reference_log_probs=ref,
                rewards=rewards,
                completion_mask=rollout.completion_mask,
                chunk_prompts=chunk,
            )
        )
        states.append([parameter.detach().clone() for parameter in model.parameters()])
    single, chunked = results
    for field in ("loss", "policy_loss", "kl", "clip_fraction", "valid_group_fraction"):
        assert torch.allclose(getattr(single, field), getattr(chunked, field), atol=1e-6)
    for a, b in zip(*states, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_rollout_log_probs_use_the_sampling_temperature() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 11)
    targets = torch.randint(0, 11, (2, 3))
    cold = sequence_log_probs(logits, targets, temperature=0.5)
    expected = (logits / 0.5).log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    assert torch.allclose(cold, expected)
    assert not torch.allclose(cold, sequence_log_probs(logits, targets))
    with pytest.raises(ValueError, match="temperature"):
        sequence_log_probs(logits, targets, temperature=0.0)


def test_pilot_gate_error_names_saturated_and_sparse_rewards() -> None:
    def check(correct: float, mixed: float) -> str | None:
        return pilot_gate_error(RewardDensity(correct, mixed), min_correct=0.05, min_mixed=0.2)

    assert check(0.4, 0.5) is None
    assert "saturated" in (check(0.994, 0.047) or "")
    assert "too sparse" in (check(0.013, 0.094) or "")
