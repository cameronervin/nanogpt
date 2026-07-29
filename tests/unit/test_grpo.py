from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from nanogpt.training.grpo import (
    group_advantages,
    grpo_loss,
    grpo_update,
    numeric_exact_reward,
    parse_numeric_answer,
    reward_density,
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
