"""Verifiable numeric rewards and the critic-free GRPO objective."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch
from torch import Tensor

_FINAL_XML = re.compile(r"<final>\s*([^<>]+?)\s*</final>", re.IGNORECASE)
_FINAL_HASH = re.compile(r"(?:^|\n)\s*####\s*([^\n]+?)\s*$")
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?")


def parse_numeric_answer(text: str) -> Fraction | None:
    """Parse one unambiguous, finite final answer without evaluating code.

    Accepted forms are signed integers, conventional comma grouping, finite
    decimals, and fractions.  Exactly one ``<final>...</final>`` or GSM8K-style
    ``####`` marker is required; malformed or multiple markers fail closed.
    """

    matches = _FINAL_XML.findall(text) + _FINAL_HASH.findall(text)
    if len(matches) != 1:
        return None
    candidate = matches[0].strip()
    if _NUMBER.fullmatch(candidate) is None:
        return None
    normalized = candidate.replace(",", "")
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            value = Fraction(int(numerator), int(denominator))
        elif "." in normalized:
            decimal = Decimal(normalized)
            if not decimal.is_finite():
                return None
            value = Fraction(decimal)
        else:
            value = Fraction(int(normalized), 1)
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None
    return value


def numeric_exact_reward(completion: str, answer: object) -> float:
    """Return one only when parsed completion and ground truth are equal."""

    parsed = parse_numeric_answer(completion)
    target_text = str(answer).strip()
    # Ground truth is data, not model output, but pass it through the same safe
    # grammar by adding the required marker.
    target = parse_numeric_answer(f"<final>{target_text}</final>")
    return float(parsed is not None and target is not None and parsed == target)


def group_advantages(rewards: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize per-prompt rewards and identify mixed-reward groups.

    ``rewards`` has shape ``[prompts, completions]``. Zero-variance rows are
    returned as zeros and excluded by ``valid_groups`` rather than amplified by
    an arbitrary epsilon.
    """

    if rewards.ndim != 2 or rewards.size(1) < 2:
        raise ValueError("rewards must have shape [prompts, at least two completions]")
    values = rewards.float()
    means = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, unbiased=False, keepdim=True)
    valid = std.squeeze(1) > 0
    advantages = torch.zeros_like(values)
    advantages[valid] = (values[valid] - means[valid]) / std[valid]
    return advantages, valid


@dataclass(frozen=True)
class GRPOLoss:
    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor
    valid_group_fraction: Tensor


def grpo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    rewards: Tensor,
    completion_mask: Tensor,
    *,
    epsilon: float = 0.2,
    beta: float = 0.02,
) -> GRPOLoss:
    """Compute clipped group-relative policy loss plus reference KL.

    Log-probability tensors and ``completion_mask`` have shape
    ``[prompts, completions, tokens]``; rewards omit the token dimension.
    The reference tensor must be detached by rollout collection.
    """

    if (
        new_log_probs.shape != old_log_probs.shape
        or new_log_probs.shape != reference_log_probs.shape
    ):
        raise ValueError("policy/reference log probabilities must have identical shapes")
    if completion_mask.shape != new_log_probs.shape or rewards.shape != new_log_probs.shape[:2]:
        raise ValueError("completion mask or reward shape is incompatible")
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError("epsilon must be in (0,1) and beta non-negative")
    tensors = (new_log_probs, old_log_probs, reference_log_probs, rewards)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError("GRPO inputs contain non-finite values")

    advantages, valid_groups = group_advantages(rewards)
    valid_tokens = completion_mask.bool() & valid_groups[:, None, None]
    token_count = valid_tokens.sum()
    if token_count == 0:
        # Preserve a differentiable connection to the policy for callers that
        # uniformly handle skipped and non-skipped rollout batches.
        zero = new_log_probs.sum() * 0
        return GRPOLoss(zero, zero, zero.detach(), zero.detach(), valid_groups.float().mean())

    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    advantage = advantages[:, :, None]
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    policy_tokens = -torch.minimum(unclipped, clipped)

    # This non-negative estimator is zero when policy and reference match and
    # is the same stable per-token approximation commonly used by GRPO.
    ref_delta = reference_log_probs.detach() - new_log_probs
    kl_tokens = ref_delta.exp() - ref_delta - 1
    weights = valid_tokens.to(new_log_probs.dtype)
    policy_loss = (policy_tokens * weights).sum() / token_count
    kl = (kl_tokens * weights).sum() / token_count
    clip_fraction = (((ratio - 1).abs() > epsilon) & valid_tokens).sum() / token_count
    loss = policy_loss + beta * kl
    if not torch.isfinite(loss):
        raise FloatingPointError("GRPO objective became non-finite")
    return GRPOLoss(loss, policy_loss, kl, clip_fraction, valid_groups.float().mean())


@dataclass(frozen=True)
class RewardDensity:
    correct_fraction: float
    mixed_group_fraction: float

    @property
    def passes(self) -> bool:
        return self.correct_fraction >= 0.05 and self.mixed_group_fraction >= 0.20


def reward_density(rewards: Tensor) -> RewardDensity:
    _, valid = group_advantages(rewards)
    return RewardDensity(float(rewards.float().mean()), float(valid.float().mean()))


def score_completions(completions: Iterable[str], answers: Iterable[object]) -> Tensor:
    values = [
        numeric_exact_reward(completion, answer)
        for completion, answer in zip(completions, answers, strict=True)
    ]
    return torch.tensor(values, dtype=torch.float32)


def sequence_log_probs(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Gather target token log probabilities from ``[..., vocab]`` logits."""

    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("target IDs must match all non-vocabulary dimensions")
    return logits.float().log_softmax(dim=-1).gather(-1, target_ids[..., None]).squeeze(-1)


def grpo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    target_ids: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    rewards: Tensor,
    completion_mask: Tensor,
    epsilon: float = 0.2,
    beta: float = 0.02,
    grad_clip: float = 1.0,
) -> GRPOLoss:
    """Perform one optimizer epoch over a fresh, already-collected rollout."""

    prompts, completions, tokens = input_ids.shape
    flat_inputs = input_ids.reshape(prompts * completions, tokens)
    output = model(flat_inputs)
    if output.logits is None:
        raise RuntimeError("policy forward did not return logits")
    flat_targets = target_ids.reshape(prompts * completions, tokens)
    new_log_probs = sequence_log_probs(output.logits, flat_targets).reshape_as(input_ids)
    objective = grpo_loss(
        new_log_probs,
        old_log_probs,
        reference_log_probs,
        rewards,
        completion_mask,
        epsilon=epsilon,
        beta=beta,
    )
    optimizer.zero_grad(set_to_none=True)
    if float(objective.valid_group_fraction) == 0.0:
        # A skipped rollout must not invoke AdamW: even zero gradients would
        # otherwise update parameters through decoupled weight decay.
        return objective
    objective.loss.backward()  # type: ignore[no-untyped-call]
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not math.isfinite(float(norm)):
        raise FloatingPointError("GRPO gradient norm became non-finite")
    optimizer.step()
    return objective
