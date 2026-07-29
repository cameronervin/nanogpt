"""Exact-match and pass@k metrics for verifiable reasoning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nanogpt.training.grpo import numeric_exact_reward, parse_numeric_answer


@dataclass(frozen=True)
class ReasoningMetrics:
    exact_match: float
    pass_at_k: float
    malformed_answer_rate: float
    prompt_count: int
    completions_per_prompt: int


def evaluate_reasoning(
    completions: Sequence[Sequence[str]], answers: Sequence[object]
) -> ReasoningMetrics:
    if len(completions) != len(answers) or not answers:
        raise ValueError("one non-empty completion group is required per answer")
    widths = {len(group) for group in completions}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("all prompts must have the same nonzero completion count")
    pass_one = 0.0
    passed = 0
    malformed = 0
    total = 0
    for group, answer in zip(completions, answers, strict=True):
        rewards = [numeric_exact_reward(completion, answer) for completion in group]
        pass_one += rewards[0]
        passed += int(any(rewards))
        malformed += sum(parse_numeric_answer(completion) is None for completion in group)
        total += len(group)
    return ReasoningMetrics(
        exact_match=pass_one / len(answers),
        pass_at_k=passed / len(answers),
        malformed_answer_rate=malformed / total,
        prompt_count=len(answers),
        completions_per_prompt=next(iter(widths)),
    )
