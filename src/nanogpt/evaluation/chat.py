"""Deterministic chat-template format validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatFormatMetrics:
    valid_termination_rate: float
    sample_count: int


def evaluate_chat_format(completions: Iterable[str]) -> ChatFormatMetrics:
    """Require exactly one final assistant terminator and no role injection."""

    values = list(completions)
    if not values:
        raise ValueError("chat format evaluation requires completions")
    valid = 0
    forbidden = ("<|system|>", "<|user|>", "<|assistant|>")
    for completion in values:
        terminators = completion.count("<|end|>")
        prefix = completion.rsplit("<|end|>", 1)[0] if terminators else completion
        # A terminator in the middle is not valid termination: accepting a
        # trailing role injection would overstate chat-format compliance.
        valid += int(
            terminators == 1
            and completion.endswith("<|end|>")
            and not any(token in prefix for token in forbidden)
        )
    return ChatFormatMetrics(valid / len(values), len(values))
