"""Verifiable numeric rewards shared by GRPO training and evaluation.

No ``eval``, code execution, or model grading: a completion earns 1.0 only if
its single final-answer marker parses to exactly the ground-truth number.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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


def score_completions(completions: Iterable[str], answers: Iterable[object]) -> Tensor:
    values = [
        numeric_exact_reward(completion, answer)
        for completion, answer in zip(completions, answers, strict=True)
    ]
    return torch.tensor(values, dtype=torch.float32)
