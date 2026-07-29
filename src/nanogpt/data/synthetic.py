"""Deterministic, license-free arithmetic fixtures for SFT and GRPO smoke runs."""

from __future__ import annotations

import random
from collections.abc import Iterator

from .schemas import ConversationRecord, GRPOPrompt


def arithmetic_conversations(count: int, *, seed: int = 17) -> Iterator[ConversationRecord]:
    rng = random.Random(seed)
    for index in range(count):
        left, right = rng.randint(0, 99), rng.randint(0, 99)
        yield ConversationRecord(
            id=f"synthetic-sft-{seed}-{index}",
            messages=(
                {"role": "user", "content": f"What is {left} + {right}?"},
                {"role": "assistant", "content": f"<final>{left + right}</final>"},
            ),
            source="nanogpt-synthetic-arithmetic",
            license="CC0-1.0",
        )


def arithmetic_prompts(count: int, *, seed: int = 23) -> Iterator[GRPOPrompt]:
    rng = random.Random(seed)
    for index in range(count):
        left, right = rng.randint(0, 999), rng.randint(0, 999)
        yield GRPOPrompt(
            id=f"synthetic-grpo-{seed}-{index}",
            messages=(
                {
                    "role": "user",
                    "content": f"Compute {left} + {right}. End with <final>answer</final>.",
                },
            ),
            answer=str(left + right),
        )
