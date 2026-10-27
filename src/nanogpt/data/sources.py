"""Raw text/conversation sources, deterministic splits, and offline fixtures.

Hugging Face ``datasets`` is imported lazily (install the ``data`` extra), so
the offline smoke run and tests never need the network.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from nanogpt.config import SourceConfig
from nanogpt.data.schemas import ConversationRecord, FineWebRecord, GRPOPrompt

FINEWEB_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
GSM8K_ID = "openai/gsm8k"
SMOLTALK_ID = "HuggingFaceTB/smoltalk"
ANSWER_INSTRUCTION = "End with <final>answer</final>."


# --------------------------------------------------------------------------- splits


def stable_bucket(record_id: str, buckets: int = 10_000) -> int:
    """Map an ID to a process-independent bucket using SHA-256."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def assign_split(record_id: str, validation_fraction: float = 0.005) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    threshold = round(validation_fraction * 10_000)
    return "validation" if stable_bucket(record_id) < threshold else "train"


def gsm8k_partition(record_id: str) -> str:
    """GSM8K train buckets 0..5 feed SFT and 6..9 feed GRPO; never both."""

    return "sft" if stable_bucket(record_id, 10) <= 5 else "grpo"


def require_allowlisted(
    records: Iterable[ConversationRecord], allowlist: set[tuple[str, str]]
) -> list[ConversationRecord]:
    accepted: list[ConversationRecord] = []
    for record in records:
        if (record.source, record.license) not in allowlist:
            raise ValueError(f"unapproved source/license: {record.source!r} / {record.license!r}")
        accepted.append(record)
    return accepted


# --------------------------------------------------------------------------- fixtures


def fixture_documents() -> list[FineWebRecord]:
    """A tiny offline pretraining corpus for the smoke run and tests."""

    paragraphs = [
        "A small language model learns by predicting the next token in ordinary prose.",
        "Transformers mix information with causal attention and then feed-forward layers.",
        "Reproducible experiments record data, configuration, seeds, and checkpoint lineage.",
        "Arithmetic has answers that software can verify without a subjective reward model.",
        "Students compare fractions, decimals, and percentages to build number sense.",
        "Photosynthesis converts light energy into chemical energy stored in sugar.",
    ]
    rng = random.Random(0)
    documents = []
    for index in range(240):
        body = " ".join(rng.choice(paragraphs) for _ in range(4))
        documents.append(FineWebRecord(f"fixture-{index}", body, "fixture-v1"))
    return documents


_COLUMNS = ("ones", "tens", "hundreds", "thousands")


def carry_steps(left: int, right: int) -> str:
    """Column-by-column addition with every digit its own token.

    BPE merges a digit run such as ``127`` into one token, which makes the sum
    a single lookup with no intermediate signal. Spacing the digits keeps each
    column's arithmetic (and the final digit sequence) visible to the model.
    """

    if left < 0 or right < 0:
        raise ValueError("carry_steps supports non-negative operands")
    a, b = str(left)[::-1], str(right)[::-1]
    lines, digits, carry = [], [], 0
    for column in range(max(len(a), len(b))):
        x = int(a[column]) if column < len(a) else 0
        y = int(b[column]) if column < len(b) else 0
        total = x + y + carry
        name = _COLUMNS[column] if column < len(_COLUMNS) else f"column {column + 1}"
        terms = f"{x} + {y}" + (f" + {carry}" if carry else "")
        written = " ".join(str(total))
        carry, digit = divmod(total, 10)
        lines.append(
            f"{name}: {terms} = {written}, write {digit}" + (f" carry {carry}" if carry else "")
        )
        digits.append(str(digit))
    if carry:
        name = _COLUMNS[len(digits)] if len(digits) < len(_COLUMNS) else "carry"
        lines.append(f"{name}: carry {carry}, write {carry}")
        digits.append(str(carry))
    answer = "".join(reversed(digits)).lstrip("0") or "0"
    lines.append(f"digits: {' '.join(answer)}")
    return "\n".join(lines) + f"\n<final>{answer}</final>"


def arithmetic_conversations(
    count: int, *, seed: int = 17, answer_format: str = "direct"
) -> Iterator[ConversationRecord]:
    if answer_format not in ("direct", "carry"):
        raise ValueError(f"unknown arithmetic answer format: {answer_format!r}")
    rng = random.Random(seed)
    for index in range(count):
        left, right = rng.randint(0, 99), rng.randint(0, 99)
        answer = (
            carry_steps(left, right)
            if answer_format == "carry"
            else f"<final>{left + right}</final>"
        )
        yield ConversationRecord(
            id=f"arithmetic-sft-{seed}-{index}",
            messages=(
                {"role": "user", "content": f"What is {left} + {right}? {ANSWER_INSTRUCTION}"},
                {"role": "assistant", "content": answer},
            ),
            source="nanogpt-arithmetic",
            license="CC0-1.0",
        )


def arithmetic_prompts(count: int, *, seed: int = 23) -> Iterator[GRPOPrompt]:
    rng = random.Random(seed)
    for index in range(count):
        left, right = rng.randint(0, 99), rng.randint(0, 99)
        yield GRPOPrompt(
            id=f"arithmetic-grpo-{seed}-{index}",
            messages=(
                {"role": "user", "content": f"What is {left} + {right}? {ANSWER_INSTRUCTION}"},
            ),
            answer=str(left + right),
        )


# --------------------------------------------------------------------------- remote


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised only with the extra
        raise RuntimeError("install the 'data' extra: uv sync --extra data") from error
    return load_dataset(*args, **kwargs)


def stream_fineweb(*, revision: str | None, split: str = "train") -> Iterator[FineWebRecord]:
    """Stream revision-pinned FineWeb-Edu records."""

    if not revision:
        raise ValueError("a pinned FineWeb-Edu revision is required")
    dataset = _load_dataset(
        FINEWEB_ID, name=FINEWEB_CONFIG, split=split, revision=revision, streaming=True
    )
    seen: set[str] = set()
    for row in dataset:
        text, record_id = row.get("text"), row.get("id")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(record_id, str) or record_id in seen:
            raise ValueError(f"FineWeb row has a missing or duplicate id: {record_id!r}")
        seen.add(record_id)
        yield FineWebRecord(record_id, text, revision)


_CALCULATOR = re.compile(r"<<[^<>]*>>")


@dataclass(frozen=True)
class GSM8KProblem:
    id: str
    question: str
    solution: str  # worked reasoning without the final "#### answer" line
    answer: str


def parse_gsm8k(index: int, row: dict[str, Any]) -> GSM8KProblem:
    """Split a GSM8K row into reasoning and final answer.

    Answers end with ``#### <number>``; the ``<<48/2=24>>`` calculator
    annotations are tool traces, not text the model should learn to emit.
    """

    reasoning, marker, final = str(row["answer"]).rpartition("####")
    if not marker or not final.strip():
        raise ValueError(f"GSM8K row {index} has no '#### answer' line")
    return GSM8KProblem(
        id=f"gsm8k-train-{index}",
        question=str(row["question"]).strip(),
        solution=_CALCULATOR.sub("", reasoning).strip(),
        answer=final.strip().replace(",", ""),
    )


def _gsm8k_problems(source: SourceConfig, partition: str) -> Iterator[GSM8KProblem]:
    """Yield problems from one stable GSM8K train partition (test is eval-only)."""

    dataset = _load_dataset(GSM8K_ID, "main", split="train", revision=source.revision)
    for index, row in enumerate(dataset):
        if gsm8k_partition(f"gsm8k-train-{index}") == partition:
            yield parse_gsm8k(index, row)


def _smoltalk_rows(source: SourceConfig, subset: str) -> Iterator[dict[str, Any]]:
    # Streaming: the full subsets are far larger than the slice a run uses.
    yield from _load_dataset(
        SMOLTALK_ID, subset, split="train", revision=source.revision, streaming=True
    )


def _smoltalk_conversations(source: SourceConfig) -> Iterator[ConversationRecord]:
    """Interleave the configured subsets so ``examples`` is split evenly across them."""

    if not source.license:
        raise ValueError("smoltalk sources must declare the allowlisted license")
    if not source.subsets:
        raise ValueError("smoltalk sources must list at least one subset")
    streams = [(subset, enumerate(_smoltalk_rows(source, subset))) for subset in source.subsets]
    while streams:
        for entry in list(streams):
            subset, rows = entry
            item = next(rows, None)
            if item is None:
                streams.remove(entry)
                continue
            index, row = item
            yield ConversationRecord.from_mapping(
                {
                    "id": f"smoltalk-{subset}-{index}",
                    "messages": row["messages"],
                    "source": f"{SMOLTALK_ID}/{subset}",
                    "license": source.license,
                }
            )


def conversations(sources: Iterable[SourceConfig]) -> list[ConversationRecord]:
    """Materialize the SFT mixture: each source contributes ``examples`` records."""

    records: list[ConversationRecord] = []
    for source in sources:
        stream: Iterable[ConversationRecord]
        if source.kind == "arithmetic":
            stream = arithmetic_conversations(
                source.examples, seed=source.seed, answer_format=source.answer_format
            )
        elif source.kind == "gsm8k":
            stream = (
                ConversationRecord(
                    id=problem.id,
                    messages=(
                        {"role": "user", "content": f"{problem.question} {ANSWER_INSTRUCTION}"},
                        {
                            "role": "assistant",
                            "content": f"{problem.solution}\n<final>{problem.answer}</final>",
                        },
                    ),
                    source=GSM8K_ID,
                    license=source.license or "MIT",
                )
                for problem in _gsm8k_problems(source, "sft")
            )
        else:
            stream = _smoltalk_conversations(source)
        records.extend(record for _, record in zip(range(source.examples), stream, strict=False))
    random.Random(0).shuffle(records)
    return records


def prompts(sources: Iterable[SourceConfig]) -> list[GRPOPrompt]:
    """Materialize the GRPO prompt mixture from verifiable sources only."""

    records: list[GRPOPrompt] = []
    for source in sources:
        stream: Iterable[GRPOPrompt]
        if source.kind == "arithmetic":
            stream = arithmetic_prompts(source.examples, seed=source.seed)
        elif source.kind == "gsm8k":
            stream = (
                GRPOPrompt(
                    id=problem.id,
                    messages=(
                        {"role": "user", "content": f"{problem.question} {ANSWER_INSTRUCTION}"},
                    ),
                    answer=problem.answer,
                )
                for problem in _gsm8k_problems(source, "grpo")
            )
        else:
            raise ValueError("GRPO sources must have verifiable answers (arithmetic or gsm8k)")
        records.extend(record for _, record in zip(range(source.examples), stream, strict=False))
    random.Random(0).shuffle(records)
    return records
