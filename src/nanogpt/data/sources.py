"""Raw text/conversation sources, deterministic splits, and offline fixtures.

Hugging Face ``datasets`` is imported lazily (install the ``data`` extra), so
the offline smoke run and tests never need the network.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
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


def arithmetic_conversations(count: int, *, seed: int = 17) -> Iterator[ConversationRecord]:
    rng = random.Random(seed)
    for index in range(count):
        left, right = rng.randint(0, 99), rng.randint(0, 99)
        yield ConversationRecord(
            id=f"arithmetic-sft-{seed}-{index}",
            messages=(
                {"role": "user", "content": f"What is {left} + {right}? {ANSWER_INSTRUCTION}"},
                {"role": "assistant", "content": f"<final>{left + right}</final>"},
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


def _gsm8k_rows(source: SourceConfig, partition: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(id, question, final_answer)`` for one stable GSM8K partition."""

    dataset = _load_dataset(GSM8K_ID, "main", split="train", revision=source.revision)
    for index, row in enumerate(dataset):
        record_id = f"gsm8k-train-{index}"
        if gsm8k_partition(record_id) == partition:
            answer = str(row["answer"]).rsplit("####", 1)[-1].strip().replace(",", "")
            yield record_id, str(row["question"]), answer


def _smoltalk_conversations(source: SourceConfig) -> Iterator[ConversationRecord]:
    if not source.license:
        raise ValueError("smoltalk sources must declare the allowlisted license")
    for subset in source.subsets:
        dataset = _load_dataset(SMOLTALK_ID, subset, split="train", revision=source.revision)
        for index, row in enumerate(dataset):
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
            stream = arithmetic_conversations(source.examples, seed=source.seed)
        elif source.kind == "gsm8k":
            stream = (
                ConversationRecord(
                    id=record_id,
                    messages=(
                        {"role": "user", "content": f"{question} {ANSWER_INSTRUCTION}"},
                        {"role": "assistant", "content": f"<final>{answer}</final>"},
                    ),
                    source=GSM8K_ID,
                    license=source.license or "MIT",
                )
                for record_id, question, answer in _gsm8k_rows(source, "sft")
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
                    id=record_id,
                    messages=({"role": "user", "content": f"{question} {ANSWER_INSTRUCTION}"},),
                    answer=answer,
                )
                for record_id, question, answer in _gsm8k_rows(source, "grpo")
            )
        else:
            raise ValueError("GRPO sources must have verifiable answers (arithmetic or gsm8k)")
        records.extend(record for _, record in zip(range(source.examples), stream, strict=False))
    random.Random(0).shuffle(records)
    return records
