"""Build a run's tokenizer and datasets (a one-time step before any training).

Each artifact is written to a temporary sibling and renamed into place, and an
existing artifact is reused rather than rebuilt: datasets are immutable inputs
whose hashes are recorded in every checkpoint manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanogpt.data.chat import encode_conversation
from nanogpt.data.schemas import FineWebRecord
from nanogpt.data.shards import prepare_shards
from nanogpt.data.sources import (
    assign_split,
    conversations,
    fixture_documents,
    prompts,
    stream_fineweb,
)
from nanogpt.tokenizer import Tokenizer, load_tokenizer, train_tokenizer

if TYPE_CHECKING:
    from nanogpt.train.trainer import RunContext

SFT_VALIDATION_FRACTION = 0.05


def documents(ctx: RunContext) -> Iterator[FineWebRecord]:
    data = ctx.config.data
    if data.source == "fixture":
        yield from fixture_documents()
    else:
        yield from stream_fineweb(revision=data.revision)


def _atomic_dir(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))


def build_tokenizer(ctx: RunContext) -> Tokenizer:
    """Train the run's tokenizer once; later calls load the saved artifact."""

    destination = ctx.paths.tokenizer
    if destination.exists():
        return load_tokenizer(destination)
    config = ctx.config.tokenizer
    tokenizer = train_tokenizer(
        config.backend,
        (record.text for record in documents(ctx)),
        vocab_size=config.vocab_size,
        max_chars=config.train_chars,
    )
    temporary = _atomic_dir(destination)
    try:
        tokenizer.save(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tokenizer


def _write_jsonl_dir(
    destination: Path, files: dict[str, list[Any]], manifest: dict[str, Any]
) -> None:
    temporary = _atomic_dir(destination)
    try:
        for name, rows in files.items():
            lines = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)
            (temporary / name).write_text(lines, encoding="utf-8")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_data(ctx: RunContext) -> dict[str, Any]:
    """Write base shards, SFT conversations, and GRPO prompts for this run."""

    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    config = ctx.config
    summary: dict[str, Any] = {}

    base_dir = ctx.paths.data("base")
    if not base_dir.exists():
        prepare_shards(
            documents(ctx),
            tokenizer,
            base_dir,
            shard_tokens=config.data.shard_tokens,
            validation_fraction=config.data.validation_fraction,
            max_tokens=config.data.max_tokens,
        )
    summary["base"] = json.loads((base_dir / "manifest.json").read_text())["token_count"]

    sft_dir = ctx.paths.data("sft")
    if not sft_dir.exists():
        limit = ctx.model_config.max_seq_len + 1
        kept, rejected = [], 0
        for record in conversations(config.sft.sources):
            # Reject (never truncate) conversations that do not fit the context.
            if len(encode_conversation(record.messages, tokenizer).input_ids) > limit:
                rejected += 1
            else:
                kept.append(record)
        splits: dict[str, list[Any]] = {"train.jsonl": [], "validation.jsonl": []}
        for record in kept:
            split = assign_split(record.id, SFT_VALIDATION_FRACTION)
            splits["train.jsonl" if split == "train" else "validation.jsonl"].append(record)
        _write_jsonl_dir(
            sft_dir,
            splits,
            {
                "tokenizer_sha256": tokenizer.sha256,
                "train": len(splits["train.jsonl"]),
                "validation": len(splits["validation.jsonl"]),
                "rejected_overlength": rejected,
                "sources": [asdict(source) for source in config.sft.sources],
            },
        )
    summary["sft"] = json.loads((sft_dir / "manifest.json").read_text())["train"]

    grpo_dir = ctx.paths.data("grpo")
    if not grpo_dir.exists():
        records = prompts(config.grpo.sources)
        _write_jsonl_dir(
            grpo_dir,
            {"train.jsonl": records},
            {"train": len(records), "sources": [asdict(s) for s in config.grpo.sources]},
        )
    summary["grpo"] = json.loads((grpo_dir / "manifest.json").read_text())["train"]
    return summary
