"""Stage 2: assistant-only supervised fine-tuning from the base checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanogpt.data.chat import TokenBatch, collate_sft
from nanogpt.data.schemas import ConversationRecord
from nanogpt.evaluate import evaluate_language_model
from nanogpt.model.gpt import GPT
from nanogpt.tokenizer import load_tokenizer
from nanogpt.train.checkpoint import load_checkpoint, read_manifest
from nanogpt.train.trainer import RunContext, Trainer, TrainResult, finished_checkpoint
from nanogpt.utils import sha256_file


def read_conversations(path: Path) -> list[ConversationRecord]:
    return [
        ConversationRecord.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_sft(ctx: RunContext, *, resume: bool = False) -> TrainResult:
    config = ctx.config.sft
    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    data_dir = ctx.paths.data("sft")
    train_path, validation_path = data_dir / "train.jsonl", data_dir / "validation.jsonl"
    records = read_conversations(train_path)
    validation = read_conversations(validation_path)
    if not records:
        raise ValueError(f"no SFT training conversations in {train_path}")
    parent = finished_checkpoint(ctx.paths, "base")

    model = GPT(ctx.model_config)
    load_checkpoint(
        parent, model=model, expected={"tokenizer_sha256": tokenizer.sha256}, restore_rng=False
    )
    max_len = ctx.model_config.max_seq_len
    # The cursor is plain data so a resumed run continues with the same records.
    cursor = {"next": 0}

    def next_batch() -> TokenBatch:
        start = cursor["next"]
        cursor["next"] += config.micro_batch_size
        batch = [records[(start + i) % len(records)] for i in range(config.micro_batch_size)]
        return collate_sft(batch, tokenizer, max_seq_len=max_len)

    def load_state(state: dict[str, Any]) -> None:
        cursor["next"] = int(state["next"])

    def evaluate() -> dict[str, float]:
        size = config.micro_batch_size
        batches = [
            collate_sft(validation[i : i + size], tokenizer, max_seq_len=max_len)
            for i in range(0, len(validation), size)
        ]
        metrics = evaluate_language_model(model, batches, device=ctx.device)
        return {"val_assistant_loss": metrics.loss}

    trainer = Trainer(
        ctx,
        "sft",
        config,
        model,
        lineage={
            "parent_model_sha256": read_manifest(parent)["model_sha256"],
            "tokenizer_sha256": tokenizer.sha256,
            "dataset_sha256": sha256_file(train_path),
        },
        evaluate=evaluate if validation else None,
        data_state=(lambda: dict(cursor), load_state),
    )
    if (finished := trainer.prepare(resume=resume)) is not None:
        return finished
    return trainer.fit(trainer.lm_step(next_batch))
