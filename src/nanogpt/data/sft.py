"""SFT provenance checks, stable GSM8K partitioning, and padded collation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from .chat import encode_conversation
from .fineweb import stable_bucket
from .schemas import ConversationRecord
from .tokenizer import ByteBPETokenizer


def gsm8k_partition(record_id: str) -> str:
    """Assign buckets 0..5 to SFT and 6..9 to GRPO."""

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


@dataclass(frozen=True)
class SFTBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor


def collate_sft(
    records: Iterable[ConversationRecord],
    tokenizer: ByteBPETokenizer,
    *,
    max_seq_len: int,
) -> SFTBatch:
    """Pad conversations and preserve complete assistant terminators.

    Overlength records are rejected instead of silently truncating the target,
    which could teach the model that an unfinished answer is valid.
    """

    encodings = [encode_conversation(record.messages, tokenizer) for record in records]
    if not encodings:
        raise ValueError("cannot collate an empty SFT batch")
    if any(not any(item.assistant_mask) for item in encodings):
        raise ValueError("every SFT conversation must contain assistant targets")
    if any(len(item.input_ids) > max_seq_len + 1 for item in encodings):
        raise ValueError("conversation exceeds max_seq_len; bucket it separately")
    pad_id = tokenizer.special_to_id["<|pad|>"]
    width = min(max(len(item.input_ids) for item in encodings) - 1, max_seq_len)
    inputs, targets, masks = [], [], []
    for item in encodings:
        input_row = item.input_ids[:-1]
        # Store ignored targets as -100 as well as retaining the explicit mask.
        # The redundant representation makes accidental prompt supervision fail
        # closed in any downstream cross-entropy implementation.
        target_row = [
            token_id if supervised else -100
            for token_id, supervised in zip(
                item.input_ids[1:], item.assistant_mask[1:], strict=True
            )
        ]
        mask_row = item.assistant_mask[1:]
        padding = width - len(input_row)
        inputs.append(input_row + [pad_id] * padding)
        targets.append(target_row + [-100] * padding)
        masks.append(mask_row + [False] * padding)
    return SFTBatch(
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )
