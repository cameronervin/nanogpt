"""Fixed chat template, assistant-only supervision, and SFT batch collation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from nanogpt.data.schemas import ConversationRecord
from nanogpt.model.gpt import IGNORE_INDEX
from nanogpt.tokenizer import Tokenizer

VALID_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class TokenBatch:
    """Next-token batch: ``targets[b, t]`` follows ``input_ids[b, t]``; -100 is ignored."""

    input_ids: Tensor
    targets: Tensor


@dataclass(frozen=True)
class ChatEncoding:
    """Tokenized chat plus a mask aligned with ``input_ids``."""

    input_ids: list[int]
    assistant_mask: list[bool]

    def labels(self, ignore_index: int = IGNORE_INDEX) -> list[int]:
        """Return next-token labels with prompt/padding positions ignored.

        Position ``i`` predicts token ``i+1``.  Consequently, supervision at
        label position ``i`` depends on whether the *next* token belongs to an
        assistant span.
        """

        if not self.input_ids:
            return []
        labels = [ignore_index] * len(self.input_ids)
        for index in range(len(self.input_ids) - 1):
            if self.assistant_mask[index + 1]:
                labels[index] = self.input_ids[index + 1]
        return labels


def _validated_messages(messages: Iterable[Mapping[str, object]]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            raise ValueError(f"message {index} must contain a valid role and string content")
        normalized.append((str(role), content))
    if not normalized:
        raise ValueError("a conversation must contain at least one message")
    return normalized


def render_conversation(messages: Iterable[Mapping[str, object]]) -> str:
    """Serialize messages with the project's immutable chat template."""

    parts = ["<|bos|>"]
    for role, content in _validated_messages(messages):
        parts.extend((f"<|{role}|>", content, "<|end|>"))
    return "".join(parts)


def encode_conversation(
    messages: Iterable[Mapping[str, object]], tokenizer: Tokenizer
) -> ChatEncoding:
    """Encode chat and mark only assistant content and terminators for loss."""

    normalized = _validated_messages(messages)
    ids = [tokenizer.special_to_id["<|bos|>"]]
    mask = [False]
    for role, content in normalized:
        role_id = tokenizer.special_to_id[f"<|{role}|>"]
        content_ids = tokenizer.encode(content)
        end_id = tokenizer.special_to_id["<|end|>"]
        ids.extend((role_id, *content_ids, end_id))
        # The role marker is context.  Assistant text and its terminator are
        # targets, which teaches both response content and clean stopping.
        is_assistant = role == "assistant"
        mask.extend((False, *([is_assistant] * len(content_ids)), is_assistant))
    return ChatEncoding(ids, mask)


def collate_sft(
    records: Iterable[ConversationRecord], tokenizer: Tokenizer, *, max_seq_len: int
) -> TokenBatch:
    """Pad conversations into ``(input_ids, targets)`` with ``-100`` off-assistant.

    Overlength records are rejected rather than truncated: cutting the target
    could teach the model that an unfinished answer is valid.
    """

    encodings = [encode_conversation(record.messages, tokenizer) for record in records]
    if not encodings:
        raise ValueError("cannot collate an empty SFT batch")
    if any(not any(item.assistant_mask) for item in encodings):
        raise ValueError("every SFT conversation must contain assistant targets")
    if any(len(item.input_ids) > max_seq_len + 1 for item in encodings):
        raise ValueError("conversation exceeds max_seq_len; filter it during preparation")
    pad_id = tokenizer.special_to_id["<|pad|>"]
    width = max(len(item.input_ids) for item in encodings) - 1
    inputs, targets = [], []
    for item in encodings:
        labels = item.labels()[:-1]
        padding = width - len(labels)
        inputs.append(item.input_ids[:-1] + [pad_id] * padding)
        targets.append(labels + [IGNORE_INDEX] * padding)
    return TokenBatch(
        torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    )
