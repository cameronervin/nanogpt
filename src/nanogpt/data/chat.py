"""Fixed chat rendering and assistant-only supervision masks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .tokenizer import ByteBPETokenizer

VALID_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ChatEncoding:
    """Tokenized chat plus a mask aligned with ``input_ids``."""

    input_ids: list[int]
    assistant_mask: list[bool]

    def labels(self, ignore_index: int = -100) -> list[int]:
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
    messages: Iterable[Mapping[str, object]], tokenizer: ByteBPETokenizer
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
