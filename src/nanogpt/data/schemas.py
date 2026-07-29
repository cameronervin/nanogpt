"""Validated lightweight record schemas used at data boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FineWebRecord:
    id: str
    text: str
    source_revision: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FineWebRecord:
        allowed = {"id", "text", "source_revision"}
        if set(value) != allowed or not all(isinstance(value[key], str) for key in allowed):
            raise ValueError("FineWeb record requires string id, text, and source_revision")
        return cls(str(value["id"]), str(value["text"]), str(value["source_revision"]))


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    messages: tuple[dict[str, str], ...]
    source: str
    license: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ConversationRecord:
        allowed = {"id", "messages", "source", "license"}
        if set(value) != allowed:
            raise ValueError(f"conversation fields must be exactly {sorted(allowed)}")
        identity_fields = ("id", "source", "license")
        if not all(isinstance(value.get(key), str) and value.get(key) for key in identity_fields):
            raise ValueError("conversation id, source, and license must be non-empty strings")
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
            raise ValueError("messages must be a sequence")
        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, Mapping):
                raise ValueError("each message must be an object")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise ValueError("message role/content is invalid")
            messages.append({"role": str(role), "content": content})
        if not messages:
            raise ValueError("conversation must contain messages")
        return cls(str(value["id"]), tuple(messages), str(value["source"]), str(value["license"]))


@dataclass(frozen=True)
class GRPOPrompt:
    id: str
    messages: tuple[dict[str, str], ...]
    answer: str
    reward_type: str = "numeric_exact"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GRPOPrompt:
        if value.get("reward_type") != "numeric_exact" or not isinstance(value.get("answer"), str):
            raise ValueError("GRPO prompt requires numeric_exact reward and string answer")
        conversation = ConversationRecord.from_mapping(
            {
                "id": value.get("id"),
                "messages": value.get("messages"),
                "source": "grpo",
                "license": "generated-or-upstream",
            }
        )
        return cls(conversation.id, conversation.messages, str(value["answer"]))
