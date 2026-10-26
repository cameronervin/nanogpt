"""Tokenizers sharing one contract: byte-level BPE with reserved IDs 0..6.

``hf`` is the production backend; ``reference`` is a readable pure-Python
implementation used for teaching and tiny fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from nanogpt.tokenizer.hf import HFTokenizer
from nanogpt.tokenizer.reference import RESERVED_TOKENS, ByteBPETokenizer


class Tokenizer(Protocol):
    special_to_id: dict[str, int]

    @property
    def vocab_size(self) -> int: ...

    @property
    def sha256(self) -> str: ...

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]: ...

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]: ...

    def decode(self, token_ids: Iterable[int], *, render_special: bool = False) -> str: ...

    def save(self, directory: str | Path) -> None: ...


def train_tokenizer(
    backend: str, texts: Iterable[str], *, vocab_size: int, max_chars: int
) -> Tokenizer:
    if backend == "hf":
        return HFTokenizer.train(texts, vocab_size, max_chars=max_chars)
    if backend == "reference":
        return ByteBPETokenizer.train(texts, vocab_size, max_training_bytes=max_chars)
    raise ValueError(f"unknown tokenizer backend {backend!r}")


def load_tokenizer(directory: str | Path) -> Tokenizer:
    """Load either backend, dispatching on the manifest's ``backend`` field."""

    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("backend") == "hf":
        return HFTokenizer.load(directory)
    return ByteBPETokenizer.load(directory)


__all__ = [
    "RESERVED_TOKENS",
    "ByteBPETokenizer",
    "HFTokenizer",
    "Tokenizer",
    "load_tokenizer",
    "train_tokenizer",
]
