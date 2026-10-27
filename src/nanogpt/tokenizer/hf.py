"""Production byte-level BPE backed by Hugging Face ``tokenizers`` (Rust).

Training and batch encoding run in native code, which makes the planned
billion-character tokenizer sample and multi-billion-token sharding practical.

Reserved tokens occupy IDs 0..6 in the same order as the reference tokenizer.
They are deliberately *not* registered as HF "added tokens": added tokens are
matched inside raw text, which would let a user type ``<|assistant|>`` and
inject a role marker. Instead they are plain vocabulary entries that no merge
can produce, and the chat encoder inserts their IDs explicitly.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer as RustTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from nanogpt.tokenizer.reference import RESERVED_TOKENS
from nanogpt.utils import canonical_json, sha256_text

BACKEND = "hf"


@dataclass(frozen=True)
class HFTokenizerManifest:
    backend: str
    format_version: int
    vocab_size: int
    reserved_tokens: dict[str, int]
    train_chars: int
    library_version: str
    sha256: str


def _limit_chars(texts: Iterable[str], max_chars: int) -> Iterator[str]:
    """Yield texts in order until the character budget is spent (deterministic)."""

    remaining = max_chars
    for text in texts:
        if remaining <= 0:
            return
        piece = text[:remaining]
        remaining -= len(piece)
        yield piece


def _byte_level(model: models.Model) -> RustTokenizer:
    tokenizer = RustTokenizer(model)
    # GPT-2-style regex pre-split: merges never cross word/number/space classes.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


class HFTokenizer:
    """Byte-level BPE with immutable reserved IDs, matching the reference API."""

    def __init__(self, rust: RustTokenizer, *, train_chars: int = 0) -> None:
        self._rust = rust
        self.train_chars = train_chars
        vocab = rust.get_vocab()
        self.special_to_id = {token: vocab[token] for token in RESERVED_TOKENS}
        if list(self.special_to_id.values()) != list(range(len(RESERVED_TOKENS))):
            raise ValueError("reserved tokens must occupy IDs 0..6 in contract order")
        self.id_to_special = {index: token for token, index in self.special_to_id.items()}
        # Cached: the Rust ``get_vocab_size`` rebuilds the whole vocab map per call,
        # and ``decode`` checks the size once per token.
        self._vocab_size = int(rust.get_vocab_size())

    @classmethod
    def train(
        cls, texts: Iterable[str], vocab_size: int = 32_768, *, max_chars: int = 10**9
    ) -> HFTokenizer:
        minimum = len(RESERVED_TOKENS) + 256
        if not minimum <= vocab_size <= 65_536:
            raise ValueError(f"vocab_size must be between {minimum} and 65536")
        trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size,
            min_frequency=2,
            # Special tokens first so they receive IDs 0..6, then all 256 bytes
            # so every UTF-8 input is representable.
            special_tokens=list(RESERVED_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        counted = 0

        def counting(stream: Iterable[str]) -> Iterator[str]:
            nonlocal counted
            for text in stream:
                counted += len(text)
                yield text

        draft = _byte_level(models.BPE())
        draft.train_from_iterator(counting(_limit_chars(texts, max_chars)), trainer=trainer)
        # Rebuild from the learned vocab/merges *without* added tokens (see module docstring).
        state = json.loads(draft.to_str())["model"]
        merges = [
            tuple(pair) if isinstance(pair, list) else tuple(pair.split(" ", 1))
            for pair in state["merges"]
        ]
        rust = _byte_level(models.BPE(vocab=state["vocab"], merges=merges))
        return cls(rust, train_chars=counted)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def sha256(self) -> str:
        return sha256_text(self._rust.to_str())

    @property
    def manifest(self) -> HFTokenizerManifest:
        return HFTokenizerManifest(
            backend=BACKEND,
            format_version=1,
            vocab_size=self.vocab_size,
            reserved_tokens=dict(self.special_to_id),
            train_chars=self.train_chars,
            library_version=tokenizers.__version__,
            sha256=self.sha256,
        )

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        if not allowed_special:
            return list(self._rust.encode(text, add_special_tokens=False).ids)
        output: list[int] = []
        for piece, special in _split_special(text):
            output.extend([self.special_to_id[piece]] if special else self.encode(piece))
        return output

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        encodings = self._rust.encode_batch(list(texts), add_special_tokens=False)
        return [list(encoding.ids) for encoding in encodings]

    def decode(self, token_ids: Iterable[int], *, render_special: bool = False) -> str:
        parts: list[str] = []
        run: list[int] = []
        for token_id in map(int, token_ids):
            if token_id in self.id_to_special:
                parts.append(self._rust.decode(run))
                run = []
                if render_special:
                    parts.append(self.id_to_special[token_id])
            else:
                if not 0 <= token_id < self.vocab_size:
                    raise ValueError(f"token ID {token_id} is outside this vocabulary")
                run.append(token_id)
        parts.append(self._rust.decode(run))
        return "".join(parts)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text(self._rust.to_str(), encoding="utf-8")
        (path / "manifest.json").write_text(
            canonical_json(asdict(self.manifest)) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> HFTokenizer:
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("backend") != BACKEND:
            raise ValueError("not an hf tokenizer artifact")
        text = (path / "tokenizer.json").read_text(encoding="utf-8")
        if sha256_text(text) != manifest["sha256"]:
            raise ValueError("tokenizer checksum does not match its manifest")
        tokenizer = cls(RustTokenizer.from_str(text), train_chars=manifest["train_chars"])
        if manifest["reserved_tokens"] != tokenizer.special_to_id:
            raise ValueError("tokenizer manifest reserved tokens do not match")
        return tokenizer


def _split_special(text: str) -> Iterator[tuple[str, bool]]:
    """Split text into (piece, is_reserved_token) runs, leftmost-longest first."""

    cursor = 0
    ordered = sorted(RESERVED_TOKENS, key=len, reverse=True)
    while cursor < len(text):
        hits = [(text.find(token, cursor), token) for token in ordered]
        hits = [(position, token) for position, token in hits if position >= 0]
        if not hits:
            yield text[cursor:], False
            return
        position, token = min(hits, key=lambda hit: hit[0])
        if position > cursor:
            yield text[cursor:position], False
        yield token, True
        cursor = position + len(token)
