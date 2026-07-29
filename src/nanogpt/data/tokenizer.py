"""A small, dependency-free byte-level BPE tokenizer.

The implementation is deliberately readable rather than optimized.  Tokenizer
training is normally a one-time data-preparation step, so keeping the merge
rules and byte payloads explicit makes the resulting artifact easy to audit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

RESERVED_TOKENS: tuple[str, ...] = (
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)

# This implementation repeatedly rewrites the tokenized corpus once per merge.
# It is an educational/reference trainer, not the accelerated data-plane needed
# for the planned billion-character tokenizer corpus. Fail early rather than
# allowing a cloud preparation job to spend unbounded memory and CPU time.
REFERENCE_TRAINER_MAX_BYTES = 1_000_000


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class TokenizerManifest:
    """Stable identity and reserved-token contract for a tokenizer."""

    format_version: int
    vocab_size: int
    reserved_tokens: dict[str, int]
    sha256: str


class ByteBPETokenizer:
    """Reversible UTF-8 byte BPE with immutable reserved token IDs.

    IDs 0..6 are the chat/control vocabulary and IDs 7..262 represent raw
    bytes.  Learned merge tokens follow, so every UTF-8 input is representable
    even when it was absent from the training sample.
    """

    def __init__(self, merges: Iterable[tuple[int, int]] = ()) -> None:
        self.special_to_id = {token: index for index, token in enumerate(RESERVED_TOKENS)}
        self.id_to_special = {index: token for token, index in self.special_to_id.items()}
        self.byte_offset = len(RESERVED_TOKENS)
        self.merges = tuple((int(left), int(right)) for left, right in merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        # Each regular token stores the exact byte sequence it decodes to.
        self._pieces: dict[int, bytes] = {
            self.byte_offset + value: bytes((value,)) for value in range(256)
        }
        for rank, (left, right) in enumerate(self.merges):
            token_id = self.byte_offset + 256 + rank
            if left not in self._pieces or right not in self._pieces:
                raise ValueError(f"merge {rank} references an unknown token")
            self._pieces[token_id] = self._pieces[left] + self._pieces[right]

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256 + len(self.merges)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32_768,
        *,
        max_training_bytes: int = REFERENCE_TRAINER_MAX_BYTES,
    ) -> ByteBPETokenizer:
        """Learn deterministic pair merges from *texts*.

        Ties are broken lexicographically by token ID, making output identical
        across processes and Python hash seeds.  Empty/degenerate corpora simply
        produce a byte tokenizer with fewer tokens than the requested ceiling.
        """

        minimum = len(RESERVED_TOKENS) + 256
        if not minimum <= vocab_size <= 65_536:
            raise ValueError(f"vocab_size must be between {minimum} and 65536")
        if max_training_bytes <= 0:
            raise ValueError("max_training_bytes must be positive")
        sequences: list[list[int]] = []
        training_bytes = 0
        for text in texts:
            if not text:
                continue
            encoded = text.encode("utf-8")
            training_bytes += len(encoded)
            if training_bytes > max_training_bytes:
                raise ValueError(
                    "reference BPE trainer input exceeds "
                    f"{max_training_bytes:,} UTF-8 bytes; use the future accelerated "
                    "production tokenizer path described in docs/data-and-licenses.md"
                )
            sequences.append([len(RESERVED_TOKENS) + byte for byte in encoded])
        merges: list[tuple[int, int]] = []
        while minimum + len(merges) < vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(pairwise(sequence))
            if not counts:
                break
            best_count = max(counts.values())
            # A once-occurring pair does not compress the corpus and creates a
            # sample-order-sensitive vocabulary, so stop at frequency one.
            if best_count < 2:
                break
            pair = min(pair for pair, count in counts.items() if count == best_count)
            new_id = minimum + len(merges)
            merges.append(pair)
            sequences = [cls._merge_pair(sequence, pair, new_id) for sequence in sequences]
        return cls(merges)

    @staticmethod
    def _merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        """Encode text, optionally recognizing the seven reserved strings."""

        if not allowed_special:
            return self._encode_bytes(text.encode("utf-8"))
        pieces = sorted(self.special_to_id, key=len, reverse=True)
        output: list[int] = []
        cursor = 0
        while cursor < len(text):
            next_match: tuple[int, str] | None = None
            for special in pieces:
                position = text.find(special, cursor)
                if position >= 0 and (next_match is None or position < next_match[0]):
                    next_match = (position, special)
            if next_match is None:
                output.extend(self._encode_bytes(text[cursor:].encode("utf-8")))
                break
            position, special = next_match
            output.extend(self._encode_bytes(text[cursor:position].encode("utf-8")))
            output.append(self.special_to_id[special])
            cursor = position + len(special)
        return output

    def _encode_bytes(self, data: bytes) -> list[int]:
        tokens = [self.byte_offset + byte for byte in data]
        # Applying learned rules in rank order is equivalent to standard BPE
        # and avoids a dependency on a tokenizer runtime during training.
        for rank, pair in enumerate(self.merges):
            tokens = self._merge_pair(tokens, pair, self.byte_offset + 256 + rank)
        return tokens

    def decode(self, token_ids: Iterable[int], *, render_special: bool = False) -> str:
        chunks: list[bytes] = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id in self.id_to_special:
                if render_special:
                    chunks.append(self.id_to_special[token_id].encode("ascii"))
                continue
            try:
                chunks.append(self._pieces[token_id])
            except KeyError as error:
                raise ValueError(f"token ID {token_id} is outside this vocabulary") from error
        return b"".join(chunks).decode("utf-8", errors="replace")

    def iter_encode(self, texts: Iterable[str]) -> Iterator[list[int]]:
        for text in texts:
            yield self.encode(text)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "nanogpt-byte-bpe",
            "version": 1,
            "reserved_tokens": list(RESERVED_TOKENS),
            "merges": [list(pair) for pair in self.merges],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    @property
    def manifest(self) -> TokenizerManifest:
        return TokenizerManifest(1, self.vocab_size, dict(self.special_to_id), self.sha256)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer_json = _canonical_json(self.to_dict()) + "\n"
        (path / "tokenizer.json").write_text(tokenizer_json, encoding="utf-8")
        (path / "manifest.json").write_text(
            _canonical_json(self.manifest.__dict__) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> ByteBPETokenizer:
        path = Path(directory)
        payload = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        if payload.get("format") != "nanogpt-byte-bpe" or payload.get("version") != 1:
            raise ValueError("unsupported tokenizer artifact")
        if tuple(payload.get("reserved_tokens", ())) != RESERVED_TOKENS:
            raise ValueError("reserved token IDs do not match the chat contract")
        tokenizer = cls(tuple(tuple(pair) for pair in payload["merges"]))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest != tokenizer.manifest.__dict__:
            raise ValueError("tokenizer manifest or checksum mismatch")
        return tokenizer
