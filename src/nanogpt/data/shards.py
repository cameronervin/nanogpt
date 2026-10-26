"""Checksummed uint16 token shards and a resumable, rank-partitioned loader."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from nanogpt.data.chat import TokenBatch
from nanogpt.data.schemas import FineWebRecord
from nanogpt.data.sources import assign_split
from nanogpt.tokenizer import Tokenizer
from nanogpt.utils import sha256_file

SPLITS = ("train", "validation")


@dataclass(frozen=True)
class ShardInfo:
    path: str
    split: str
    tokens: int
    min_id: int
    max_id: int
    sha256: str


def _write_shard(directory: Path, split: str, index: int, tokens: list[int]) -> ShardInfo:
    values = np.asarray(tokens, dtype=np.int64)
    if values.min() < 0 or values.max() > np.iinfo(np.uint16).max:
        raise ValueError("token IDs must fit unsigned 16-bit storage")
    path = directory / f"{split}-{index:05d}.bin"
    # "<u2" pins little-endian bytes regardless of the host CPU.
    with path.open("wb") as handle:
        values.astype("<u2").tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    return ShardInfo(
        path.name, split, len(tokens), int(values.min()), int(values.max()), sha256_file(path)
    )


def _batched(items: Iterable[FineWebRecord], size: int) -> Iterator[list[FineWebRecord]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def prepare_shards(
    records: Iterable[FineWebRecord],
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    shard_tokens: int = 100_000_000,
    validation_fraction: float = 0.005,
    max_tokens: int | None = None,
    encode_batch_size: int = 1024,
) -> dict[str, object]:
    """Pretokenize documents into shards without letting a document cross splits.

    Each document is wrapped as ``<|bos|> text <|eos|>``. Output is written to a
    temporary sibling and renamed into place only when complete.
    """

    if shard_tokens <= 1:
        raise ValueError("shard_tokens must be greater than one")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable dataset at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    bos, eos = tokenizer.special_to_id["<|bos|>"], tokenizer.special_to_id["<|eos|>"]
    buffers: dict[str, list[int]] = {split: [] for split in SPLITS}
    shards: list[ShardInfo] = []
    revisions: set[str] = set()
    documents = 0

    def flush(split: str) -> None:
        if buffers[split]:
            index = sum(info.split == split for info in shards)
            shards.append(_write_shard(temporary, split, index, buffers[split]))
            buffers[split] = []

    def train_tokens() -> int:
        return sum(info.tokens for info in shards if info.split == "train") + len(buffers["train"])

    def budget_spent() -> bool:
        return max_tokens is not None and train_tokens() >= max_tokens

    try:
        for batch in _batched(records, encode_batch_size):
            if budget_spent():
                break
            for record, ids in zip(
                batch, tokenizer.encode_batch(r.text for r in batch), strict=True
            ):
                # Whole documents only: stop at the first one past the budget.
                if budget_spent():
                    break
                revisions.add(record.source_revision)
                documents += 1
                split = assign_split(record.id, validation_fraction)
                buffers[split].extend([bos, *ids, eos])
                if len(buffers[split]) >= shard_tokens:
                    flush(split)
        for split in SPLITS:
            flush(split)
        if not any(info.split == "train" for info in shards):
            raise ValueError("preparation produced no training tokens")
        manifest: dict[str, object] = {
            "format_version": 2,
            "dtype": "uint16-le",
            "tokenizer_sha256": tokenizer.sha256,
            "source_revisions": sorted(revisions),
            "validation_fraction": validation_fraction,
            "documents": documents,
            "token_count": sum(info.tokens for info in shards),
            "shards": [asdict(info) for info in shards],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class Cursor:
    """Serializable position of the next unread token in this rank's stream."""

    epoch: int = 0
    shard: int = 0
    offset: int = 0


class TokenLoader:
    """Yield packed ``[batch, seq_len]`` next-token batches from local shards.

    Shards are memory-mapped (no per-token Python objects). Each rank owns the
    shards whose index is congruent to its rank, so ranks never overlap. The
    stream wraps into a new epoch when exhausted, and ``state_dict`` captures
    the exact cursor so a resumed run sees the identical batch sequence.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        split: str,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        verify_checksums: bool = True,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256_file(manifest_path)
        in_split = [item for item in self.manifest["shards"] if item["split"] == split]
        self.shards = in_split[rank::world_size]
        if not self.shards:
            raise ValueError(f"rank {rank} owns no {split!r} shards in {self.directory}")
        if verify_checksums:
            for item in self.shards:
                if sha256_file(self.directory / item["path"]) != item["sha256"]:
                    raise ValueError(f"checksum mismatch for {item['path']}")
        self._maps = [
            np.memmap(self.directory / item["path"], dtype="<u2", mode="r") for item in self.shards
        ]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.cursor = Cursor()

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.seq_len

    def _read(self, cursor: Cursor, count: int) -> tuple[np.ndarray, Cursor]:
        """Read ``count`` tokens from ``cursor``, crossing shards/epochs as needed."""

        pieces: list[np.ndarray] = []
        epoch, shard, offset = cursor.epoch, cursor.shard, cursor.offset
        while count > 0:
            data = self._maps[shard]
            piece = data[offset : offset + count]
            pieces.append(piece)
            count -= len(piece)
            offset += len(piece)
            if offset >= len(data):
                shard, offset = shard + 1, 0
                if shard == len(self._maps):
                    epoch, shard = epoch + 1, 0
        return np.concatenate(pieces), Cursor(epoch, shard, offset)

    def next_batch(self) -> TokenBatch:
        # Read one extra token for the shifted targets, but advance by exactly
        # batch*seq so consecutive batches tile the stream without gaps.
        tokens, after = self._read(self.cursor, self.tokens_per_batch)
        extra, _ = self._read(after, 1)
        window = torch.from_numpy(np.concatenate((tokens, extra)).astype(np.int64))
        self.cursor = after
        return TokenBatch(
            window[:-1].view(self.batch_size, self.seq_len),
            window[1:].view(self.batch_size, self.seq_len),
        )

    def __iter__(self) -> Iterator[TokenBatch]:
        while True:
            yield self.next_batch()

    def state_dict(self) -> dict[str, int]:
        return asdict(self.cursor)

    def load_state_dict(self, state: dict[str, int]) -> None:
        cursor = Cursor(**state)
        if not 0 <= cursor.shard < len(self.shards) or not (
            0 <= cursor.offset < len(self._maps[cursor.shard])
        ):
            raise ValueError("resume cursor is outside this rank's shards")
        self.cursor = cursor
