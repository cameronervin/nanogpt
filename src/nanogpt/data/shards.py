"""Checksum-addressed uint16 token shards and exactly resumable iteration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from array import array
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .fineweb import assign_split
from .schemas import FineWebRecord
from .tokenizer import ByteBPETokenizer


@dataclass(frozen=True)
class ShardInfo:
    path: str
    split: str
    tokens: int
    min_id: int
    max_id: int
    sha256: str


@dataclass(frozen=True)
class Cursor:
    """Serializable location of the next token to consume."""

    shard_index: int = 0
    token_offset: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_uint16(path: Path, token_ids: list[int]) -> ShardInfo:
    if not token_ids:
        raise ValueError("cannot write an empty shard")
    if min(token_ids) < 0 or max(token_ids) > 65_535:
        raise ValueError("token IDs must fit unsigned 16-bit storage")
    values = array("H", token_ids)
    # Manifest bytes are explicitly little-endian, independent of host CPU.
    if values.itemsize != 2:
        raise RuntimeError("platform does not provide a 16-bit unsigned short")
    import sys

    if sys.byteorder != "little":  # pragma: no cover - CI platforms are little-endian
        values.byteswap()
    with path.open("wb") as handle:
        values.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    return ShardInfo(path.name, "", len(token_ids), min(token_ids), max(token_ids), _sha256(path))


def prepare_shards(
    records: Iterable[FineWebRecord],
    tokenizer: ByteBPETokenizer,
    output_dir: str | Path,
    *,
    shard_tokens: int = 1_000_000,
    validation_fraction: float = 0.005,
) -> dict[str, object]:
    """Pretokenize documents without allowing them to cross data splits."""

    if shard_tokens <= 1:
        raise ValueError("shard_tokens must be greater than one")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable dataset at {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    buffers: dict[str, list[int]] = {"train": [], "validation": []}
    shards: list[ShardInfo] = []
    revisions: set[str] = set()

    def flush(split: str) -> None:
        values = buffers[split]
        if not values:
            return
        filename = f"{split}-{sum(info.split == split for info in shards):05d}.bin"
        partial = _write_uint16(temporary / filename, values)
        shards.append(
            ShardInfo(
                partial.path,
                split,
                partial.tokens,
                partial.min_id,
                partial.max_id,
                partial.sha256,
            )
        )
        buffers[split] = []

    try:
        for record in records:
            revisions.add(record.source_revision)
            split = assign_split(record.id, validation_fraction)
            document = [tokenizer.special_to_id["<|bos|>"]]
            document.extend(tokenizer.encode(record.text))
            document.append(tokenizer.special_to_id["<|eos|>"])
            # Very large documents get their own shard; they remain wholly in
            # one split, preserving the no-leakage invariant.
            if buffers[split] and len(buffers[split]) + len(document) > shard_tokens:
                flush(split)
            buffers[split].extend(document)
            if len(buffers[split]) >= shard_tokens:
                flush(split)
        flush("train")
        flush("validation")
        manifest: dict[str, object] = {
            "format_version": 1,
            "dtype": "uint16-le",
            "tokenizer_sha256": tokenizer.sha256,
            "source_revisions": sorted(revisions),
            "validation_fraction": validation_fraction,
            "token_count": sum(info.tokens for info in shards),
            "shards": [asdict(info) for info in shards],
        }
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class ShardReader:
    """Iterate one deterministic rank partition with an exact resume cursor."""

    def __init__(
        self,
        directory: str | Path,
        *,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        verify_checksums: bool = True,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        all_shards = [item for item in self.manifest["shards"] if item["split"] == split]
        self.shards = [item for index, item in enumerate(all_shards) if index % world_size == rank]
        if verify_checksums:
            for item in self.shards:
                if _sha256(self.directory / item["path"]) != item["sha256"]:
                    raise ValueError(f"checksum mismatch for {item['path']}")

    def iter_tokens(self, cursor: Cursor | None = None) -> Iterator[tuple[int, Cursor]]:
        cursor = cursor or Cursor()
        if cursor.shard_index < 0 or cursor.shard_index > len(self.shards):
            raise ValueError("resume shard index is outside this rank partition")
        if cursor.shard_index == len(self.shards):
            if cursor.token_offset != 0:
                raise ValueError("a completed resume cursor must have zero token offset")
            return
        if (
            cursor.token_offset < 0
            or cursor.token_offset >= self.shards[cursor.shard_index]["tokens"]
        ):
            raise ValueError("resume token offset is outside the selected shard")
        for shard_index in range(cursor.shard_index, len(self.shards)):
            item = self.shards[shard_index]
            values = array("H")
            with (self.directory / item["path"]).open("rb") as handle:
                values.fromfile(handle, item["tokens"])
            import sys

            if sys.byteorder != "little":  # pragma: no cover
                values.byteswap()
            start = cursor.token_offset if shard_index == cursor.shard_index else 0
            for offset in range(start, len(values)):
                next_cursor = (
                    Cursor(shard_index + 1, 0)
                    if offset + 1 == len(values)
                    else Cursor(shard_index, offset + 1)
                )
                yield int(values[offset]), next_cursor
