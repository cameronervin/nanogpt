"""FineWeb-Edu source adaptation and deterministic split assignment.

The online Hugging Face dependency is imported only inside ``stream_fineweb``.
This keeps every checked-in fixture and the local smoke pipeline offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from .schemas import FineWebRecord

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def stable_bucket(record_id: str, buckets: int = 10_000) -> int:
    """Map an ID to a process-independent bucket using SHA-256."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def assign_split(record_id: str, validation_fraction: float = 0.005) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    threshold = round(validation_fraction * 10_000)
    return "validation" if stable_bucket(record_id) < threshold else "train"


def stream_fineweb(*, revision: str, split: str = "train") -> Iterator[FineWebRecord]:
    """Stream revision-pinned FineWeb-Edu records.

    Calling this function requires the optional ``datasets`` package and
    network access.  A revision is mandatory so manifests remain reproducible.
    """

    if not revision:
        raise ValueError("a pinned FineWeb-Edu revision is required")
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised only with optional extra
        raise RuntimeError("install the 'data' extra to download FineWeb-Edu") from error
    dataset = load_dataset(
        DATASET_ID,
        name=DATASET_CONFIG,
        split=split,
        revision=revision,
        streaming=True,
    )
    for index, row in enumerate(dataset):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        record_id = str(row.get("id") or row.get("dump") or index)
        yield FineWebRecord(record_id, text, revision)
