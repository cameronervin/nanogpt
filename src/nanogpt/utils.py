"""Small shared helpers: hashing, canonical JSON, seeding, and device selection."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def canonical_json(value: Any) -> str:
    """Encode a JSON-compatible value with a stable byte representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (including all visible CUDA devices).

    Deterministic algorithms make the smoke run and exact-resume checks
    meaningful. Paid runs may disable this when throughput matters more.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(name: str) -> torch.device:
    """Map ``auto`` to the best local accelerator and reject unavailable CUDA."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def resolve_dtype(name: str) -> torch.dtype:
    """Return the autocast compute dtype named in a run config."""

    try:
        return _DTYPES[name]
    except KeyError as error:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {name!r}") from error
