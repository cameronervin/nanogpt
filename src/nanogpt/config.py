"""Strict configuration, artifact paths, and reproducibility helpers.

Configuration is deliberately small and explicit. Training runs are expensive, so a
misspelled key must fail before a cloud instance starts instead of being silently
ignored by a permissive configuration framework.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import yaml


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture parameters for the decoder-only Transformer."""

    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_query_heads: int
    n_kv_heads: int
    ffn_hidden_size: int
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_query_heads": self.n_query_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_hidden_size": self.ffn_hidden_size,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_query_heads:
            raise ValueError("d_model must be divisible by n_query_heads")
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        for numeric_name, numeric_value in (
            ("rope_theta", self.rope_theta),
            ("rms_norm_eps", self.rms_norm_eps),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, (int, float)):
                raise ValueError(
                    f"{numeric_name} must be a finite positive number, got {numeric_value!r}"
                )
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(
                    f"{numeric_name} must be a finite positive number, got {numeric_value!r}"
                )
        for boolean_name, boolean_value in (
            ("tie_embeddings", self.tie_embeddings),
            ("qk_norm", self.qk_norm),
        ):
            if not isinstance(boolean_value, bool):
                raise ValueError(f"{boolean_name} must be a boolean, got {boolean_value!r}")

    @property
    def head_dim(self) -> int:
        """Width of a single attention head."""

        return self.d_model // self.n_query_heads

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> Self:
        """Build from a mapping and reject unknown or missing keys."""

        if not isinstance(values, dict):
            raise TypeError("model configuration must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"unknown ModelConfig fields: {names}")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid ModelConfig: {error}") from error

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a model profile from YAML using safe parsing."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if not isinstance(values, dict):
            raise ValueError(f"{source} must contain a YAML mapping")
        # Permit a single descriptive wrapper while keeping the actual model strict.
        model_values = values.get("model", values)
        if set(values) == {"model"} and isinstance(model_values, dict):
            return cls.from_mapping(model_values)
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def canonical_json(self) -> str:
        """Serialize deterministically for manifests and compatibility checks."""

        return canonical_json(self.to_dict())

    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical representation."""

        return sha256_text(self.canonical_json())


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Canonical runtime directories rooted at ``NANOGPT_HOME``."""

    root: Path

    @classmethod
    def from_environment(cls, root: str | Path | None = None) -> Self:
        """Resolve the explicit root, environment override, or local default."""

        value = root if root is not None else os.environ.get("NANOGPT_HOME", "artifacts")
        return cls(Path(value).expanduser().resolve())

    @property
    def tokenizer(self) -> Path:
        return self.root / "tokenizer"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evaluations(self) -> Path:
        return self.root / "evaluations"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def create(self) -> None:
        """Create the documented runtime layout without touching existing data."""

        for path in (
            self.tokenizer,
            *(self.datasets / stage for stage in ("base", "sft", "grpo")),
            *(self.checkpoints / stage for stage in ("base", "sft", "grpo")),
            self.evaluations,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)


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
    """Hash UTF-8 text for configuration and manifest fingerprints."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python and PyTorch, including all visible CUDA devices.

    Deterministic algorithms make smoke tests and exact-resume checks meaningful.
    A production throughput probe may explicitly disable this when speed matters.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    # NumPy's compatibility/global generator is used at data boundaries and
    # must follow the same deterministic seed and checkpoint contract.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
