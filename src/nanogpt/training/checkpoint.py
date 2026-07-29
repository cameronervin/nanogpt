"""Atomic, fingerprint-validated stage checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch import nn

from nanogpt.config import canonical_json


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely continue the requested run."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    # Format-v1 checkpoints created before NumPy was used do not contain this
    # optional key; current checkpoints always record it.
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    random.setstate(state["python"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])  # type: ignore[arg-type]


def _model_config(model: nn.Module) -> dict[str, object]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("checkpointed models must expose a config attribute")
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError("model config must expose to_dict()")


def save_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    trainer_state: Mapping[str, object],
    run_manifest: Mapping[str, object],
    scheduler: object | None = None,
    scaler: object | None = None,
    metrics: Iterable[Mapping[str, object]] = (),
) -> Path:
    """Write all handoff files to a temporary sibling, then rename atomically."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"refusing to mutate immutable checkpoint {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        # The model-aware API records tied-weight aliases in metadata, avoiding
        # a duplicate embedding matrix while preserving strict reload behavior.
        save_model(model, str(temporary / "model.safetensors"))
        config = _model_config(model)
        (temporary / "model_config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        trusted_state: dict[str, object] = dict(trainer_state)
        trusted_state["rng"] = capture_rng_state()
        if optimizer is not None:
            trusted_state["optimizer"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            trusted_state["scheduler"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            trusted_state["scaler"] = scaler.state_dict()
        torch.save(trusted_state, temporary / "trainer_state.pt")

        manifest = dict(run_manifest)
        manifest.setdefault("git_commit", _git_commit())
        manifest["model_config_sha256"] = hashlib.sha256(
            canonical_json(config).encode()
        ).hexdigest()
        manifest["model_sha256"] = file_sha256(temporary / "model.safetensors")
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # A checkpoint always contains the append-only metric stream, even if
        # no periodic metric was emitted before an early budget exit.
        metric_lines = "".join(
            json.dumps(dict(event), sort_keys=True, allow_nan=False) + "\n" for event in metrics
        )
        (temporary / "metrics.jsonl").write_text(metric_lines, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: object | None = None,
    expected: Mapping[str, object] | None = None,
    restore_rng: bool = True,
) -> dict[str, object]:
    """Load trusted-local trainer state after validating immutable identities."""

    source = Path(directory)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((source / "model_config.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    if config_hash != manifest.get("model_config_sha256") or config != _model_config(model):
        raise CheckpointCompatibilityError("model configuration does not match checkpoint")
    if file_sha256(source / "model.safetensors") != manifest.get("model_sha256"):
        raise CheckpointCompatibilityError("model weights checksum mismatch")
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise CheckpointCompatibilityError(f"manifest mismatch for {key}")
    missing, unexpected = load_model(model, source / "model.safetensors", strict=True, device="cpu")
    if missing or unexpected:  # Defensive: strict=True should raise before this branch.
        raise CheckpointCompatibilityError(
            f"checkpoint keys differ (missing={missing}, unexpected={unexpected})"
        )
    # weights_only=False is intentional: trainer_state.pt is explicitly a
    # trusted-local artifact containing Python and optimizer objects.
    state = torch.load(source / "trainer_state.pt", map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state["scaler"])
    if restore_rng and "rng" in state:
        restore_rng_state(state["rng"])
    return dict(state)
