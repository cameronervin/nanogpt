"""Atomic, fingerprint-validated stage checkpoints."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch import nn

from nanogpt.utils import canonical_json, sha256_file, sha256_text


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely continue the requested run."""


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
        torch.save(trusted_state, temporary / "trainer_state.pt")

        manifest = dict(run_manifest)
        manifest.setdefault("git_commit", _git_commit())
        manifest["model_config_sha256"] = sha256_text(canonical_json(config))
        manifest["model_sha256"] = sha256_file(temporary / "model.safetensors")
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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
    expected: Mapping[str, object] | None = None,
    restore_rng: bool = True,
) -> dict[str, object]:
    """Load trusted-local trainer state after validating immutable identities."""

    source = Path(directory)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((source / "model_config.json").read_text(encoding="utf-8"))
    config_hash = sha256_text(canonical_json(config))
    if config_hash != manifest.get("model_config_sha256") or config != _model_config(model):
        raise CheckpointCompatibilityError("model configuration does not match checkpoint")
    if sha256_file(source / "model.safetensors") != manifest.get("model_sha256"):
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
    if restore_rng and "rng" in state:
        restore_rng_state(state["rng"])
    return dict(state)


def read_manifest(directory: str | Path) -> dict[str, object]:
    return dict(json.loads((Path(directory) / "run_manifest.json").read_text(encoding="utf-8")))
