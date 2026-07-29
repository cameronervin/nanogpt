from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from nanogpt.config import ModelConfig
from nanogpt.model.gpt import GPT
from nanogpt.training.budget import BudgetExceeded, BudgetGuard, project_cost_usd, select_profile
from nanogpt.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=1,
        d_model=16,
        n_query_heads=2,
        n_kv_heads=1,
        ffn_hidden_size=32,
    )


def test_budget_projection_profile_and_deadline() -> None:
    assert project_cost_usd(3600, 1, 2, safety_margin=0) == 2
    assert select_profile(90) == "gpt_100m"
    assert select_profile(90.01) == "gpt_50m"
    now = [0.0]
    guard = BudgetGuard(3.6, 0.01, checkpoint_reserve_seconds=1, clock=lambda: now[0])
    now[0] = 9.5
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_atomic_checkpoint_roundtrip_and_compatibility(tmp_path) -> None:
    torch.manual_seed(7)
    model = GPT(_config())
    optimizer = torch.optim.AdamW(model.parameters())
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "step-1"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        trainer_state={"step": 1},
        run_manifest={"stage": "base", "tokenizer_hash": "abc"},
        metrics=[{"step": 1, "loss": 3.0}],
    )
    assert {path.name for path in checkpoint.iterdir()} == {
        "metrics.jsonl",
        "model.safetensors",
        "model_config.json",
        "run_manifest.json",
        "trainer_state.pt",
    }
    assert json.loads((checkpoint / "metrics.jsonl").read_text())["loss"] == 3.0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected={"tokenizer_hash": "abc"},
        restore_rng=False,
    )
    assert state["step"] == 1
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(checkpoint, model=model, expected={"tokenizer_hash": "wrong"})


def test_checkpoint_restores_numpy_rng(tmp_path) -> None:
    np.random.seed(123)
    model = GPT(_config())
    checkpoint = tmp_path / "rng"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        trainer_state={"step": 0},
        run_manifest={"stage": "base"},
    )
    expected = np.random.random(4)
    np.random.random(10)
    load_checkpoint(checkpoint, model=model, restore_rng=True)
    assert np.array_equal(np.random.random(4), expected)
