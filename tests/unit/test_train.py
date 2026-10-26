from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nanogpt.config import ModelConfig, RunConfig, RunPaths, from_mapping
from nanogpt.data.chat import TokenBatch
from nanogpt.model import GPT
from nanogpt.train.budget import BudgetExceeded, BudgetGuard, project_cost_usd, select_profile
from nanogpt.train.checkpoint import CheckpointCompatibilityError, load_checkpoint, save_checkpoint
from nanogpt.train.optim import cosine_lr
from nanogpt.train.trainer import RunContext, Trainer, stage_status
from nanogpt.utils import seed_everything

SMOKE = Path(__file__).parents[2] / "configs" / "runs" / "smoke.yaml"


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        n_layers=1,
        d_model=16,
        n_query_heads=2,
        n_kv_heads=1,
        ffn_hidden_size=32,
    )


def _context(tmp_path: Path, budget: dict[str, float] | None = None, **base: object) -> RunContext:
    values = yaml.safe_load(SMOKE.read_text())
    values["device"] = "cpu"
    if budget is not None:
        values["budget"] = budget
    values["base"] = {
        "max_steps": 6,
        "micro_batch_size": 2,
        "sequence_length": 8,
        "learning_rate": 0.01,
        "gradient_accumulation_steps": 2,
        "warmup_fraction": 0.2,
        "log_every": 1,
        **base,
    }
    config = from_mapping(RunConfig, values, where="run")
    return RunContext(config, _model_config(), RunPaths.for_run("test", tmp_path))


class _Stream:
    """Deterministic batch source whose position is checkpointable state."""

    def __init__(self) -> None:
        self.index = 0

    def next_batch(self) -> TokenBatch:
        generator = torch.Generator().manual_seed(self.index)
        self.index += 1
        tokens = torch.randint(0, 32, (2, 9), generator=generator)
        return TokenBatch(tokens[:, :-1], tokens[:, 1:])

    def state(self) -> dict[str, int]:
        return {"index": self.index}

    def load(self, state: dict[str, int]) -> None:
        self.index = state["index"]


def _trainer(ctx: RunContext, stream: _Stream) -> Trainer:
    seed_everything(0)
    return Trainer(
        ctx,
        "base",
        ctx.config.base,
        GPT(ctx.model_config),
        lineage={"dataset_sha256": "x"},
        data_state=(stream.state, stream.load),
    )


def test_resumed_training_is_bitwise_identical_to_uninterrupted(tmp_path) -> None:
    straight_stream = _Stream()
    straight = _trainer(_context(tmp_path / "a"), straight_stream)
    straight.fit(straight.lm_step(straight_stream.next_batch))

    # Interrupted: crash during step 4, after the step-3 checkpoint was written.
    ctx = _context(tmp_path / "b", checkpoint_every=3)
    first_stream = _Stream()
    first = _trainer(ctx, first_stream)
    step = first.lm_step(first_stream.next_batch)

    def crashing_step(lr: float) -> dict[str, float]:
        if first.step == 3:
            raise KeyboardInterrupt
        return step(lr)

    with pytest.raises(KeyboardInterrupt):
        first.fit(crashing_step)
    assert stage_status(ctx.paths, "base") is None

    resumed_stream = _Stream()
    resumed = _trainer(ctx, resumed_stream)
    assert resumed.prepare(resume=True) is None and resumed.step == 3
    result = resumed.fit(resumed.lm_step(resumed_stream.next_batch))

    assert result.status == "complete" and result.step == 6
    for name, value in straight.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    assert stage_status(ctx.paths, "base") == {
        "status": "complete",
        "checkpoint": "step_000006",
        "step": 6,
    }
    # A finished stage is skipped with --resume and refused without it.
    assert _trainer(ctx, _Stream()).prepare(resume=True) is not None
    with pytest.raises(FileExistsError, match="--resume"):
        _trainer(ctx, _Stream()).prepare(resume=False)


def test_budget_guard_checkpoints_and_exits_cleanly(tmp_path) -> None:
    budget = {"hourly_rate_usd": 3.6, "total_usd": 1.0, "checkpoint_reserve_seconds": 1}
    ctx = _context(tmp_path, budget=budget, budget_usd=0.01)
    stream = _Stream()
    trainer = _trainer(ctx, stream)
    now = [0.0]
    assert trainer.guard is not None
    trainer.guard.clock = lambda: now[0]

    step = trainer.lm_step(stream.next_batch)

    def slow_step(lr: float) -> dict[str, float]:
        now[0] += 4.0  # $0.01 at $3.60/h is 10 s; the reserve is 1 s.
        return step(lr)

    result = trainer.fit(slow_step)
    assert result.status == "budget_exhausted"
    assert 0 < result.step < 6
    assert (result.checkpoint / "model.safetensors").exists()
    events = [json.loads(line) for line in (ctx.paths.logs / "base.jsonl").read_text().splitlines()]
    assert any(event.get("event") == "budget_exhausted" for event in events)


def test_cosine_schedule_warmup_and_floor() -> None:
    values = [
        cosine_lr(step, max_steps=10, peak=1.0, warmup_steps=2, min_fraction=0.1)
        for step in range(11)
    ]
    assert values[0] == 0.5 and values[1] == 1.0
    assert values[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in pairwise(values[1:]))


def test_budget_projection_and_profile_selection() -> None:
    assert project_cost_usd(3600, 1, 2, safety_margin=0) == 2
    assert select_profile({"gpt_100m": 70.0, "gpt_50m": 40.0}, cap_usd=62) == "gpt_50m"
    assert select_profile({"gpt_100m": 70.0}, cap_usd=62) is None
    now = [0.0]
    guard = BudgetGuard(3.6, 0.01, checkpoint_reserve_seconds=1, clock=lambda: now[0])
    now[0] = 9.5
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_atomic_checkpoint_roundtrip_and_compatibility(tmp_path) -> None:
    torch.manual_seed(7)
    model = GPT(_model_config())
    optimizer = torch.optim.AdamW(model.parameters())
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "step-1"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        trainer_state={"step": 1},
        run_manifest={"stage": "base", "tokenizer_sha256": "abc"},
    )
    assert {path.name for path in checkpoint.iterdir()} == {
        "model.safetensors",
        "model_config.json",
        "run_manifest.json",
        "trainer_state.pt",
    }
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model=model, optimizer=None, trainer_state={}, run_manifest={})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected={"tokenizer_sha256": "abc"},
        restore_rng=False,
    )
    assert state["step"] == 1
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(checkpoint, model=model, expected={"tokenizer_sha256": "wrong"})


def test_checkpoint_restores_numpy_rng(tmp_path) -> None:
    np.random.seed(123)
    model = GPT(_model_config())
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
