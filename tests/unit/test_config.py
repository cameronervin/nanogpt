"""Tests for strict configuration and reproducibility primitives."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch
import yaml

from nanogpt.config import ModelConfig, RunConfig, RunPaths, from_mapping, load_run_config
from nanogpt.utils import canonical_json, seed_everything

CONFIGS = Path(__file__).parents[2] / "configs"


def valid_values() -> dict[str, object]:
    return {
        "vocab_size": 256,
        "max_seq_len": 32,
        "n_layers": 2,
        "d_model": 64,
        "n_query_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_size": 176,
    }


def test_model_config_rejects_unknown_fields() -> None:
    values = valid_values() | {"n_layer": 2}
    with pytest.raises(ValueError, match=r"unknown.*n_layer"):
        ModelConfig.from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("d_model", 65, "divisible by n_query_heads"),
        ("n_query_heads", 3, "divisible by n_query_heads"),
        ("n_kv_heads", 3, "divisible by n_kv_heads"),
        ("vocab_size", 0, "positive integer"),
    ],
)
def test_model_config_validates_shape_invariants(field: str, value: int, message: str) -> None:
    values = valid_values() | {field: value}
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_mapping(values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("d_model", 60, "head dimension must be even"),
        ("rope_theta", math.nan, "finite positive"),
        ("rms_norm_eps", math.inf, "finite positive"),
        ("tie_embeddings", 1, "must be bool"),
        ("qk_norm", "yes", "must be bool"),
    ],
)
def test_model_config_rejects_values_that_are_not_canonical(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_mapping(valid_values() | {field: value})


def test_canonical_hash_is_independent_of_mapping_order() -> None:
    values = valid_values()
    reversed_values = dict(reversed(list(values.items())))
    assert (
        ModelConfig.from_mapping(values).fingerprint()
        == ModelConfig.from_mapping(reversed_values).fingerprint()
    )
    assert json.loads(ModelConfig.from_mapping(values).canonical_json()) == values | {
        "qk_norm": True,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "tie_embeddings": True,
    }


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"learning_rate": math.nan})


@pytest.mark.parametrize(
    ("profile", "expected_parameters"),
    [("tiny", 11_340_032), ("gpt_50m", 50_606_592), ("gpt_100m", 100_682_496)],
)
def test_documented_profiles_load_and_have_expected_parameter_counts(
    profile: str, expected_parameters: int
) -> None:
    path = CONFIGS / "models" / f"{profile}.yaml"
    config = ModelConfig.from_yaml(path)
    # Meta tensors validate the actual module graph without allocating 100M weights.
    with torch.device("meta"):
        from nanogpt.model import GPT

        model = GPT(config)

    assert model.parameter_count() == expected_parameters


def test_seed_everything_repeats_cpu_randomness() -> None:
    seed_everything(7)
    first = (random.random(), torch.rand(3))
    seed_everything(7)
    second = (random.random(), torch.rand(3))
    assert first[0] == second[0]
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


def test_run_paths_group_artifacts_by_run(tmp_path) -> None:
    paths = RunPaths.for_run("demo", tmp_path)
    assert paths.root == tmp_path.resolve() / "runs" / "demo"
    assert paths.checkpoints("sft") == paths.root / "checkpoints" / "sft"
    assert paths.data("base") == paths.root / "data" / "base"


@pytest.mark.parametrize("name", ["smoke", "cloud", "rehearsal", "rehearsal_gpu"])
def test_checked_in_run_configs_load(name: str) -> None:
    run, model = load_run_config(CONFIGS / "runs" / f"{name}.yaml")
    assert run.name == name
    assert model.vocab_size >= run.tokenizer.vocab_size or run.tokenizer.backend == "hf"


def _smoke() -> dict[str, object]:
    return yaml.safe_load((CONFIGS / "runs" / "smoke.yaml").read_text())


def test_run_config_rejects_unknown_nested_fields() -> None:
    values = _smoke()
    values["base"]["warmup_steps"] = 3  # type: ignore[index]
    with pytest.raises(ValueError, match=r"unknown run\.base fields: warmup_steps"):
        from_mapping(RunConfig, values, where="run")


def test_run_config_rejects_wrong_types_and_literals() -> None:
    values = _smoke()
    values["base"]["max_steps"] = "20"  # type: ignore[index]
    with pytest.raises(ValueError, match="max_steps must be int"):
        from_mapping(RunConfig, values, where="run")
    values = _smoke()
    values["dtype"] = "float8"
    with pytest.raises(ValueError, match="dtype must be one of"):
        from_mapping(RunConfig, values, where="run")


def test_stage_budgets_cannot_exceed_total() -> None:
    values = _smoke()
    values["budget"] = {"total_usd": 10.0}
    values["base"]["budget_usd"] = 11.0  # type: ignore[index]
    with pytest.raises(ValueError, match=r"exceed budget\.total_usd"):
        from_mapping(RunConfig, values, where="run")
    values["budget"] = {"total_usd": 101.0}
    with pytest.raises(ValueError, match="between 0 and 100"):
        from_mapping(RunConfig, values, where="run")


def test_environment_overrides_cost_inputs(monkeypatch) -> None:
    monkeypatch.setenv("NANOGPT_HOURLY_RATE_USD", "2.99")
    monkeypatch.setenv("NANOGPT_TOTAL_BUDGET_USD", "99")
    run, _ = load_run_config(CONFIGS / "runs" / "cloud.yaml")
    assert run.budget.hourly_rate_usd == 2.99
    assert run.budget.total_usd == 99.0
