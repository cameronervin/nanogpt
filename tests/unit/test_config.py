"""Tests for strict configuration and reproducibility primitives."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch

from nanogpt.config import ArtifactPaths, ModelConfig, canonical_json, seed_everything


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
        ("tie_embeddings", 1, "must be a boolean"),
        ("qk_norm", "yes", "must be a boolean"),
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


def test_yaml_loader_accepts_model_wrapper(tmp_path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "model:\n"
        "  vocab_size: 256\n  max_seq_len: 32\n  n_layers: 2\n"
        "  d_model: 64\n  n_query_heads: 4\n  n_kv_heads: 2\n"
        "  ffn_hidden_size: 176\n",
        encoding="utf-8",
    )
    assert ModelConfig.from_yaml(path).head_dim == 16


@pytest.mark.parametrize(
    ("profile", "expected_parameters"),
    [("tiny", 11_340_032), ("gpt_50m", 50_606_592), ("gpt_100m", 100_682_496)],
)
def test_documented_profiles_load_and_have_expected_parameter_counts(
    profile: str, expected_parameters: int
) -> None:
    path = Path(__file__).parents[2] / "configs" / "model" / f"{profile}.yaml"
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


def test_artifact_paths_create_documented_layout(tmp_path) -> None:
    paths = ArtifactPaths.from_environment(tmp_path)
    paths.create()
    expected = [
        paths.tokenizer,
        paths.datasets / "base",
        paths.datasets / "sft",
        paths.datasets / "grpo",
        paths.checkpoints / "base",
        paths.checkpoints / "sft",
        paths.checkpoints / "grpo",
        paths.evaluations,
        paths.logs,
    ]
    assert all(path.is_dir() for path in expected)
