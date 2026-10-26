from __future__ import annotations

import json

import pytest

from nanogpt.cli import main

pytestmark = pytest.mark.integration


def _json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.startswith('{"')]


def test_dry_run_prints_resolved_plan(capsys) -> None:
    assert main(["pipeline", "--config", "configs/runs/smoke.yaml", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["model_parameters"] > 0
    assert output["summary"]["base_tokens"] == 20 * 4 * 64
    assert output["config"]["data"]["source"] == "fixture"


def test_paid_runs_need_a_rate_and_confirmation(monkeypatch, capsys) -> None:
    monkeypatch.delenv("NANOGPT_HOURLY_RATE_USD", raising=False)
    assert main(["pipeline", "--config", "configs/runs/cloud.yaml"]) == 2
    assert "hourly_rate_usd" in capsys.readouterr().err
    monkeypatch.setenv("NANOGPT_HOURLY_RATE_USD", "2.99")
    assert main(["pipeline", "--config", "configs/runs/cloud.yaml"]) == 2
    assert "--yes" in capsys.readouterr().err


def test_run_config_rejects_unknown_fields(tmp_path, capsys) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("name: bad\nseed: 1\nmodel: tiny\nmisspelled: true\n")
    with pytest.raises(SystemExit) as error:
        main(["pipeline", "--config", str(config), "--dry-run"])
    assert error.value.code == 2
    assert "unknown run fields: misspelled" in capsys.readouterr().err


def test_smoke_pipeline_runs_offline_end_to_end_and_resumes(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("NANOGPT_HOME", str(tmp_path))
    args = ["pipeline", "--config", "configs/runs/smoke.yaml"]
    assert main(args) == 0
    stages = [line for line in _json_lines(capsys.readouterr().out) if "stage" in line]
    assert [(line["stage"], line["status"]) for line in stages] == [
        ("base", "complete"),
        ("sft", "complete"),
        ("grpo", "complete"),
    ]
    report = json.loads((tmp_path / "runs/smoke/eval/report.json").read_text())
    assert report["release_gates"]["base_loss_finite"] is True

    # Immutable checkpoints: a rerun needs --resume, which skips finished stages.
    with pytest.raises(SystemExit):
        main(args)
    capsys.readouterr()
    assert main([*args, "--resume"]) == 0
    resumed = [line for line in _json_lines(capsys.readouterr().out) if "stage" in line]
    assert [line["checkpoint"] for line in resumed] == [line["checkpoint"] for line in stages]
