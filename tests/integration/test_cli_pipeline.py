from __future__ import annotations

import json

import pytest

from nanogpt.commands.cli import main


def test_pipeline_dry_run_prints_resolved_plan(capsys) -> None:
    assert main(["pipeline", "--config", "configs/run/local_smoke.yaml", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["model_parameters"] > 0
    assert output["summary"]["projected_base_tokens"] == 2560
    assert output["resolved"]["offline_fixture"] is True


def test_cloud_environment_overrides_are_applied(monkeypatch, capsys) -> None:
    monkeypatch.setenv("NANOGPT_HOURLY_RATE_USD", "2.99")
    monkeypatch.setenv("NANOGPT_TOTAL_BUDGET_USD", "99")
    assert main(["pipeline", "--config", "configs/run/cloud_100.yaml", "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["hourly_rate_usd"] == 2.99
    assert output["summary"]["budget_usd"] == 99.0


def test_run_config_rejects_unknown_fields(tmp_path, capsys) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("run_name: bad\nseed: 1\ndevice: cpu\nmisspelled: true\n")
    with pytest.raises(SystemExit) as error:
        main(["pipeline", "--config", str(config), "--dry-run"])
    assert error.value.code == 2
    assert "unknown run config fields" in capsys.readouterr().err
