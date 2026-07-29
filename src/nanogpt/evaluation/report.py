"""Machine-readable comparison report and release-gate evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


def _metric(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"metric {key} must be numeric")
    return float(value)


@dataclass(frozen=True)
class ReleaseGates:
    base_loss_finite: bool
    sft_format_rate_at_least_95pct: bool
    rl_exact_match_gain_at_least_10pts: bool
    within_budget: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())


def build_report(
    *,
    base: Mapping[str, object],
    sft: Mapping[str, object],
    grpo: Mapping[str, object],
    run: Mapping[str, object],
) -> dict[str, object]:
    """Build and gate the documented base/SFT/GRPO comparison."""

    base_loss = _metric(base, "loss")
    sft_format = _metric(sft, "format_valid_rate")
    sft_exact = _metric(sft, "exact_match")
    grpo_exact = _metric(grpo, "exact_match")
    cost = _metric(run, "estimated_cost_usd")
    budget = _metric(run, "budget_usd")
    gates = ReleaseGates(
        base_loss_finite=base_loss == base_loss and abs(base_loss) != float("inf"),
        sft_format_rate_at_least_95pct=sft_format >= 0.95,
        rl_exact_match_gain_at_least_10pts=grpo_exact - sft_exact >= 0.10,
        within_budget=cost <= budget,
    )
    return {
        "format_version": 1,
        "base": dict(base),
        "sft": dict(sft),
        "grpo": dict(grpo),
        "run": dict(run),
        "release_gates": {**asdict(gates), "passed": gates.passed},
    }


def write_report(path: str | Path, report: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
