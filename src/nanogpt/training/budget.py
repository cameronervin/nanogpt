"""Measured-cost projection and a hard per-stage wall-clock watchdog."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised before the next step would exceed a paid-stage budget."""


def project_cost_usd(
    work_items: int,
    items_per_second: float,
    hourly_rate_usd: float,
    *,
    safety_margin: float = 0.20,
) -> float:
    if work_items < 0 or items_per_second <= 0 or hourly_rate_usd < 0 or safety_margin < 0:
        raise ValueError("cost projection inputs are outside their valid range")
    hours = work_items / items_per_second / 3600
    return hours * hourly_rate_usd * (1 + safety_margin)


def select_profile(primary_cost_usd: float, *, compute_cap_usd: float = 90.0) -> str:
    """Apply the documented deterministic 100M/50M throughput decision."""

    if primary_cost_usd < 0 or compute_cap_usd <= 0:
        raise ValueError("costs must be non-negative and cap positive")
    return "gpt_100m" if primary_cost_usd <= compute_cap_usd else "gpt_50m"


@dataclass
class BudgetGuard:
    """Track cost using monotonic time and reserve time for a clean checkpoint."""

    hourly_rate_usd: float
    stage_budget_usd: float
    checkpoint_reserve_seconds: float = 60.0
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.hourly_rate_usd < 0 or self.stage_budget_usd < 0:
            raise ValueError("hourly rate and stage budget must be non-negative")
        if self.checkpoint_reserve_seconds < 0:
            raise ValueError("checkpoint reserve cannot be negative")
        self.started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started_at)

    @property
    def estimated_cost_usd(self) -> float:
        return self.elapsed_seconds / 3600 * self.hourly_rate_usd

    @property
    def remaining_seconds(self) -> float:
        if self.hourly_rate_usd == 0:
            return float("inf")
        allowance = self.stage_budget_usd / self.hourly_rate_usd * 3600
        return max(0.0, allowance - self.elapsed_seconds)

    def check(self, *, next_step_seconds: float = 0.0) -> None:
        if next_step_seconds < 0:
            raise ValueError("next_step_seconds cannot be negative")
        required = next_step_seconds + self.checkpoint_reserve_seconds
        if self.remaining_seconds <= required:
            raise BudgetExceeded(
                f"stage budget guard reached at ${self.estimated_cost_usd:.2f}; checkpoint and exit"
            )
