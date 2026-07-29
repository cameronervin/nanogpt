"""Append-only structured metric logging."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


class MetricsLogger:
    """Write one flushed JSON object per line for crash-tolerant inspection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: Mapping[str, object]) -> None:
        if "step" not in metrics:
            raise ValueError("every metric event must contain a step")
        line = json.dumps(dict(metrics), sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
