"""The one training loop shared by base, SFT, and GRPO.

A stage supplies a ``step_fn`` that performs one optimizer update at a given
learning rate; the ``Trainer`` owns everything around it: schedule, metrics,
evaluation cadence, atomic checkpoints, exact resume, and the budget guard.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from nanogpt.config import ModelConfig, RunConfig, RunPaths, StageConfig
from nanogpt.data.chat import TokenBatch
from nanogpt.train.budget import BudgetExceeded, BudgetGuard
from nanogpt.train.checkpoint import load_checkpoint, read_manifest, save_checkpoint
from nanogpt.train.optim import build_adamw, cosine_lr
from nanogpt.utils import resolve_device, resolve_dtype

Status = Literal["complete", "budget_exhausted"]
STATUS_FILE = "STATUS.json"
StepFn = Callable[[float], dict[str, float]]
DataState = tuple[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]]


@dataclass
class RunContext:
    """Everything a stage needs about the run, resolved once by the CLI."""

    config: RunConfig
    model_config: ModelConfig
    paths: RunPaths

    @cached_property
    def device(self) -> torch.device:
        # Resolved lazily so a cloud config can be dry-run on a laptop.
        return resolve_device(self.config.device)

    @property
    def dtype(self) -> torch.dtype:
        return resolve_dtype(self.config.dtype)

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """BF16 autocast when configured; parameters and optimizer stay FP32."""

        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)


class MetricsLogger:
    """Append one flushed JSON object per line for crash-tolerant inspection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def stage_status(paths: RunPaths, stage: str) -> dict[str, Any] | None:
    """Return ``{"status", "checkpoint", "step"}`` for a finished stage, else ``None``."""

    path = paths.checkpoints(stage) / STATUS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def finished_checkpoint(paths: RunPaths, stage: str) -> Path:
    """The checkpoint a child stage (or evaluation) starts from."""

    status = stage_status(paths, stage)
    if status is None:
        raise FileNotFoundError(f"stage {stage!r} has not finished; run it first")
    return paths.checkpoints(stage) / str(status["checkpoint"])


def latest_checkpoint(directory: Path) -> Path | None:
    steps = sorted(path for path in directory.glob("step_*") if path.is_dir())
    return steps[-1] if steps else None


@dataclass
class TrainResult:
    status: Status
    step: int
    checkpoint: Path


class Trainer:
    def __init__(
        self,
        ctx: RunContext,
        stage: str,
        config: StageConfig,
        model: nn.Module,
        *,
        lineage: dict[str, Any],
        evaluate: Callable[[], dict[str, float]] | None = None,
        data_state: DataState | None = None,
    ) -> None:
        self.ctx = ctx
        self.stage = stage
        self.config = config
        self.model = model.to(ctx.device)
        # torch.compile wraps the forward; checkpoints always use the raw module.
        compile_model = ctx.config.compile if config.compile is None else config.compile
        self.forward_model: nn.Module = (
            torch.compile(self.model) if compile_model else self.model  # type: ignore[assignment]
        )
        self.optimizer = build_adamw(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=ctx.device.type == "cuda",
        )
        # Identity fields that must match on resume: parent, tokenizer, and data hashes.
        self.lineage = lineage
        self.evaluate = evaluate
        self.data_state = data_state
        self.directory = ctx.paths.checkpoints(stage)
        self.metrics = MetricsLogger(ctx.paths.logs / f"{stage}.jsonl")
        self.step = 0
        self.tokens = 0
        self.elapsed_before_resume = 0.0
        self._saved_step: int | None = None
        rate = ctx.config.budget.hourly_rate_usd
        self.guard = (
            BudgetGuard(
                rate,
                config.budget_usd,
                checkpoint_reserve_seconds=ctx.config.budget.checkpoint_reserve_seconds,
            )
            if rate is not None and config.budget_usd > 0
            else None
        )

    # ------------------------------------------------------------------ steps

    def learning_rate(self, step: int) -> float:
        return cosine_lr(
            step,
            max_steps=self.config.max_steps,
            peak=self.config.learning_rate,
            warmup_steps=round(self.config.warmup_fraction * self.config.max_steps),
            min_fraction=self.config.min_lr_fraction,
        )

    def set_learning_rate(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def clip_and_step(self) -> float:
        """Clip, check, and apply the accumulated gradient; return its norm."""

        norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip))
        # One host sync per optimizer step; never apply a non-finite update.
        if not math.isfinite(norm):
            raise FloatingPointError(f"{self.stage}: gradient norm is non-finite")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def lm_step(self, next_batch: Callable[[], TokenBatch]) -> StepFn:
        """The standard next-token step with gradient accumulation."""

        accumulation = self.config.gradient_accumulation_steps

        def step(lr: float) -> dict[str, float]:
            self.set_learning_rate(lr)
            loss_sum = torch.zeros((), device=self.ctx.device)
            tokens = 0
            # Async host->device copies are only safe from pinned memory on CUDA;
            # on MPS they can race with the source tensor being freed.
            non_blocking = self.ctx.device.type == "cuda"
            for _ in range(accumulation):
                batch = next_batch()
                inputs = batch.input_ids.to(self.ctx.device, non_blocking=non_blocking)
                targets = batch.targets.to(self.ctx.device, non_blocking=non_blocking)
                with self.ctx.autocast():
                    loss = self.forward_model(inputs, targets=targets).loss
                (loss / accumulation).backward()
                loss_sum += loss.detach()
                tokens += inputs.numel()
            grad_norm = self.clip_and_step()
            return {
                "loss": float(loss_sum) / accumulation,
                "grad_norm": grad_norm,
                "tokens": tokens,
            }

        return step

    # ------------------------------------------------------------------ lifecycle

    def resume(self) -> bool:
        """Restore the latest checkpoint of this stage; return whether one existed."""

        checkpoint = latest_checkpoint(self.directory)
        if checkpoint is None:
            return False
        state = load_checkpoint(
            checkpoint, model=self.model, optimizer=self.optimizer, expected=self.lineage
        )
        self.step = int(state["step"])  # type: ignore[call-overload]
        self.tokens = int(state["tokens"])  # type: ignore[call-overload]
        manifest = read_manifest(checkpoint)
        self.elapsed_before_resume = float(manifest["wall_time_seconds"])  # type: ignore[arg-type]
        if self.data_state is not None and state.get("data") is not None:
            self.data_state[1](state["data"])  # type: ignore[arg-type]
        self._saved_step = self.step
        return True

    def prepare(self, *, resume: bool) -> TrainResult | None:
        """Apply resume policy; return a result if the stage is already finished.

        Checkpoints are immutable, so re-running a stage without ``--resume``
        when it has checkpoints is an error rather than a silent overwrite.
        """

        status = stage_status(self.ctx.paths, self.stage)
        if status is not None:
            if not resume:
                raise FileExistsError(
                    f"stage {self.stage!r} already finished; pass --resume to skip it"
                )
            return TrainResult(
                status["status"], status["step"], self.directory / status["checkpoint"]
            )
        if latest_checkpoint(self.directory) is not None:
            if not resume:
                raise FileExistsError(
                    f"stage {self.stage!r} has partial checkpoints; pass --resume to continue"
                )
            self.resume()
        return None

    def _elapsed(self, started: float) -> float:
        return self.elapsed_before_resume + time.monotonic() - started

    def _save(self, started: float) -> Path:
        path = self.directory / f"step_{self.step:06d}"
        if self._saved_step == self.step:
            return path
        rate = self.ctx.config.budget.hourly_rate_usd or 0.0
        elapsed = self._elapsed(started)
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            trainer_state={
                "step": self.step,
                "tokens": self.tokens,
                "data": None if self.data_state is None else self.data_state[0](),
            },
            run_manifest={
                **self.lineage,
                "stage": self.stage,
                "step": self.step,
                "tokens": self.tokens,
                "device": str(self.ctx.device),
                "wall_time_seconds": elapsed,
                "estimated_cost_usd": elapsed / 3600 * rate,
                "resolved_config": self.ctx.config.to_dict(),
            },
        )
        self._saved_step = self.step
        self._prune()
        return path

    def _prune(self) -> None:
        """Delete all but the newest ``keep_checkpoints`` committed step directories.

        Runs only after a new checkpoint has been atomically committed, so the
        stage can never be left without a resumable checkpoint.
        """

        keep = self.ctx.config.keep_checkpoints
        if keep <= 0:
            return
        steps = sorted(path for path in self.directory.glob("step_*") if path.is_dir())
        for old in steps[:-keep]:
            shutil.rmtree(old)

    def _finish(self, status: Status, started: float) -> TrainResult:
        path = self._save(started)
        record = {"status": status, "checkpoint": path.name, "step": self.step}
        temporary = self.directory / f".{STATUS_FILE}.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.directory / STATUS_FILE)
        self.metrics.log({"event": "finished", **record})
        return TrainResult(status, self.step, path)

    def fit(self, step_fn: StepFn) -> TrainResult:
        """Run optimizer steps until ``max_steps`` or the stage budget is spent."""

        self.model.train()
        started = time.monotonic()
        if self.guard is not None:
            # Time spent before a resume still counts against this stage's budget.
            self.guard.started_at = self.guard.clock() - self.elapsed_before_resume
        window_started, window_tokens = started, 0
        last_step_seconds = 0.0
        while self.step < self.config.max_steps:
            if self.guard is not None:
                try:
                    self.guard.check(next_step_seconds=last_step_seconds)
                except BudgetExceeded as error:
                    self.metrics.log(
                        {"event": "budget_exhausted", "step": self.step, "reason": str(error)}
                    )
                    return self._finish("budget_exhausted", started)
            step_started = time.monotonic()
            lr = self.learning_rate(self.step)
            metrics = step_fn(lr)
            self.step += 1
            self.tokens += int(metrics.get("tokens", 0))
            window_tokens += int(metrics.get("tokens", 0))
            last_step_seconds = time.monotonic() - step_started

            if not math.isfinite(metrics.get("loss", 0.0)):
                raise FloatingPointError(f"{self.stage}: loss is non-finite at step {self.step}")
            done = self.step == self.config.max_steps
            if self.step % self.config.log_every == 0 or done:
                now = time.monotonic()
                self.metrics.log(
                    {
                        **metrics,
                        "step": self.step,
                        "lr": lr,
                        "tokens_total": self.tokens,
                        "tokens_per_second": window_tokens / max(now - window_started, 1e-9),
                        "elapsed_seconds": self._elapsed(started),
                        "estimated_cost_usd": (
                            0.0 if self.guard is None else self.guard.estimated_cost_usd
                        ),
                    }
                )
                window_started, window_tokens = now, 0
            if (
                self.evaluate is not None
                and self.config.eval_every
                and (self.step % self.config.eval_every == 0 or done)
            ):
                self.metrics.log({"event": "eval", "step": self.step, **self.evaluate()})
                self.model.train()
            if self.config.checkpoint_every and self.step % self.config.checkpoint_every == 0:
                self._save(started)
        return self._finish("complete", started)
