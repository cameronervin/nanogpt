"""Measure base-training throughput on the target GPU and project its cost."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from nanogpt.config import ModelConfig, resolve_model_path
from nanogpt.model.gpt import GPT
from nanogpt.train.budget import project_cost_usd, select_profile
from nanogpt.train.optim import build_adamw
from nanogpt.train.trainer import RunContext


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_tokens_per_second(ctx: RunContext, model_config: ModelConfig) -> float:
    """Time real optimizer steps on random tokens with the base batch shape."""

    probe, base = ctx.config.probe, ctx.config.base
    assert probe is not None
    model = GPT(model_config).to(ctx.device)
    forward = torch.compile(model) if ctx.config.compile else model
    optimizer = build_adamw(
        model,
        learning_rate=base.learning_rate,
        weight_decay=base.weight_decay,
        fused=ctx.device.type == "cuda",
    )
    shape = (base.micro_batch_size, base.sequence_length)
    tokens = torch.randint(0, model_config.vocab_size, shape, device=ctx.device)
    started = 0.0
    for step in range(probe.warmup_steps + probe.measure_steps):
        if step == probe.warmup_steps:
            _synchronize(ctx.device)
            started = time.perf_counter()
        for _ in range(base.gradient_accumulation_steps):
            with ctx.autocast():
                loss = forward(tokens, targets=tokens).loss
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _synchronize(ctx.device)
    elapsed = time.perf_counter() - started
    per_step = base.micro_batch_size * base.sequence_length * base.gradient_accumulation_steps
    return probe.measure_steps * per_step / elapsed


def run_probe(ctx: RunContext, config_path: Path) -> dict[str, Any]:
    probe, base = ctx.config.probe, ctx.config.base
    if probe is None:
        raise ValueError("this run config has no probe section")
    rate = ctx.config.budget.hourly_rate_usd
    total_tokens = (
        base.max_steps
        * base.micro_batch_size
        * base.sequence_length
        * base.gradient_accumulation_steps
    )
    results: dict[str, Any] = {}
    costs: dict[str, float] = {}
    for candidate in probe.candidates:
        config = ModelConfig.from_yaml(resolve_model_path(candidate, config_path))
        throughput = measure_tokens_per_second(ctx, config)
        entry: dict[str, Any] = {
            "parameters": GPT(config).parameter_count(),
            "tokens_per_second": throughput,
            "base_hours": total_tokens / throughput / 3600,
        }
        if rate is not None:
            costs[candidate] = project_cost_usd(
                total_tokens, throughput, rate, safety_margin=probe.safety_margin
            )
            entry["projected_base_cost_usd"] = costs[candidate]
        results[candidate] = entry
    selected = select_profile(costs, cap_usd=base.budget_usd) if costs else None
    return {
        "base_tokens": total_tokens,
        "base_budget_usd": base.budget_usd,
        "candidates": results,
        "recommended_model": selected,
    }
