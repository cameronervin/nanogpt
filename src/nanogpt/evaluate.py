"""Held-out metrics, chat-format and reasoning checks, and the stage report."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from nanogpt.data.chat import TokenBatch
from nanogpt.data.schemas import GRPOPrompt
from nanogpt.model.gpt import GPT, IGNORE_INDEX
from nanogpt.rewards import numeric_exact_reward, parse_numeric_answer

if TYPE_CHECKING:
    from nanogpt.tokenizer import Tokenizer
    from nanogpt.train.trainer import RunContext

ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>")


# --------------------------------------------------------------------------- language


@dataclass(frozen=True)
class LanguageMetrics:
    loss: float
    perplexity: float
    bits_per_byte: float | None
    active_tokens: int


@torch.inference_mode()
def evaluate_language_model(
    model: nn.Module,
    batches: Iterable[TokenBatch],
    *,
    device: torch.device | str = "cpu",
    byte_count: int | None = None,
) -> LanguageMetrics:
    """Token-weighted held-out loss; restores the model's train/eval mode."""

    was_training = model.training
    model.eval()
    loss_sum, active_tokens = 0.0, 0
    try:
        for batch in batches:
            targets = batch.targets.to(device)
            count = int(targets.ne(IGNORE_INDEX).sum())
            if count == 0:
                continue
            loss = model(batch.input_ids.to(device), targets=targets).loss
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("evaluation loss is missing or non-finite")
            loss_sum += float(loss) * count
            active_tokens += count
    finally:
        model.train(was_training)
    if active_tokens == 0:
        raise ValueError("evaluation contained no active target tokens")
    loss = loss_sum / active_tokens
    bits_per_byte = None
    if byte_count is not None:
        if byte_count <= 0:
            raise ValueError("byte_count must be positive")
        bits_per_byte = loss_sum / (math.log(2) * byte_count)
    perplexity = math.exp(loss) if loss < 80 else float("inf")
    return LanguageMetrics(loss, perplexity, bits_per_byte, active_tokens)


# --------------------------------------------------------------------------- chat format


@dataclass(frozen=True)
class ChatFormatMetrics:
    valid_termination_rate: float
    sample_count: int


def evaluate_chat_format(completions: Iterable[str]) -> ChatFormatMetrics:
    """Require exactly one final ``<|end|>`` and no role-marker injection."""

    values = list(completions)
    if not values:
        raise ValueError("chat format evaluation requires completions")
    valid = 0
    for completion in values:
        prefix = completion.rsplit("<|end|>", 1)[0]
        # A terminator in the middle is not valid termination: accepting a
        # trailing role injection would overstate chat-format compliance.
        valid += int(
            completion.count("<|end|>") == 1
            and completion.endswith("<|end|>")
            and not any(marker in prefix for marker in ROLE_MARKERS)
        )
    return ChatFormatMetrics(valid / len(values), len(values))


# --------------------------------------------------------------------------- reasoning


@dataclass(frozen=True)
class ReasoningMetrics:
    exact_match: float
    pass_at_k: float
    malformed_answer_rate: float
    prompt_count: int
    completions_per_prompt: int


def evaluate_reasoning(
    completions: Sequence[Sequence[str]], answers: Sequence[object]
) -> ReasoningMetrics:
    """``exact_match`` scores the first completion per prompt (greedy pass@1)."""

    if len(completions) != len(answers) or not answers:
        raise ValueError("one non-empty completion group is required per answer")
    widths = {len(group) for group in completions}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("all prompts must have the same nonzero completion count")
    pass_one = passed = malformed = total = 0.0
    for group, answer in zip(completions, answers, strict=True):
        rewards = [numeric_exact_reward(completion, answer) for completion in group]
        pass_one += rewards[0]
        passed += float(any(rewards))
        malformed += sum(parse_numeric_answer(completion) is None for completion in group)
        total += len(group)
    return ReasoningMetrics(
        exact_match=pass_one / len(answers),
        pass_at_k=passed / len(answers),
        malformed_answer_rate=malformed / total,
        prompt_count=len(answers),
        completions_per_prompt=next(iter(widths)),
    )


def chat_reasoning_eval(
    model: GPT, tokenizer: Tokenizer, prompts: Sequence[GRPOPrompt], *, max_new_tokens: int
) -> dict[str, float]:
    """Greedy pass@1, format validity, and malformed rate on held-out prompts."""

    from nanogpt.generate import sample_chat

    texts: list[str] = []
    rendered: list[str] = []
    for prompt in prompts:
        _, (completion,) = sample_chat(
            model, tokenizer, prompt.messages, num_samples=1, max_new_tokens=max_new_tokens
        )
        texts.append(tokenizer.decode(completion))
        rendered.append(tokenizer.decode(completion, render_special=True))
    reasoning = evaluate_reasoning([[text] for text in texts], [p.answer for p in prompts])
    chat = evaluate_chat_format(rendered)
    return {
        "exact_match": reasoning.exact_match,
        "malformed_answer_rate": reasoning.malformed_answer_rate,
        "format_valid_rate": chat.valid_termination_rate,
        "prompt_count": float(reasoning.prompt_count),
    }


# --------------------------------------------------------------------------- report


@dataclass(frozen=True)
class ReleaseGates:
    base_loss_finite: bool
    sft_format_rate_at_least_95pct: bool
    rl_exact_match_gain_at_least_10pts: bool
    rl_no_regression_on_easiest_bucket: bool
    within_budget: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())


def build_report(
    *,
    base: Mapping[str, float],
    sft: Mapping[str, float],
    grpo: Mapping[str, float],
    run: Mapping[str, float],
) -> dict[str, object]:
    """Build and gate the base/SFT/GRPO comparison.

    ``exact_match`` is the hardest arithmetic bucket; ``easiest_exact_match``
    (when present) is the easiest, which RL may not lower by more than 2 points.
    """

    easiest = "easiest_exact_match"
    gates = ReleaseGates(
        base_loss_finite=math.isfinite(base["loss"]),
        sft_format_rate_at_least_95pct=sft["format_valid_rate"] >= 0.95,
        rl_exact_match_gain_at_least_10pts=grpo["exact_match"] - sft["exact_match"] >= 0.10,
        rl_no_regression_on_easiest_bucket=grpo.get(easiest, grpo["exact_match"])
        >= sft.get(easiest, sft["exact_match"]) - 0.02,
        within_budget=run["estimated_cost_usd"] <= run["budget_usd"],
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


# --------------------------------------------------------------------------- stage entry


def run_eval(ctx: RunContext) -> dict[str, object]:
    """Evaluate every finished stage checkpoint and write ``eval/report.json``."""

    from nanogpt.data.shards import TokenLoader
    from nanogpt.data.sources import arithmetic_prompts
    from nanogpt.tokenizer import load_tokenizer
    from nanogpt.train.checkpoint import load_checkpoint, read_manifest
    from nanogpt.train.trainer import finished_checkpoint

    config = ctx.config
    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    # A seed never used by any training source: held-out numbers and order.
    buckets = sorted(set(config.eval.arithmetic_digits))
    if not buckets:
        raise ValueError("eval.arithmetic_digits must name at least one bucket")
    held_out = {
        digits: list(
            arithmetic_prompts(
                config.eval.reasoning_prompts, seed=987_654, min_digits=1, max_digits=digits
            )
        )
        for digits in buckets
    }
    results: dict[str, dict[str, float]] = {}
    cost = 0.0
    for stage in ("base", "sft", "grpo"):
        checkpoint = finished_checkpoint(ctx.paths, stage)
        model = GPT(ctx.model_config)
        load_checkpoint(checkpoint, model=model, restore_rng=False)
        model.to(ctx.device).eval()
        cost += float(read_manifest(checkpoint)["estimated_cost_usd"])  # type: ignore[arg-type]
        if stage == "base":
            loader = TokenLoader(
                ctx.paths.data("base"),
                split="validation",
                batch_size=config.base.micro_batch_size,
                seq_len=config.base.sequence_length,
            )
            batches = [loader.next_batch() for _ in range(config.eval.validation_batches)]
            byte_count = sum(
                len(tokenizer.decode(batch.targets.flatten().tolist()).encode("utf-8"))
                for batch in batches
            )
            metrics = evaluate_language_model(
                model, batches, device=ctx.device, byte_count=byte_count
            )
            results[stage] = {
                key: float(value) for key, value in asdict(metrics).items() if value is not None
            }
        else:
            per_bucket = {
                digits: chat_reasoning_eval(
                    model, tokenizer, prompts, max_new_tokens=config.eval.max_new_tokens
                )
                for digits, prompts in held_out.items()
            }
            results[stage] = {
                **per_bucket[buckets[-1]],
                "easiest_exact_match": per_bucket[buckets[0]]["exact_match"],
                **{
                    f"d{digits}_{key}": value
                    for digits, metrics in per_bucket.items()
                    for key, value in metrics.items()
                },
            }
    report = build_report(
        base=results["base"],
        sft=results["sft"],
        grpo=results["grpo"],
        run={"estimated_cost_usd": cost, "budget_usd": config.budget.total_usd},
    )
    write_report(ctx.paths.eval / "report.json", report)
    return report
