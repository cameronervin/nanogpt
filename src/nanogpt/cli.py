"""The ``nanogpt`` command line: one subcommand per pipeline step.

    tokenizer -> prepare -> train base -> train sft -> train grpo -> eval

``pipeline`` runs them in order and stops at the first stage that does not
complete. Every command accepts ``--dry-run`` to print the resolved plan, and
a run with a dollar budget refuses to start without ``--yes`` and a rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nanogpt.config import RunPaths, load_run_config
from nanogpt.model.gpt import GPT
from nanogpt.train.trainer import RunContext, TrainResult
from nanogpt.utils import seed_everything

STAGES = ("base", "sft", "grpo")


def _context(args: argparse.Namespace) -> RunContext:
    config, model_config = load_run_config(args.config)
    return RunContext(config, model_config, RunPaths.for_run(config.name))


def _summary(ctx: RunContext) -> dict[str, Any]:
    config = ctx.config
    base = config.base
    tokens_per_step = (
        base.micro_batch_size * base.sequence_length * base.gradient_accumulation_steps
    )
    import torch

    with torch.device("meta"):
        parameters = GPT(ctx.model_config).parameter_count()
    return {
        "run": config.name,
        "artifacts": str(ctx.paths.root),
        "device": config.device,
        "model_parameters": parameters,
        "base_tokens": base.max_steps * tokens_per_step,
        "hourly_rate_usd": config.budget.hourly_rate_usd,
        "total_budget_usd": config.budget.total_usd,
        "stage_budgets_usd": {
            "data": config.data.budget_usd,
            "base": base.budget_usd,
            "sft": config.sft.budget_usd,
            "grpo": config.grpo.budget_usd,
            "eval": config.eval.budget_usd,
        },
    }


def _authorize(ctx: RunContext, args: argparse.Namespace) -> int | None:
    """Print the plan; return an exit code if the command must not proceed."""

    print(json.dumps({"config": ctx.config.to_dict(), "summary": _summary(ctx)}, indent=2))
    if args.dry_run:
        return 0
    if ctx.config.paid:
        if ctx.config.budget.hourly_rate_usd is None:
            print(
                "Paid runs require budget.hourly_rate_usd (or NANOGPT_HOURLY_RATE_USD).",
                file=sys.stderr,
            )
            return 2
        if not args.yes:
            print(
                "Refusing to spend money without --yes (use --dry-run to inspect).", file=sys.stderr
            )
            return 2
    seed_everything(ctx.config.seed, deterministic=ctx.config.deterministic)
    return None


def _report(result: TrainResult, stage: str) -> int:
    print(
        json.dumps(
            {
                "stage": stage,
                "status": result.status,
                "step": result.step,
                "checkpoint": str(result.checkpoint),
            }
        )
    )
    return 0 if result.status == "complete" else 3


def _train_stage(ctx: RunContext, stage: str, resume: bool) -> TrainResult:
    from nanogpt.train.base import run_base
    from nanogpt.train.grpo import run_grpo
    from nanogpt.train.sft import run_sft

    runners: dict[str, Callable[..., TrainResult]] = {
        "base": run_base,
        "sft": run_sft,
        "grpo": run_grpo,
    }
    return runners[stage](ctx, resume=resume)


def cmd_tokenizer(ctx: RunContext, args: argparse.Namespace) -> int:
    from nanogpt.data.prepare import build_tokenizer

    tokenizer = build_tokenizer(ctx)
    print(
        json.dumps(
            {
                "tokenizer": str(ctx.paths.tokenizer),
                "vocab_size": tokenizer.vocab_size,
                "sha256": tokenizer.sha256,
            }
        )
    )
    return 0


def cmd_prepare(ctx: RunContext, args: argparse.Namespace) -> int:
    from nanogpt.data.prepare import prepare_data

    print(json.dumps({"prepared": prepare_data(ctx)}))
    return 0


def cmd_train(ctx: RunContext, args: argparse.Namespace) -> int:
    return _report(_train_stage(ctx, args.stage, args.resume), args.stage)


def cmd_eval(ctx: RunContext, args: argparse.Namespace) -> int:
    from nanogpt.evaluate import run_eval

    report = run_eval(ctx)
    print(json.dumps(report, indent=2))
    return 0


def cmd_probe(ctx: RunContext, args: argparse.Namespace) -> int:
    from nanogpt.train.probe import run_probe

    print(json.dumps(run_probe(ctx, Path(args.config)), indent=2))
    return 0


def cmd_pipeline(ctx: RunContext, args: argparse.Namespace) -> int:
    cmd_tokenizer(ctx, args)
    cmd_prepare(ctx, args)
    for stage in STAGES:
        code = _report(_train_stage(ctx, stage, args.resume), stage)
        if code:
            # Never start the next stage after a budget exit or failure.
            return code
    return cmd_eval(ctx, args)


def cmd_sample(ctx: RunContext, args: argparse.Namespace) -> int:
    from nanogpt.generate import sample_chat
    from nanogpt.tokenizer import load_tokenizer
    from nanogpt.train.checkpoint import load_checkpoint
    from nanogpt.train.trainer import finished_checkpoint

    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    model = GPT(ctx.model_config)
    load_checkpoint(finished_checkpoint(ctx.paths, args.stage), model=model, restore_rng=False)
    model.to(ctx.device)
    _, completions = sample_chat(
        model,
        tokenizer,
        [{"role": "user", "content": args.prompt}],
        num_samples=args.samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    for completion in completions:
        print(tokenizer.decode(completion))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanogpt", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler: Callable[..., int], help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--config", required=True, help="run config, e.g. configs/runs/smoke.yaml"
        )
        command.add_argument("--dry-run", action="store_true", help="print the plan and exit")
        command.add_argument("--yes", action="store_true", help="confirm a paid run")
        command.set_defaults(handler=handler)
        return command

    add("tokenizer", cmd_tokenizer, "train (or reuse) the run's tokenizer")
    add("prepare", cmd_prepare, "write base shards, SFT conversations, and GRPO prompts")
    train = add("train", cmd_train, "run one training stage")
    train.add_argument("stage", choices=STAGES)
    train.add_argument("--resume", action="store_true", help="continue or skip a finished stage")
    add("eval", cmd_eval, "evaluate all stages and write eval/report.json")
    add("probe", cmd_probe, "measure GPU throughput and project base-training cost")
    pipeline = add("pipeline", cmd_pipeline, "tokenizer -> prepare -> base -> sft -> grpo -> eval")
    pipeline.add_argument("--resume", action="store_true", help="continue or skip finished stages")
    sample = add("sample", cmd_sample, "chat with a finished stage checkpoint")
    sample.add_argument("--stage", choices=STAGES, default="sft")
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--samples", type=int, default=1)
    sample.add_argument("--max-new-tokens", type=int, default=128)
    sample.add_argument("--temperature", type=float, default=0.0)
    sample.add_argument("--top-k", type=int, default=None)
    sample.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and turn validation failures into concise user errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ctx = _context(args)
        if args.handler is not cmd_sample and (code := _authorize(ctx, args)) is not None:
            return code
        return int(args.handler(ctx, args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


def run() -> None:  # pragma: no cover - process-level behaviour
    """Console entry point: run, flush, then exit without interpreter teardown.

    Abandoning a Hugging Face streaming dataset part-way (we always stop at a
    token or character budget) deadlocks pyarrow's thread-pool destructor at
    interpreter shutdown (pyarrow 25 / datasets 4.8). All artifacts are already
    fsynced and atomically renamed when ``main`` returns, so skipping teardown
    is safe.
    """

    try:
        code = main()
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":  # pragma: no cover
    run()
