"""The ``nanogpt`` command-line interface.

The local pipeline is intentionally self-contained and offline.  Cloud configs
share the same interface, but a paid launch is blocked until its explicit
hourly rate and prepared dataset artifacts are supplied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
import yaml

from nanogpt.config import ArtifactPaths, ModelConfig, canonical_json, seed_everything
from nanogpt.data.schemas import ConversationRecord, FineWebRecord
from nanogpt.data.sft import SFTBatch, collate_sft
from nanogpt.data.shards import prepare_shards
from nanogpt.data.synthetic import arithmetic_conversations
from nanogpt.data.tokenizer import REFERENCE_TRAINER_MAX_BYTES, ByteBPETokenizer
from nanogpt.model.gpt import GPT
from nanogpt.training.base import train_base
from nanogpt.training.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from nanogpt.training.common import TrainLoopConfig, causal_batches
from nanogpt.training.grpo import (
    grpo_update,
    numeric_exact_reward,
    sequence_log_probs,
)
from nanogpt.training.sft import train_sft


def _read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a mapping")
    return value


_RUN_FIELDS = {
    "artifact_root",
    "base",
    "budget_poll_seconds",
    "compile",
    "data",
    "deadline_checkpoint_margin_seconds",
    "device",
    "distributed",
    "dtype",
    "evaluation",
    "grpo",
    "hourly_rate_usd",
    "model_candidates",
    "model_config",
    "num_workers",
    "offline_fixture",
    "probe_tokens",
    "profile_probe_config",
    "projection_safety_margin",
    "run_name",
    "seed",
    "sft",
    "storage_budget_usd",
    "total_budget_usd",
    "warmup_steps",
    "measurement_steps",
    "maximum_projected_compute_usd",
    "stage_budget_usd",
}
_SECTION_FIELDS = {
    "base": {
        "checkpoint_interval",
        "eval_interval",
        "gradient_accumulation_steps",
        "gradient_clip_norm",
        "learning_rate",
        "max_steps",
        "micro_batch_size",
        "minimum_learning_rate_fraction",
        "sequence_length",
        "stage_budget_usd",
        "target_tokens_100m",
        "target_tokens_50m",
        "warmup_fraction",
        "weight_decay",
    },
    "sft": {
        "config",
        "eval_interval",
        "global_batch_tokens",
        "gradient_accumulation_steps",
        "gradient_clip_norm",
        "learning_rate",
        "max_steps",
        "max_tokens",
        "micro_batch_size",
        "stage_budget_usd",
        "warmup_fraction",
        "weight_decay",
    },
    "grpo": {
        "config",
        "gradient_clip_norm",
        "group_size",
        "learning_rate",
        "max_new_tokens",
        "max_steps",
        "prompts_per_step",
        "ratio_clip_epsilon",
        "reference_kl_beta",
        "stage_budget_usd",
    },
    "evaluation": {"deterministic", "max_examples", "stage_budget_usd"},
    "data": {"config", "stage_budget_usd"},
}


def _validate_run_config(run: dict[str, Any]) -> None:
    """Reject misspelled run fields before any artifact or cloud action."""

    unknown = set(run) - _RUN_FIELDS
    if unknown:
        raise ValueError(f"unknown run config fields: {', '.join(sorted(unknown))}")
    required = {"run_name", "seed", "device"}
    missing = required - set(run)
    if missing:
        raise ValueError(f"missing run config fields: {', '.join(sorted(missing))}")
    for section, allowed in _SECTION_FIELDS.items():
        value = run.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"run section {section} must be a mapping")
        nested_unknown = set(value) - allowed
        if nested_unknown:
            names = ", ".join(sorted(nested_unknown))
            raise ValueError(f"unknown {section} config fields: {names}")


def _resolve_path(value: str, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config_path.parent / candidate


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _pipeline_summary(run: dict[str, Any], model_config: ModelConfig) -> dict[str, object]:
    model = GPT(model_config)
    base = run.get("base", {})
    sequence = int(base.get("sequence_length", model_config.max_seq_len))
    micro = int(base.get("micro_batch_size", 1))
    accumulation = int(base.get("gradient_accumulation_steps", 1))
    steps = int(base.get("max_steps", 0))
    projected_tokens = steps * sequence * micro * accumulation
    if "target_tokens_100m" in base:
        projected_tokens = int(base["target_tokens_100m"])
    rate = run.get("hourly_rate_usd")
    return {
        "run_name": run.get("run_name"),
        "device": run.get("device"),
        "model_parameters": model.parameter_count(),
        "projected_base_tokens": projected_tokens,
        "hourly_rate_usd": rate,
        "budget_usd": run.get("total_budget_usd", 0.0),
        "model_config_sha256": model_config.fingerprint(),
        "resolved_config_sha256": hashlib.sha256(canonical_json(run).encode()).hexdigest(),
    }


def _load_model_config(run: dict[str, Any], config_path: Path) -> ModelConfig:
    value = run.get("model_config")
    if value == "auto":
        # Dry-run uses the primary candidate; a real cloud launch must perform
        # its measured probe before this command is confirmed.
        value = "configs/model/gpt_100m.yaml"
    if not isinstance(value, str):
        raise ValueError("run config requires model_config")
    return ModelConfig.from_yaml(_resolve_path(value, config_path))


def _fixture_corpus() -> list[str]:
    paragraphs = [
        "A small language model learns by predicting the next token in ordinary educational prose.",
        "Transformers mix information with causal attention and then use feed-forward layers.",
        "Reproducible experiments record data, configuration, seeds, and checkpoint lineage.",
        "Arithmetic has answers that software can verify without a subjective reward model.",
    ]
    return paragraphs * 80


def _offline_dataset_identity(stage: str, run: dict[str, Any]) -> tuple[str, str]:
    """Fingerprint the exact deterministic fixture consumed by each stage."""

    if stage == "base":
        revision = "offline-base-fixture-v1"
        payload: object = _fixture_corpus()
    elif stage == "sft":
        revision = "offline-synthetic-sft-v1"
        count = max(16, int(run["sft"]["max_steps"]) * 2)
        payload = [
            {
                "id": record.id,
                "messages": record.messages,
                "source": record.source,
                "license": record.license,
            }
            for record in arithmetic_conversations(count)
        ]
    elif stage == "grpo":
        revision = "offline-deterministic-grpo-rollout-v1"
        payload = {
            "prompt_specs": [[2, 3], [7, 8]],
            "group_size": int(run["grpo"].get("group_size", 4)),
            "completion_rule": "alternating-correct-off-by-one",
        }
    else:
        raise ValueError(f"unknown offline stage {stage!r}")
    fingerprint = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return revision, fingerprint


def _manifest(
    stage: str,
    run: dict[str, Any],
    tokenizer: ByteBPETokenizer,
    *,
    parent: Path | None,
    resume: str,
    step_count: int,
    token_count: int,
    wall_time_seconds: float,
) -> dict[str, object]:
    lock_path = Path("uv.lock")
    rate = float(run.get("hourly_rate_usd") or 0.0)
    dataset_revision, dataset_hash = _offline_dataset_identity(stage, run)
    return {
        "format_version": 1,
        "stage": stage,
        "parent_checkpoint_hash": (
            None if parent is None else file_sha256(parent / "model.safetensors")
        ),
        "tokenizer_hash": tokenizer.sha256,
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": dataset_hash,
        "dataset_split_hashes": {"train": dataset_hash},
        "package_lock_hash": file_sha256(lock_path) if lock_path.exists() else None,
        "resolved_config_sha256": hashlib.sha256(canonical_json(run).encode()).hexdigest(),
        "command": (
            f"nanogpt pipeline --config configs/run/local_smoke.yaml --yes --resume {resume}"
        ),
        "step_count": step_count,
        "token_count": token_count,
        "device": str(run["device"]),
        "wall_time_seconds": wall_time_seconds,
        "estimated_cost_usd": wall_time_seconds / 3600 * rate,
    }


def _expected_manifest(
    stage: str,
    run: dict[str, Any],
    tokenizer: ByteBPETokenizer,
    *,
    parent: Path | None,
) -> dict[str, object]:
    lock_path = Path("uv.lock")
    dataset_revision, dataset_hash = _offline_dataset_identity(stage, run)
    return {
        "stage": stage,
        "parent_checkpoint_hash": (
            None if parent is None else file_sha256(parent / "model.safetensors")
        ),
        "tokenizer_hash": tokenizer.sha256,
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": dataset_hash,
        "package_lock_hash": file_sha256(lock_path) if lock_path.exists() else None,
        "resolved_config_sha256": hashlib.sha256(canonical_json(run).encode()).hexdigest(),
    }


def _stage_checkpoint(paths: ArtifactPaths, stage: str, run_name: str) -> Path:
    return paths.checkpoints / stage / f"{run_name}-final"


def _sft_batches(
    records: list[ConversationRecord],
    tokenizer: ByteBPETokenizer,
    max_seq_len: int,
    steps: int,
) -> list[SFTBatch]:
    output: list[SFTBatch] = []
    for index in range(steps):
        pair = [records[(index * 2) % len(records)], records[(index * 2 + 1) % len(records)]]
        output.append(collate_sft(pair, tokenizer, max_seq_len=max_seq_len))
    return output


def _grpo_fixture_update(
    model: GPT, tokenizer: ByteBPETokenizer, config: dict[str, Any]
) -> tuple[dict[str, float], torch.optim.AdamW, int]:
    """Exercise the real GRPO math on a mixed deterministic offline rollout."""

    device = next(model.parameters()).device
    prompt_specs = [(2, 3), (7, 8)]
    group_size = int(config.get("group_size", 4))
    rows: list[list[list[int]]] = []
    masks: list[list[list[bool]]] = []
    rewards: list[list[float]] = []
    for left, right in prompt_specs:
        group_rows, group_masks, group_rewards = [], [], []
        prompt = tokenizer.encode(f"Compute {left}+{right}. ")
        for sample in range(group_size):
            answer = left + right if sample % 2 == 0 else left + right + 1
            completion = f"<final>{answer}</final>"
            completion_ids = tokenizer.encode(completion)
            sequence = prompt + completion_ids
            group_rows.append(sequence)
            group_masks.append([False] * max(0, len(prompt) - 1) + [True] * len(completion_ids))
            group_rewards.append(numeric_exact_reward(completion, str(left + right)))
        rows.append(group_rows)
        masks.append(group_masks)
        rewards.append(group_rewards)
    width = max(len(sequence) - 1 for group in rows for sequence in group)
    pad = tokenizer.special_to_id["<|pad|>"]
    inputs, targets, completion_masks = [], [], []
    for group, group_masks in zip(rows, masks, strict=True):
        input_group, target_group, mask_group = [], [], []
        for sequence, mask in zip(group, group_masks, strict=True):
            padding = width - (len(sequence) - 1)
            input_group.append(sequence[:-1] + [pad] * padding)
            target_group.append(sequence[1:] + [pad] * padding)
            mask_group.append(mask + [False] * padding)
        inputs.append(input_group)
        targets.append(target_group)
        completion_masks.append(mask_group)
    input_tensor = torch.tensor(inputs, device=device)
    target_tensor = torch.tensor(targets, device=device)
    mask_tensor = torch.tensor(completion_masks, dtype=torch.bool, device=device)

    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        flat = input_tensor.flatten(0, 1)
        old_logits = model(flat).logits
        reference_logits = reference(flat).logits
        assert old_logits is not None and reference_logits is not None
        flat_targets = target_tensor.flatten(0, 1)
        old = sequence_log_probs(old_logits, flat_targets).reshape_as(input_tensor)
        reference_log_probs = sequence_log_probs(reference_logits, flat_targets).reshape_as(
            input_tensor
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-5)))
    result = grpo_update(
        model=model,
        optimizer=optimizer,
        input_ids=input_tensor,
        target_ids=target_tensor,
        old_log_probs=old,
        reference_log_probs=reference_log_probs,
        rewards=torch.tensor(rewards, device=device),
        completion_mask=mask_tensor,
        grad_clip=1.0,
    )
    completion_tokens = int(mask_tensor.sum())
    metric = {
        "step": 1.0,
        "loss": float(result.loss.detach()),
        "policy_loss": float(result.policy_loss.detach()),
        "kl": float(result.kl.detach()),
        "clip_fraction": float(result.clip_fraction),
        "mean_reward": float(torch.tensor(rewards).mean()),
        "mixed_group_fraction": float(result.valid_group_fraction),
        "mean_completion_length": completion_tokens / (len(prompt_specs) * group_size),
        "malformed_answer_rate": 0.0,
    }
    return metric, optimizer, completion_tokens


def _run_local_pipeline(run: dict[str, Any], model_config: ModelConfig, *, resume: str) -> None:
    seed_everything(int(run.get("seed", 1337)))
    device = _resolve_device(str(run.get("device", "auto")))
    run["device"] = str(device)
    paths = ArtifactPaths.from_environment(run.get("artifact_root"))
    paths.create()
    run_name = str(run.get("run_name", "local_smoke"))

    tokenizer_path = paths.tokenizer / f"{run_name}-byte-bpe"
    if tokenizer_path.exists():
        tokenizer = ByteBPETokenizer.load(tokenizer_path)
    else:
        tokenizer = ByteBPETokenizer.train(_fixture_corpus(), vocab_size=512)
        tokenizer.save(tokenizer_path)
    model = GPT(model_config).to(device)

    base_path = _stage_checkpoint(paths, "base", run_name)
    base_config = run["base"]
    if resume == "auto" and base_path.exists():
        load_checkpoint(
            base_path,
            model=model,
            expected=_expected_manifest("base", run, tokenizer, parent=None),
            restore_rng=False,
        )
    else:
        stage_started = time.monotonic()
        text = "\n".join(_fixture_corpus())
        tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        base_batches = list(
            causal_batches(
                tokens,
                batch_size=int(base_config["micro_batch_size"]),
                seq_len=int(base_config["sequence_length"]),
            )
        )
        loop = TrainLoopConfig(
            max_steps=int(base_config["max_steps"]),
            learning_rate=float(base_config["learning_rate"]),
            accumulation_steps=int(base_config.get("gradient_accumulation_steps", 1)),
        )
        if not base_batches:
            raise ValueError("offline base fixture produced no complete training batch")
        # The local fixture is deliberately tiny; cycling it ensures the smoke
        # run validates the configured number of optimizer updates.
        optimizer, metrics = train_base(model, cycle(base_batches), loop, device=device)
        base_tokens = (
            len(metrics)
            * int(base_config["sequence_length"])
            * int(base_config["micro_batch_size"])
            * int(base_config.get("gradient_accumulation_steps", 1))
        )
        save_checkpoint(
            base_path,
            model=model,
            optimizer=optimizer,
            trainer_state={
                "step": len(metrics),
                "tokens": base_tokens,
            },
            run_manifest=_manifest(
                "base",
                run,
                tokenizer,
                parent=None,
                resume=resume,
                step_count=len(metrics),
                token_count=base_tokens,
                wall_time_seconds=time.monotonic() - stage_started,
            ),
            metrics=metrics,
        )

    sft_path = _stage_checkpoint(paths, "sft", run_name)
    sft_config = run["sft"]
    if resume == "auto" and sft_path.exists():
        load_checkpoint(
            sft_path,
            model=model,
            expected=_expected_manifest("sft", run, tokenizer, parent=base_path),
            restore_rng=False,
        )
    else:
        stage_started = time.monotonic()
        records = list(arithmetic_conversations(max(16, int(sft_config["max_steps"]) * 2)))
        sft_batches = _sft_batches(
            records, tokenizer, model_config.max_seq_len, int(sft_config["max_steps"])
        )
        loop = TrainLoopConfig(
            max_steps=int(sft_config["max_steps"]),
            learning_rate=float(sft_config["learning_rate"]),
            weight_decay=0.01,
            accumulation_steps=int(sft_config.get("gradient_accumulation_steps", 1)),
        )
        optimizer, metrics = train_sft(model, sft_batches, loop, device=device)
        consumed_microbatches = len(metrics) * loop.accumulation_steps
        sft_tokens = sum(
            int(batch.loss_mask.sum()) for batch in sft_batches[:consumed_microbatches]
        )
        save_checkpoint(
            sft_path,
            model=model,
            optimizer=optimizer,
            trainer_state={"step": len(metrics)},
            run_manifest=_manifest(
                "sft",
                run,
                tokenizer,
                parent=base_path,
                resume=resume,
                step_count=len(metrics),
                token_count=sft_tokens,
                wall_time_seconds=time.monotonic() - stage_started,
            ),
            metrics=metrics,
        )

    grpo_path = _stage_checkpoint(paths, "grpo", run_name)
    if resume == "auto" and grpo_path.exists():
        load_checkpoint(
            grpo_path,
            model=model,
            expected=_expected_manifest("grpo", run, tokenizer, parent=sft_path),
            restore_rng=False,
        )
    else:
        stage_started = time.monotonic()
        metric, optimizer, grpo_tokens = _grpo_fixture_update(model, tokenizer, run["grpo"])
        save_checkpoint(
            grpo_path,
            model=model,
            optimizer=optimizer,
            trainer_state={"step": 1, "offline_rollout_fixture": True},
            run_manifest=_manifest(
                "grpo",
                run,
                tokenizer,
                parent=sft_path,
                resume=resume,
                step_count=1,
                token_count=grpo_tokens,
                wall_time_seconds=time.monotonic() - stage_started,
            ),
            metrics=[metric],
        )
    print(json.dumps({"status": "complete", "run": run_name, "artifact_root": str(paths.root)}))


def _pipeline(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    run = _read_yaml(config_path)
    _validate_run_config(run)
    if "NANOGPT_HOME" in os.environ:
        run["artifact_root"] = os.environ["NANOGPT_HOME"]
    if "NANOGPT_HOURLY_RATE_USD" in os.environ:
        run["hourly_rate_usd"] = float(os.environ["NANOGPT_HOURLY_RATE_USD"])
    if "NANOGPT_TOTAL_BUDGET_USD" in os.environ:
        run["total_budget_usd"] = float(os.environ["NANOGPT_TOTAL_BUDGET_USD"])
    budget = float(run.get("total_budget_usd", 0.0))
    if not math.isfinite(budget) or budget < 0 or budget > 100:
        raise ValueError("total_budget_usd must be between zero and 100")
    rate_value = run.get("hourly_rate_usd")
    if rate_value is not None:
        rate = float(rate_value)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("hourly_rate_usd must be a finite positive number")
        run["hourly_rate_usd"] = rate
    if run.get("offline_fixture", False):
        # Make the printed canonical configuration exactly the one fingerprinted
        # into local manifests; leaving `auto` unresolved creates false lineage.
        run["device"] = str(_resolve_device(str(run["device"])))
    model_config = _load_model_config(run, config_path)
    summary = _pipeline_summary(run, model_config)
    print(json.dumps({"resolved": run, "summary": summary}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    if not args.yes:
        print(
            "Refusing to start training without --yes (use --dry-run to inspect).",
            file=sys.stderr,
        )
        return 2
    if not run.get("offline_fixture", False):
        if run.get("hourly_rate_usd") is None:
            print("Paid runs require an explicit hourly_rate_usd.", file=sys.stderr)
            return 2
        print(
            "Cloud orchestration requires prepared tokenizer/shards and a measured profile probe; "
            "run this configuration with --dry-run until those artifacts exist.",
            file=sys.stderr,
        )
        return 2
    _run_local_pipeline(run, model_config, resume=args.resume)
    return 0


def _tokenizer(args: argparse.Namespace) -> int:
    source = Path(args.input)
    input_bytes = source.stat().st_size
    print(
        json.dumps(
            {
                "input_bytes": input_bytes,
                "reference_trainer_max_bytes": REFERENCE_TRAINER_MAX_BYTES,
                "vocab_size_ceiling": args.vocab_size,
            }
        )
    )
    if args.dry_run:
        return 0
    if not args.yes:
        return 2
    if input_bytes > REFERENCE_TRAINER_MAX_BYTES:
        raise ValueError(
            "input is too large for the reference BPE trainer; "
            "the accelerated production tokenizer path is not implemented"
        )
    texts = source.read_text(encoding="utf-8").splitlines()
    ByteBPETokenizer.train(texts, args.vocab_size).save(args.output)
    return 0


def _prepare(args: argparse.Namespace) -> int:
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    records = [FineWebRecord.from_mapping(row) for row in rows]
    print(json.dumps({"documents": len(records), "tokenizer_hash": tokenizer.sha256}))
    if args.dry_run:
        return 0
    if not args.yes:
        return 2
    prepare_shards(records, tokenizer, args.output, shard_tokens=args.shard_tokens)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanogpt")
    subcommands = parser.add_subparsers(dest="command", required=True)
    pipeline = subcommands.add_parser("pipeline", help="run base -> SFT -> GRPO")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--yes", action="store_true")
    pipeline.add_argument("--resume", choices=("none", "auto"), default="none")
    pipeline.set_defaults(handler=_pipeline)

    tokenizer = subcommands.add_parser("tokenizer", help="train the byte BPE tokenizer")
    tokenizer.add_argument("--input", required=True, help="UTF-8 text with one document per line")
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=32_768)
    tokenizer.add_argument("--dry-run", action="store_true")
    tokenizer.add_argument("--yes", action="store_true")
    tokenizer.set_defaults(handler=_tokenizer)

    prepare = subcommands.add_parser("prepare", help="prepare local FineWeb-schema JSONL")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--shard-tokens", type=int, default=1_000_000)
    prepare.add_argument("--dry-run", action="store_true")
    prepare.add_argument("--yes", action="store_true")
    prepare.set_defaults(handler=_prepare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert validation failures into concise user errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
