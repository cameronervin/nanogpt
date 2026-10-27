"""GRPO: critic-free, group-relative policy optimization with exact rewards.

For each prompt the current policy samples a group of ``G`` completions. Each
completion's reward is normalized against its own group, so no learned value
model is needed. The update is a PPO-style clipped surrogate plus a KL penalty
toward the frozen SFT reference.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from nanogpt.data.schemas import GRPOPrompt
from nanogpt.data.sources import arithmetic_prompts
from nanogpt.evaluate import chat_reasoning_eval
from nanogpt.generate import sample_chat
from nanogpt.model.gpt import GPT
from nanogpt.rewards import numeric_exact_reward, parse_numeric_answer
from nanogpt.tokenizer import Tokenizer, load_tokenizer
from nanogpt.train.checkpoint import load_checkpoint, read_manifest
from nanogpt.train.trainer import RunContext, Trainer, TrainResult, finished_checkpoint
from nanogpt.utils import sha256_file


def group_advantages(rewards: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize per-prompt rewards and identify mixed-reward groups.

    ``rewards`` has shape ``[prompts, completions]``. Zero-variance rows are
    returned as zeros and excluded by ``valid_groups`` rather than amplified by
    an arbitrary epsilon.
    """

    if rewards.ndim != 2 or rewards.size(1) < 2:
        raise ValueError("rewards must have shape [prompts, at least two completions]")
    values = rewards.float()
    means = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, unbiased=False, keepdim=True)
    valid = std.squeeze(1) > 0
    advantages = torch.zeros_like(values)
    advantages[valid] = (values[valid] - means[valid]) / std[valid]
    return advantages, valid


@dataclass(frozen=True)
class GRPOLoss:
    loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    clip_fraction: Tensor
    valid_group_fraction: Tensor


def grpo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    rewards: Tensor,
    completion_mask: Tensor,
    *,
    epsilon: float = 0.2,
    beta: float = 0.02,
) -> GRPOLoss:
    """Compute clipped group-relative policy loss plus reference KL.

    Log-probability tensors and ``completion_mask`` have shape
    ``[prompts, completions, tokens]``; rewards omit the token dimension.
    The reference tensor must be detached by rollout collection.
    """

    if (
        new_log_probs.shape != old_log_probs.shape
        or new_log_probs.shape != reference_log_probs.shape
    ):
        raise ValueError("policy/reference log probabilities must have identical shapes")
    if completion_mask.shape != new_log_probs.shape or rewards.shape != new_log_probs.shape[:2]:
        raise ValueError("completion mask or reward shape is incompatible")
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError("epsilon must be in (0,1) and beta non-negative")
    tensors = (new_log_probs, old_log_probs, reference_log_probs, rewards)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise FloatingPointError("GRPO inputs contain non-finite values")

    advantages, valid_groups = group_advantages(rewards)
    valid_tokens = completion_mask.bool() & valid_groups[:, None, None]
    token_count = valid_tokens.sum()
    if token_count == 0:
        # Preserve a differentiable connection to the policy for callers that
        # uniformly handle skipped and non-skipped rollout batches.
        zero = new_log_probs.sum() * 0
        return GRPOLoss(zero, zero, zero.detach(), zero.detach(), valid_groups.float().mean())

    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    advantage = advantages[:, :, None]
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    policy_tokens = -torch.minimum(unclipped, clipped)

    # This non-negative estimator is zero when policy and reference match and
    # is the same stable per-token approximation commonly used by GRPO.
    ref_delta = reference_log_probs.detach() - new_log_probs
    kl_tokens = ref_delta.exp() - ref_delta - 1
    weights = valid_tokens.to(new_log_probs.dtype)
    policy_loss = (policy_tokens * weights).sum() / token_count
    kl = (kl_tokens * weights).sum() / token_count
    clip_fraction = (((ratio - 1).abs() > epsilon) & valid_tokens).sum() / token_count
    loss = policy_loss + beta * kl
    if not torch.isfinite(loss):
        raise FloatingPointError("GRPO objective became non-finite")
    return GRPOLoss(loss, policy_loss, kl, clip_fraction, valid_groups.float().mean())


@dataclass(frozen=True)
class RewardDensity:
    correct_fraction: float
    mixed_group_fraction: float

    @property
    def passes(self) -> bool:
        return self.correct_fraction >= 0.05 and self.mixed_group_fraction >= 0.20


def reward_density(rewards: Tensor) -> RewardDensity:
    _, valid = group_advantages(rewards)
    return RewardDensity(float(rewards.float().mean()), float(valid.float().mean()))


def sequence_log_probs(logits: Tensor, target_ids: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Gather target token log probabilities from ``[..., vocab]`` logits.

    ``temperature`` should match the rollout sampler so the ratio and KL are
    taken under the distribution the completions were actually drawn from.
    """

    if logits.shape[:-1] != target_ids.shape:
        raise ValueError("target IDs must match all non-vocabulary dimensions")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits.float() / temperature
    return scaled.log_softmax(dim=-1).gather(-1, target_ids[..., None]).squeeze(-1)


def grpo_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    target_ids: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    rewards: Tensor,
    completion_mask: Tensor,
    epsilon: float = 0.2,
    beta: float = 0.02,
    grad_clip: float = 1.0,
    chunk_prompts: int | None = None,
    temperature: float = 1.0,
) -> GRPOLoss:
    """Perform one optimizer epoch over a fresh, already-collected rollout.

    ``chunk_prompts`` bounds peak memory: the policy forward/backward runs over
    that many prompt groups at a time and gradients accumulate. Each chunk's
    token-mean loss is weighted by its share of valid tokens, so the update
    equals the single-pass one.
    """

    prompts, completions, tokens = input_ids.shape
    size = chunk_prompts or prompts
    if size <= 0:
        raise ValueError("chunk_prompts must be positive")
    _, valid_groups = group_advantages(rewards)
    valid_tokens = completion_mask.bool() & valid_groups[:, None, None]
    total = int(valid_tokens.sum())
    optimizer.zero_grad(set_to_none=True)
    parts: list[tuple[float, GRPOLoss]] = []
    for start in range(0, prompts, size):
        chunk = slice(start, start + size)
        chunk_inputs = input_ids[chunk]
        rows = chunk_inputs.size(0) * completions
        output = model(chunk_inputs.reshape(rows, tokens))
        if output.logits is None:
            raise RuntimeError("policy forward did not return logits")
        new_log_probs = sequence_log_probs(
            output.logits, target_ids[chunk].reshape(rows, tokens), temperature=temperature
        ).reshape_as(chunk_inputs)
        part = grpo_loss(
            new_log_probs,
            old_log_probs[chunk],
            reference_log_probs[chunk],
            rewards[chunk],
            completion_mask[chunk],
            epsilon=epsilon,
            beta=beta,
        )
        weight = int(valid_tokens[chunk].sum()) / total if total else 0.0
        if weight:
            (part.loss * weight).backward()  # type: ignore[no-untyped-call]
        parts.append((weight, part))

    def combine(field: str) -> Tensor:
        return sum(
            (weight * getattr(part, field).detach() for weight, part in parts),
            torch.zeros((), device=input_ids.device),
        )

    objective = GRPOLoss(
        combine("loss"),
        combine("policy_loss"),
        combine("kl"),
        combine("clip_fraction"),
        valid_groups.float().mean(),
    )
    if total == 0:
        # A skipped rollout must not invoke AdamW: even zero gradients would
        # otherwise update parameters through decoupled weight decay.
        return objective
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not math.isfinite(float(norm)):
        raise FloatingPointError("GRPO gradient norm became non-finite")
    optimizer.step()
    return objective


# --------------------------------------------------------------------------- rollouts


@dataclass(frozen=True)
class Rollout:
    """``[prompts, group, tokens]`` tensors for one fresh batch of samples."""

    input_ids: Tensor
    target_ids: Tensor
    completion_mask: Tensor
    rewards: Tensor
    malformed_rate: float
    mean_completion_tokens: float


def collect_rollout(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[GRPOPrompt],
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> Rollout:
    """Sample ``group_size`` completions per prompt and score them exactly."""

    sequences: list[list[list[int]]] = []
    prompt_lengths: list[int] = []
    rewards: list[list[float]] = []
    malformed = 0
    completion_tokens = 0
    for index, prompt in enumerate(prompts):
        prompt_ids, completions = sample_chat(
            model,
            tokenizer,
            prompt.messages,
            num_samples=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed + index,
        )
        texts = [tokenizer.decode(completion) for completion in completions]
        sequences.append([prompt_ids + completion for completion in completions])
        prompt_lengths.append(len(prompt_ids))
        rewards.append([numeric_exact_reward(text, prompt.answer) for text in texts])
        malformed += sum(parse_numeric_answer(text) is None for text in texts)
        completion_tokens += sum(len(completion) for completion in completions)

    # Shift by one for next-token targets and pad every row to a common width.
    # The completion mask selects target positions that are completion tokens,
    # i.e. index >= prompt_length - 1 and within the unpadded sequence.
    width = max(len(sequence) - 1 for group in sequences for sequence in group)
    pad = tokenizer.special_to_id["<|pad|>"]
    inputs, targets, masks = [], [], []
    for group, prompt_length in zip(sequences, prompt_lengths, strict=True):
        group_inputs, group_targets, group_masks = [], [], []
        for sequence in group:
            length = len(sequence) - 1
            padding = width - length
            group_inputs.append(sequence[:-1] + [pad] * padding)
            group_targets.append(sequence[1:] + [pad] * padding)
            group_masks.append(
                [prompt_length - 1 <= position < length for position in range(width)]
            )
        inputs.append(group_inputs)
        targets.append(group_targets)
        masks.append(group_masks)
    device = model.token_embedding.weight.device
    samples = len(prompts) * group_size
    return Rollout(
        input_ids=torch.tensor(inputs, device=device),
        target_ids=torch.tensor(targets, device=device),
        completion_mask=torch.tensor(masks, dtype=torch.bool, device=device),
        rewards=torch.tensor(rewards, device=device),
        malformed_rate=malformed / samples,
        mean_completion_tokens=completion_tokens / samples,
    )


@torch.inference_mode()
def rollout_log_probs(
    model: torch.nn.Module,
    rollout: Rollout,
    *,
    chunk_prompts: int | None = None,
    temperature: float = 1.0,
) -> Tensor:
    prompts = rollout.input_ids.size(0)
    size = chunk_prompts or prompts
    pieces = []
    for start in range(0, prompts, size):
        inputs = rollout.input_ids[start : start + size]
        logits = model(inputs.flatten(0, 1)).logits
        targets = rollout.target_ids[start : start + size].flatten(0, 1)
        pieces.append(
            sequence_log_probs(logits, targets, temperature=temperature).reshape_as(inputs)
        )
    return torch.cat(pieces)


# --------------------------------------------------------------------------- stage


def _read_prompts(path: Path) -> list[GRPOPrompt]:
    return [
        GRPOPrompt.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_grpo(ctx: RunContext, *, resume: bool = False) -> TrainResult:
    """Initialize from SFT, gate on reward density, then run GRPO updates."""

    config = ctx.config.grpo
    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    prompt_path = ctx.paths.data("grpo") / "train.jsonl"
    prompts = _read_prompts(prompt_path)
    parent = finished_checkpoint(ctx.paths, "sft")

    model = GPT(ctx.model_config)
    load_checkpoint(
        parent, model=model, expected={"tokenizer_sha256": tokenizer.sha256}, restore_rng=False
    )
    # The reference is the frozen SFT policy: eval mode, no gradients, ever.
    reference = copy.deepcopy(model).to(ctx.device).eval().requires_grad_(False)
    # Held-out greedy accuracy during training, on a seed no source or the final
    # eval uses, so a run that degrades the SFT policy is visible early.
    held_out = list(arithmetic_prompts(config.eval_prompts, seed=424_242))

    def evaluate() -> dict[str, float]:
        model.eval()
        metrics = chat_reasoning_eval(
            model, tokenizer, held_out, max_new_tokens=config.max_new_tokens
        )
        return {f"heldout_{key}": value for key, value in metrics.items()}

    trainer = Trainer(
        ctx,
        "grpo",
        config,
        model,
        evaluate=evaluate if held_out else None,
        lineage={
            "parent_model_sha256": read_manifest(parent)["model_sha256"],
            "tokenizer_sha256": tokenizer.sha256,
            "dataset_sha256": sha256_file(prompt_path),
        },
    )
    if (finished := trainer.prepare(resume=resume)) is not None:
        return finished

    def rollout_for(step: int, batch: list[GRPOPrompt]) -> Rollout:
        model.eval()
        try:
            return collect_rollout(
                model,
                tokenizer,
                batch,
                group_size=config.group_size,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                # Seeds derive from the step so a resumed run samples identically.
                seed=ctx.config.seed * 1_000_003 + (step + 1) * 1_009,
            )
        finally:
            model.train()

    if trainer.step == 0 and config.pilot_prompts:
        pilot = rollout_for(-1, prompts[: config.pilot_prompts])
        density = reward_density(pilot.rewards)
        trainer.metrics.log(
            {
                "event": "pilot",
                "step": 0,
                "correct_fraction": density.correct_fraction,
                "mixed_group_fraction": density.mixed_group_fraction,
            }
        )
        if (
            density.correct_fraction < config.min_correct_fraction
            or density.mixed_group_fraction < config.min_mixed_group_fraction
        ):
            raise RuntimeError(
                "GRPO pilot reward is too sparse "
                f"(correct {density.correct_fraction:.1%}, mixed groups "
                f"{density.mixed_group_fraction:.1%}); strengthen SFT before spending on RL"
            )

    def step(lr: float) -> dict[str, float]:
        trainer.set_learning_rate(lr)
        start = trainer.step * config.micro_batch_size
        batch = [
            prompts[(start + offset) % len(prompts)] for offset in range(config.micro_batch_size)
        ]
        rollout = rollout_for(trainer.step, batch)
        chunk = config.update_chunk_prompts
        with ctx.autocast():
            # Same precision as the update below: an fp32 old/reference against a
            # bf16 policy makes the ratio and KL carry rounding noise, not drift.
            old = rollout_log_probs(
                model, rollout, chunk_prompts=chunk, temperature=config.temperature
            )
            ref = rollout_log_probs(
                reference, rollout, chunk_prompts=chunk, temperature=config.temperature
            )
            result = grpo_update(
                model=trainer.forward_model,
                optimizer=trainer.optimizer,
                input_ids=rollout.input_ids,
                target_ids=rollout.target_ids,
                old_log_probs=old,
                reference_log_probs=ref,
                rewards=rollout.rewards,
                completion_mask=rollout.completion_mask,
                epsilon=config.ratio_clip_epsilon,
                beta=config.reference_kl_beta,
                grad_clip=config.grad_clip,
                chunk_prompts=chunk,
                temperature=config.temperature,
            )
        return {
            "loss": float(result.loss.detach()),
            "policy_loss": float(result.policy_loss.detach()),
            "kl": float(result.kl),
            "clip_fraction": float(result.clip_fraction),
            "mixed_group_fraction": float(result.valid_group_fraction),
            "mean_reward": float(rollout.rewards.mean()),
            "malformed_answer_rate": rollout.malformed_rate,
            "mean_completion_tokens": rollout.mean_completion_tokens,
            "tokens": int(rollout.completion_mask.sum()),
        }

    return trainer.fit(step)
