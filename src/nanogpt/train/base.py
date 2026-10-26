"""Stage 1: causal language-model pretraining on prepared token shards."""

from __future__ import annotations

from dataclasses import asdict

from nanogpt.data.shards import TokenLoader
from nanogpt.evaluate import evaluate_language_model
from nanogpt.model.gpt import GPT
from nanogpt.tokenizer import load_tokenizer
from nanogpt.train.trainer import RunContext, Trainer, TrainResult


def run_base(ctx: RunContext, *, resume: bool = False) -> TrainResult:
    config = ctx.config.base
    if config.sequence_length > ctx.model_config.max_seq_len:
        raise ValueError("base.sequence_length exceeds the model's max_seq_len")
    tokenizer = load_tokenizer(ctx.paths.tokenizer)
    if tokenizer.vocab_size > ctx.model_config.vocab_size:
        raise ValueError("tokenizer vocabulary is larger than the model's vocab_size")
    data_dir = ctx.paths.data("base")
    loader = TokenLoader(
        data_dir, split="train", batch_size=config.micro_batch_size, seq_len=config.sequence_length
    )
    if loader.manifest["tokenizer_sha256"] != tokenizer.sha256:
        raise ValueError("base shards were prepared with a different tokenizer")
    model = GPT(ctx.model_config)

    def evaluate() -> dict[str, float]:
        # A fresh loader always reads the same fixed validation window.
        validation = TokenLoader(
            data_dir,
            split="validation",
            batch_size=config.micro_batch_size,
            seq_len=config.sequence_length,
            verify_checksums=False,
        )
        batches = [validation.next_batch() for _ in range(ctx.config.eval.validation_batches)]
        metrics = evaluate_language_model(model, batches, device=ctx.device)
        return {
            f"val_{key}": float(value)
            for key, value in asdict(metrics).items()
            if value is not None
        }

    has_validation = any(item["split"] == "validation" for item in loader.manifest["shards"])
    trainer = Trainer(
        ctx,
        "base",
        config,
        model,
        lineage={
            "parent_model_sha256": None,
            "tokenizer_sha256": tokenizer.sha256,
            "dataset_sha256": loader.manifest_sha256,
        },
        evaluate=evaluate if has_validation else None,
        data_state=(loader.state_dict, loader.load_state_dict),
    )
    if (finished := trainer.prepare(resume=resume)) is not None:
        return finished
    return trainer.fit(trainer.lm_step(loader.next_batch))
