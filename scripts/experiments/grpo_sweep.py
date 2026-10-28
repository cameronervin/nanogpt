"""Short GRPO runs from one SFT checkpoint, one per config, with held-out evals.

    uv run python scripts/experiments/grpo_sweep.py CHECKPOINT \
        '{"lr": 1e-6, "batch": 64, "beta": 0.02, "steps": 60, "eval_every": 20}' \
        '{"lr": 3e-6, "batch": 64, "beta": 0.02, "steps": 60, "eval_every": 20}'

Prints one JSON line per eval: held-out greedy exact match on 200 prompts
(seed 424242), recent mean reward, and KL to the frozen SFT reference.
"""

import copy
import json
import os
import sys
import time

import torch

from nanogpt.config import load_run_config
from nanogpt.data.sources import arithmetic_prompts
from nanogpt.evaluate import chat_reasoning_eval
from nanogpt.model.gpt import GPT
from nanogpt.tokenizer import load_tokenizer
from nanogpt.train.checkpoint import load_checkpoint
from nanogpt.train.grpo import collect_rollout, grpo_update, rollout_log_probs
from nanogpt.train.optim import build_adamw

TEMPERATURE, TOP_K, GROUP_SIZE, MAX_NEW_TOKENS = 0.8, 50, 8, 96

_, model_config = load_run_config("configs/runs/cloud.yaml")
root = os.path.join(os.environ.get("NANOGPT_HOME", "artifacts"), "runs/cloud")
tokenizer = load_tokenizer(os.path.join(root, "tokenizer"))
base = GPT(model_config)
load_checkpoint(sys.argv[1], model=base, restore_rng=False)
held_out = list(arithmetic_prompts(200, seed=424_242))
train = list(arithmetic_prompts(20_000, seed=28_001))

for config in (json.loads(arg) for arg in sys.argv[2:]):
    batch, steps, every = config["batch"], config["steps"], config["eval_every"]
    torch.manual_seed(0)
    model = copy.deepcopy(base).cuda()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = build_adamw(model, learning_rate=config["lr"], weight_decay=0.0, fused=True)
    start, rewards = time.time(), []
    for step in range(steps):
        model.eval()
        rollout = collect_rollout(
            model,
            tokenizer,
            train[step * batch : (step + 1) * batch],
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            seed=1000 + step,
        )
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            old = rollout_log_probs(model, rollout, chunk_prompts=16, temperature=TEMPERATURE)
            ref = rollout_log_probs(reference, rollout, chunk_prompts=16, temperature=TEMPERATURE)
            result = grpo_update(
                model=model,
                optimizer=optimizer,
                input_ids=rollout.input_ids,
                target_ids=rollout.target_ids,
                old_log_probs=old,
                reference_log_probs=ref,
                rewards=rollout.rewards,
                completion_mask=rollout.completion_mask,
                beta=config["beta"],
                chunk_prompts=16,
                temperature=TEMPERATURE,
            )
        rewards.append(float(rollout.rewards.mean()))
        if (step + 1) % every == 0:
            model.eval()
            exact = chat_reasoning_eval(model, tokenizer, held_out, max_new_tokens=96)
            line = {
                **config,
                "step": step + 1,
                "heldout": exact["exact_match"],
                "reward_recent": sum(rewards[-every:]) / every,
                "kl": round(float(result.kl), 4),
                "secs": round(time.time() - start),
            }
            print(json.dumps(line), flush=True)
