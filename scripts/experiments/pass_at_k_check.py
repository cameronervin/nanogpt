"""Go/no-go check before GRPO: greedy and sampled accuracy of one SFT checkpoint.

    uv run python scripts/experiments/pass_at_k_check.py \
        $NANOGPT_HOME/runs/cloud/checkpoints/sft/step_002000

Scores 500 held-out 2-digit sums (seed 987654). GRPO has room to improve when
greedy pass@1 is partial (15-70%), pass@8 is well above sampled pass@1, and
at least 20% of sample groups are mixed.
"""

import json
import os
import sys

from nanogpt.config import load_run_config
from nanogpt.data.sources import arithmetic_prompts
from nanogpt.evaluate import chat_reasoning_eval
from nanogpt.model.gpt import GPT
from nanogpt.tokenizer import load_tokenizer
from nanogpt.train.checkpoint import load_checkpoint
from nanogpt.train.grpo import sampled_pass_at_k

_, model_config = load_run_config("configs/runs/cloud.yaml")
root = os.path.join(os.environ.get("NANOGPT_HOME", "artifacts"), "runs/cloud")
tokenizer = load_tokenizer(os.path.join(root, "tokenizer"))
model = GPT(model_config)
load_checkpoint(sys.argv[1], model=model, restore_rng=False)
model = model.cuda().eval()

prompts = list(arithmetic_prompts(500, seed=987_654))
result = {
    "checkpoint": sys.argv[1],
    "greedy": chat_reasoning_eval(model, tokenizer, prompts, max_new_tokens=96),
    "sampled": sampled_pass_at_k(
        model, tokenizer, prompts, k=8, temperature=0.8, top_k=50, max_new_tokens=96, seed=7
    ),
}
print(json.dumps(result))
