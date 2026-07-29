# Training stages

The project produces three separate checkpoints. Each stage starts from an
immutable parent and creates a new optimizer and schedule. It never silently
updates its parent in place.

The checked-in pipeline executes all three stages on a small offline fixture
and validates stage handoffs. The large FineWeb-Edu, SFT-mixture, measured
cloud-probe, and paid-watchdog paths described below remain implementation
requirements, not currently enabled cloud commands.

```text
tokenizer + FineWeb shards -> base -> SFT -> GRPO
```

All training commands support `--dry-run`, print the resolved configuration,
parameter count, data fingerprints, projected work, and projected cost, and
require `--yes` for paid execution.

## Shared run lifecycle

1. Load strict typed YAML; reject unknown fields and invalid combinations.
2. Resolve paths and environment overrides, serialize canonical JSON, and hash
   it. Copy the resolved JSON into the run log.
3. Validate tokenizer, dataset, and parent-checkpoint fingerprints.
4. Seed Python, NumPy, PyTorch, device RNGs, sampler, and workers.
5. Build the model in eager mode, validate one batch, then optionally compile.
6. Train while logging append-only JSON Lines metrics. Evaluate and checkpoint
   at both work-count and wall-clock intervals.
7. Before the stage deadline, atomically checkpoint and exit successfully with
   a distinct `budget_exhausted` status. Non-finite state exits unsuccessfully.

Gradient accumulation divides each micro-batch loss by the accumulation count.
Clipping and the optimizer step occur only after all micro-batches. Reported
losses are token-weighted across ranks, not averages of unequal local means.

## Base language modeling

For tokens `x_0 ... x_T`, base training minimizes

```text
L_base = -sum(t=0..T-1) log p_theta(x_(t+1) | x_(0:t)) / T
```

The default 100M recipe targets 2B tokens with 262,144 tokens per optimizer
step: micro-batch 16 x length 1,024 x accumulation 16 on one H100. AdamW uses
`betas=(0.9,0.95)`, epsilon `1e-8`, weight decay `0.1`, maximum LR `6e-4`, 2%
linear warmup, then cosine decay to `6e-5`. Norm scales receive no weight decay.
Global gradient norm is clipped at 1.0.

The cost probe may select the 50M profile and 1B-token target instead. This is a
recorded profile choice, never an unannounced architecture change. Validation
uses a fixed FineWeb shard window and reports cross-entropy, perplexity, and
bits per UTF-8 byte.

## Supervised fine-tuning

SFT starts from base weights and teaches the fixed conversation format. For
target mask `m_t` that is one only on assistant content and its `<|end|>`:

```text
L_sft = -sum(t) m_t log p_theta(x_(t+1) | x_(0:t)) / sum(t) m_t
```

The implementation realizes this by assigning `-100` to ignored targets.
Empty assistant masks are rejected. The default mixture is 80% allowlisted
SmolTalk-generated data, 10% GSM8K SFT buckets, and 10% project-generated
reasoning, sampled by assistant tokens rather than record count.

The initial cap is 50M tokens, global batch 65,536 tokens, LR `2e-5`, 3%
warmup, cosine decay, weight decay `0.01`, and gradient clip 1.0. SFT has a new
optimizer/schedule and records the exact base checkpoint hash as its parent.

## GRPO with exact rewards

GRPO starts from SFT weights. A frozen copy of the SFT policy is the reference;
it remains in evaluation mode and must receive no gradients. For each prompt,
the current policy samples a group of `G=8` completions. The exact numeric
checker returns `r_i in {0,1}`. Within a group:

```text
A_i = (r_i - mean(r)) / std(r)
```

Groups with zero reward variance are skipped. No epsilon-normalized artificial
signal is created. For completion token `t`, the clipped surrogate uses the
ratio between the current log probability and the rollout policy log
probability:

```text
rho_i,t = exp(log pi_theta - log pi_old)
surrogate = min(rho_i,t * A_i, clip(rho_i,t, 1-e, 1+e) * A_i)
L_grpo = -mean_completion_tokens(surrogate) + beta * KL(pi_theta || pi_ref)
```

Defaults are `e=0.2`, `beta=0.02`, AdamW LR `5e-6`, one optimizer epoch per
fresh rollout, 16 prompts per step, and at most 500 steps. Completion-token
masking excludes prompt and padding tokens from both surrogate and KL means.

Before spending the RL allocation, the policy samples 128 pilot prompts. At
least 5% of completions must be correct and 20% of groups must contain mixed
rewards. One bounded synthetic SFT warmup is allowed after a failed pilot; a
second failure ends the stage without spending the remaining GRPO budget.

## Checkpoints and exact resume

Each checkpoint contains:

- `model.safetensors` for parameters;
- `model_config.json` for the exact architecture;
- `trainer_state.pt` for trusted-local optimizer, scheduler, scaler, RNG, and
  data cursor state;
- `run_manifest.json` for lineage, hashes, environment, work, time, and cost;
- `metrics.jsonl` for append-only measurements.

Checkpoint creation writes a temporary sibling directory and atomically
renames it only after every required file is flushed and its digest is known.
On resume, compare architecture, tokenizer, dataset, parent, and resolved config
fingerprints. Restore RNG and sampler state before constructing the next batch.
A resume integration test must produce byte-identical final model and optimizer
state to an uninterrupted deterministic CPU run.

`trainer_state.pt` is pickle-based trusted-local data. Never load one obtained
from an untrusted party. For model distribution, publish only SafeTensors and
non-executable JSON metadata after license and data review.
