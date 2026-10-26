# Runbook

## Local smoke run

```bash
uv sync --dev
./scripts/smoke.sh                     # ~20 s on an M1; no network, no GPU
uv run nanogpt sample --config configs/runs/smoke.yaml --stage sft --prompt "What is 2 + 3?"
```

Artifacts go to `$NANOGPT_HOME/runs/smoke/` (default `./artifacts`). Run it
again and finished stages are skipped. Delete the run directory to start
over.

## Real-data rehearsal

Before renting a GPU, run the cloud recipe's real sources on a tiny slice with
the nano model. It takes a few minutes and is free:

```bash
uv sync --dev --extra data
NANOGPT_HOME=/tmp/nanogpt uv run nanogpt pipeline --config configs/runs/rehearsal.yaml
```

This exercises the FineWeb, GSM8K and SmolTalk adapters at their pinned
revisions. Setting `HF_TOKEN` raises Hugging Face rate limits. Delete the
directory afterwards.

## Sub-$100 cloud run

The dollar ceiling is a hard stop, not a target. Stage caps live in
`configs/runs/cloud.yaml`:

| Stage | Cap |
| --- | ---: |
| Data + tokenizer + probe | $5 |
| Base | $62 |
| SFT | $8 |
| GRPO | $17 |
| Evaluation / retry reserve | $6 |
| Storage (unallocated headroom) | $2 |

Taxes, egress, minimum billing increments and detached storage may fall
outside a provider's GPU price. Check them before launch and lower the caps
if needed.

### Data preparation cost

The rehearsal measured on an 8-core M1:

- FineWeb streams at about 4,900 docs/s, so the network is not the
  bottleneck.
- Training a 32k tokenizer on 50M characters took 26 s.
- Shard preparation runs at about 455k tokens/s (FineWeb-Edu averages about
  1,040 tokens per document with a 32k vocab). Byte-level BPE encoding is the
  bottleneck.

At that rate, 2B tokens take about 75 minutes, and the 1B-character tokenizer
sample takes roughly another 10. On the GPU instance at $3/h that is about
$4 and nearly exhausts the $5 data allocation. Two ways around it:

- **Preferred:** run `nanogpt tokenizer` and `nanogpt prepare` on a cheap
  CPU instance with many cores, attached to the same persistent volume, then
  start the GPU instance for training.
- If you prepare on the GPU box, first check its encode rate on a slice and
  lower `data.max_tokens` if the projection does not fit.

The prepared shards take about 4 GB (2 bytes per token).

### Before renting a GPU

1. `./scripts/smoke.sh` and `uv run pytest` pass from a clean checkout.
2. You have persistent storage that survives instance termination. Point
   `NANOGPT_HOME` at it.
3. `NANOGPT_HOURLY_RATE_USD=<price> ./scripts/cloud.sh` (a dry run) prints a
   plan you have read: model, token horizon, stage caps and paths.

Never put `HF_TOKEN` or provider credentials in a config, image, log or
manifest.

### Launch

```bash
uv sync --extra data
export NANOGPT_HOME=/persistent/nanogpt NANOGPT_HOURLY_RATE_USD=<exact launch price>
./scripts/cloud.sh --execute
```

`--execute` first runs `nanogpt probe`. The probe times real base steps for
each candidate model and projects the base-stage cost at the configured token
horizon, including a 20% safety margin. It recommends the largest candidate
that fits `base.budget_usd`. If that isn't `gpt_100m`, answer `n`, set
`model:` in `cloud.yaml` (or lower `base.max_steps`), and rerun. Never shrink
the safety margin just to pass.

The pipeline then runs tokenizer → prepare → base → SFT → GRPO → eval. To run
or debug one step at a time:

```bash
uv run nanogpt train base --config configs/runs/cloud.yaml --yes --resume
```

### Budget guard

Each stage tracks monotonic wall time × hourly rate. Time spent before a
resume still counts. Before each step it checks that the next step plus the
`checkpoint_reserve_seconds` margin fits the stage cap. If it does not, the
stage checkpoints, writes `STATUS.json` with `budget_exhausted`, and the
pipeline exits with code 3 without starting the next stage. A partial, valid
checkpoint is better than going over budget.

These are estimates from wall time, not provider invoices.

### Preemption and resume

Attach a new instance to the same volume and rerun the same command with
`--resume`. The latest `step_*` checkpoint is verified (weights checksum,
tokenizer, dataset, parent) and training continues from the exact data
cursor and RNG state. Leftover `.step_*-xxxx` temporary directories come from
interrupted writes. They are incomplete and safe to delete.

Use spot capacity only after you have tested a kill-and-resume on the smoke
run.

### Shutdown

- Confirm the final checkpoints and `eval/report.json` exist on persistent
  storage.
- Copy the configs, manifests, `logs/*.jsonl` and the report to backup.
- Terminate the GPU instance and confirm it is stopped, not just
  disconnected. Then delete unattached volumes, IPs and snapshots.
- Never publish prepared shards or `trainer_state.pt` (pickle). Share models
  as `model.safetensors` + `model_config.json` + tokenizer, after reviewing
  the data licenses.

## Troubleshooting

**Out of memory:** lower `micro_batch_size` and raise
`gradient_accumulation_steps` by the same factor, which keeps tokens per step
unchanged. Try `compile: false`. Rerun the probe, since cost changes.

**Non-finite loss or gradient:** the trainer stops before applying the
update. Rerun the step in FP32 (`dtype: float32`, `compile: false`) and
check the token range, the LR and the gradient norm in `logs/<stage>.jsonl`.
Resume from the last checkpoint only after you understand the cause.

**Compile or kernel failures:** run eager (`compile: false`). The SDPA math
fallback is correct but slower, so rerun the probe.

**Checkpoint rejected on resume:** the error names the mismatched field
(tokenizer, dataset, parent or model config). Point at the right artifacts or
start a new run name. Do not bypass the check.

**Corrupt shard:** delete the whole `data/base` directory and run `nanogpt
prepare` again. Never edit a checksum.

**GRPO pilot too sparse:** the SFT model rarely produces a correct
`<final>` answer. Strengthen SFT (more arithmetic/GSM8K examples or more
steps) before spending the RL budget. Lowering the gate only hides the
problem.

**Reward rises but held-out accuracy falls:** look for marker spam, copied
prompt numbers or template shortcuts. Compare the KL, malformed rate and
completion length in the GRPO log.

**Disk fills up:** each checkpoint stores weights plus AdamW state (about 3×
the model size). Raise `checkpoint_every` or prune old `step_*` directories
of finished stages.
