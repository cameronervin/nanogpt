# Runtime artifact policy

This directory is the default value of `$NANOGPT_HOME`. Everything below it
except this document is ignored by Git because it may be large, derived from
licensed data, expensive to regenerate, or unsafe to publish accidentally.

Commands create this layout on demand:

```text
artifacts/
├── tokenizer/
├── datasets/
│   ├── base/
│   ├── sft/
│   └── grpo/
├── checkpoints/
│   ├── base/
│   ├── sft/
│   └── grpo/
├── evaluations/
└── logs/
```

Checkpoint directories are immutable once committed. A complete checkpoint
contains `model.safetensors`, `model_config.json`, `trainer_state.pt`,
`run_manifest.json`, and `metrics.jsonl`. Writers build a temporary sibling
directory and atomically rename it. Explicit file and directory `fsync`
hardening is still required before relying on this path across cloud host
crashes. Do not edit a committed checkpoint in place.

For cloud runs, point `NANOGPT_HOME` at persistent storage and verify that it
survives instance termination. Treat `trainer_state.pt` as trusted-local input:
PyTorch optimizer state uses pickle and must never be loaded from an untrusted
source. Model-only sharing should use `model.safetensors` plus the JSON
configuration and manifest.
