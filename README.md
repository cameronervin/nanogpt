# Nano-GPT from scratch

An educational, modern decoder-only Transformer that can be trained through
three explicit stages:

1. base-language modeling on a deterministic FineWeb-Edu subset;
2. assistant-only supervised fine-tuning (SFT); and
3. group-relative policy optimization (GRPO) with exact, verifiable math
   rewards.

The project keeps the model and training loops readable while using practical
components such as RMSNorm, RoPE, grouped-query attention, QK normalization,
SwiGLU, PyTorch scaled-dot-product attention, resumable checkpoints, and strict
cost guards. The cloud recipe is designed for a single GPU and a total budget
of at most **$100**. An Apple Silicon laptop is sufficient for tests and the
tiny offline smoke pipeline, but not the full pretraining run.

## Project status

The codebase is an initial implementation scaffold. The authoritative scope,
interfaces, profiles, stage contracts, and acceptance tests are in
[the initial build plan](docs/initial-build-plan.md).

## Quick start

Prerequisites: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and about 2 GB
of free disk space for the local development environment.

```bash
uv sync --dev
uv run nanogpt --help
./scripts/run_local_smoke.sh
```

The smoke script writes disposable outputs under `artifacts/` and is intended
to work without a CUDA GPU. It safely resumes compatible completed stages, so
running it again does not overwrite immutable checkpoints. Set `NANOGPT_HOME`
to move runtime data elsewhere:

```bash
NANOGPT_HOME=/tmp/nanogpt-artifacts ./scripts/run_local_smoke.sh
```

Before renting a GPU, read [the cloud budget runbook](docs/cloud-budget.md).
The initial scaffold provides a cloud configuration and cost preflight, not a
paid cloud orchestrator. Supply the provider's current hourly rate to inspect
the resolved plan:

```bash
NANOGPT_HOURLY_RATE_USD=2.99 ./scripts/run_cloud_100.sh
```

`--execute` deliberately exits without starting training. A measured
throughput probe, accelerated billion-character tokenizer training, prepared
cloud datasets, mid-stage resume, and paid-stage budget watchdog integration
remain the next implementation phase. The checked-in pure-Python BPE trainer
is guarded at 1 MB and exists only for fixtures and educational inspection.

## Development commands

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
```

The main configuration profiles live in `configs/`. Runtime datasets,
checkpoints, logs, and evaluation reports live under `$NANOGPT_HOME` and are
never committed.

## Documentation

- [Architecture](docs/architecture.md)
- [Data lineage, licensing, and tokenizer](docs/data-and-licenses.md)
- [Training stages and checkpoint handoffs](docs/training-stages.md)
- [Evaluation and release gates](docs/evaluation.md)
- [Cloud budget runbook](docs/cloud-budget.md)
- [Troubleshooting](docs/troubleshooting.md)

## License and data notice

The source code is licensed under Apache-2.0; see [LICENSE](LICENSE). That
license does not apply to downloaded datasets, trained weights, or generated
outputs. Each data source retains its own terms. FineWeb-Edu is open-web data
and may contain copyrighted, harmful, biased, private, or otherwise undesirable
content even after filtering. Review [the data documentation](docs/data-and-licenses.md)
before distributing data-derived artifacts.
