# nano-gpt

Train a small GPT from scratch in readable PyTorch, through three stages:

1. **Base:** next-token pretraining on a deterministic FineWeb-Edu subset.
2. **SFT:** supervised fine-tuning on the assistant turns only.
3. **GRPO:** reinforcement learning with exact, verifiable math rewards.

The cloud recipe targets one GPU and **at most $100** in total. A laptop
(including Apple Silicon) runs the tests and a complete offline smoke
pipeline. That pipeline uses the same tokenizer, shard loader, trainer and
GRPO rollout code as the cloud run.

## Quick start

Needs Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --dev
./scripts/smoke.sh
```

The smoke run trains a tokenizer, writes shards, then runs base → SFT → GRPO →
eval on a tiny model in about 20 seconds. Results land in
`artifacts/runs/smoke/`; set `NANOGPT_HOME` to put them somewhere else.
Running it again skips stages that already finished.

Each step can also be run on its own:

```bash
uv run nanogpt tokenizer --config configs/runs/smoke.yaml
uv run nanogpt prepare   --config configs/runs/smoke.yaml
uv run nanogpt train base --config configs/runs/smoke.yaml --resume
uv run nanogpt eval      --config configs/runs/smoke.yaml
uv run nanogpt sample    --config configs/runs/smoke.yaml --stage sft --prompt "What is 2 + 3?"
```

Before renting a GPU, read the [runbook](docs/runbook.md):

```bash
NANOGPT_HOURLY_RATE_USD=2.99 ./scripts/cloud.sh             # dry run: prints the resolved plan
NANOGPT_HOURLY_RATE_USD=2.99 ./scripts/cloud.sh --execute   # probe, confirm, then run
```

## Results

The first cloud run trained `gpt_100m` (101M parameters) through all three
stages for about $26. Base pretraining on 2B FineWeb-Edu tokens reached a
validation perplexity of 25.8. On 500 held-out 2-digit addition prompts,
GRPO raised accuracy over its SFT starting point:

| Metric | SFT | GRPO |
| --- | ---: | ---: |
| Greedy exact match | 32.0% | 46.6% |
| Sampled pass@1 (T = 0.8) | 9.3% | 19.0% |
| Sampled pass@8 | 48.6% | 63.8% |

The full write-up, with training curves and costs, is in
[docs/results.md](docs/results.md). The weights are at
[huggingface.co/cameronervin1/nanogpt-100m](https://huggingface.co/cameronervin1/nanogpt-100m).

## Layout

```text
configs/models/   architecture profiles (nano, tiny, gpt_50m, gpt_100m)
configs/runs/     one self-contained file per run (smoke, cloud)
src/nanogpt/      config · model/ · tokenizer/ · data/ · train/ · generate · rewards · evaluate · cli
tests/            unit/ and integration/ (includes the full smoke pipeline)
docs/             design.md · runbook.md · roadmap.md · results.md
scripts/          smoke and cloud launchers; experiments/ holds the GRPO go/no-go check and sweep
```

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest
```

## Docs

- [Design](docs/design.md): model, tokenizer, data, the three stages,
  checkpoints, evaluation.
- [Runbook](docs/runbook.md): smoke run, cloud budget, resume, shutdown,
  troubleshooting.
- [Results](docs/results.md): the first cloud run, from base to GRPO.
- [Roadmap](docs/roadmap.md): what is done and what comes next.

## License and data notice

The code is Apache-2.0 (see [LICENSE](LICENSE)). That license does not cover
downloaded datasets, trained weights or generated outputs; each data source
keeps its own terms. FineWeb-Edu is open-web data and may contain
copyrighted, harmful, biased or private content. See
[Design → Sources and licenses](docs/design.md#sources-and-licenses).
