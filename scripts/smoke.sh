#!/usr/bin/env bash
# Run the complete tiny pipeline offline: tokenizer -> shards -> base -> SFT -> GRPO -> eval.
# Re-running resumes: finished stages are skipped, partial ones continue.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --frozen proves the checked-in lock is sufficient.
exec uv run --frozen nanogpt pipeline --config configs/runs/smoke.yaml --resume "$@"
