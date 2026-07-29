#!/usr/bin/env bash
# Run the complete tiny pipeline with checked-in/offline fixtures.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 127
fi

export NANOGPT_HOME="${NANOGPT_HOME:-$repo_root/artifacts}"
mkdir -p "$NANOGPT_HOME"

echo "Nano-GPT local smoke"
echo "  repository: $repo_root"
echo "  artifacts:  $NANOGPT_HOME"
echo "  config:     configs/run/local_smoke.yaml"

# --frozen proves the checked-in lock is sufficient and prevents a smoke run
# from silently changing dependency resolution.
exec uv run --frozen nanogpt pipeline \
  --config configs/run/local_smoke.yaml \
  --resume auto \
  --yes
