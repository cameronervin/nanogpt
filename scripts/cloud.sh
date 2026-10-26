#!/usr/bin/env bash
# Single-GPU, sub-$100 run. Without --execute this is a dry run that prints the
# resolved plan; with it, the throughput probe runs first, then the pipeline.
#
#   NANOGPT_HOURLY_RATE_USD=2.99 NANOGPT_HOME=/persistent/nanogpt scripts/cloud.sh [--execute]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

config=configs/runs/cloud.yaml
case "${1:-}" in
  "") exec uv run --frozen nanogpt pipeline --config "$config" --dry-run ;;
  --execute)
    uv run --frozen --extra data nanogpt probe --config "$config" --yes
    read -r -p "Proceed with the configured model and budget? [y/N] " answer
    [[ "$answer" == [yY] ]] || exit 1
    exec uv run --frozen --extra data nanogpt pipeline --config "$config" --yes --resume ;;
  *) echo "usage: $0 [--execute]" >&2; exit 2 ;;
esac
