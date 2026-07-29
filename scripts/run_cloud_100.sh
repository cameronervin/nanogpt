#!/usr/bin/env bash
# Inspect the planned single-GPU, sub-$100 training workflow.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: scripts/run_cloud_100.sh [--execute]

Resolves the cloud configuration and performs a dry run. The --execute flag is
reserved for the next implementation phase and currently exits without
starting training or performing any paid action.
EOF
}

execute=false
case "${1:-}" in
  "") ;;
  --execute) execute=true ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; echo "error: unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  echo "error: too many arguments" >&2
  exit 2
fi

if [[ "$execute" == true ]]; then
  echo "error: paid cloud orchestration is not implemented in the initial scaffold" >&2
  echo "run without --execute to validate configuration and cost inputs" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required; install it from https://docs.astral.sh/uv/" >&2
  exit 127
fi

export NANOGPT_HOME="${NANOGPT_HOME:-$repo_root/artifacts}"
export NANOGPT_TOTAL_BUDGET_USD="${NANOGPT_TOTAL_BUDGET_USD:-100}"

if [[ -z "${NANOGPT_HOURLY_RATE_USD:-}" ]]; then
  echo "error: set NANOGPT_HOURLY_RATE_USD to the provider's current GPU rate" >&2
  exit 2
fi

# Reject malformed, zero, or negative rates/caps before the Python process can
# use them in a projection. This check does not round monetary values.
if ! awk -v value="$NANOGPT_HOURLY_RATE_USD" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0) }'; then
  echo "error: NANOGPT_HOURLY_RATE_USD must be a positive decimal" >&2
  exit 2
fi
if ! awk -v value="$NANOGPT_TOTAL_BUDGET_USD" \
  'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value + 0 > 0 && value + 0 <= 100) }'; then
  echo "error: NANOGPT_TOTAL_BUDGET_USD must be a positive decimal no greater than 100" >&2
  exit 2
fi

mkdir -p "$NANOGPT_HOME"

echo "Nano-GPT cloud workflow"
echo "  mode:        dry-run preflight"
echo "  artifacts:   $NANOGPT_HOME"
echo "  hourly rate: \$$NANOGPT_HOURLY_RATE_USD"
echo "  total cap:   \$$NANOGPT_TOTAL_BUDGET_USD"
echo "  config:      configs/run/cloud_100.yaml"

exec uv run --frozen nanogpt pipeline \
  --config configs/run/cloud_100.yaml \
  --dry-run
