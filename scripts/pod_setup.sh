#!/usr/bin/env bash
# Idempotent pod setup. Everything (uv, Python, venv, caches, artifacts) lives
# on the network volume at /workspace, so a later pod such as the H100 reuses
# it without reinstalling. Afterwards: `source /workspace/env.sh`.
set -euo pipefail
ws=/workspace

cat > "$ws/env.sh" <<ENV
export PATH=$ws/.local/bin:\$PATH
export UV_CACHE_DIR=$ws/.uv-cache UV_PYTHON_INSTALL_DIR=$ws/.uv-python
export NANOGPT_HOME=$ws/artifacts HF_HOME=$ws/hf
cd $ws/nanogpt
ENV
# shellcheck disable=SC1091
source "$ws/env.sh"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR="$ws/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv sync --frozen --extra data

echo "gpu:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "vcpus:  $(nproc)"
echo "volume: $(df -h "$ws" | tail -1)"
echo "hf token: $([[ -n "${HF_TOKEN:-}" ]] && echo set || echo 'not set (optional)')"
