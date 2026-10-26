#!/usr/bin/env bash
# Copy this repository to a RunPod pod's network volume and set it up there.
# Uses a git bundle over SSH, so it works for a private repo with no extra
# credentials and keeps real commit hashes for the run manifests.
#
#   scripts/pod_sync.sh nanogpt-pod        # an SSH host alias from ~/.ssh/config
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

host=${1:?usage: scripts/pod_sync.sh <ssh-host>}
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: uncommitted changes are not synced; commit them first" >&2
fi

git bundle create - --all | ssh "$host" 'cat > /workspace/nanogpt.bundle'
ssh "$host" bash -s -- "$branch" <<'REMOTE'
set -euo pipefail
branch=$1
repo=/workspace/nanogpt
if [[ -d "$repo/.git" ]]; then
  git -C "$repo" fetch -q /workspace/nanogpt.bundle "+refs/heads/*:refs/remotes/bundle/*"
  git -C "$repo" reset -q --hard
  git -C "$repo" checkout -q -B "$branch" "bundle/$branch"
else
  git clone -q -b "$branch" /workspace/nanogpt.bundle "$repo"
fi
rm -f /workspace/nanogpt.bundle
echo "synced: $(git -C "$repo" log --oneline -1)"
bash "$repo/scripts/pod_setup.sh"
REMOTE
