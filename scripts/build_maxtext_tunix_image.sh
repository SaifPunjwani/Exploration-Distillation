#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${TMX_XPK_CONTAINER_IMAGE:-}"
if [[ -z "$IMAGE" ]]; then
  echo "ERROR: set TMX_XPK_CONTAINER_IMAGE, e.g. europe-west4-docker.pkg.dev/<gcp-project>/tmx/maxtext-tunix:smoke" >&2
  exit 2
fi

python3 scripts/tmx_submit.py doctor configs tmx/orchestration scripts/trc2_v5lite256_guard.sh

docker build \
  -f docker/maxtext-tunix/Dockerfile \
  -t "$IMAGE" \
  .

docker push "$IMAGE"

echo "[tmx] pushed $IMAGE"
