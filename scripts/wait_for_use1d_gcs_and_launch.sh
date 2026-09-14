#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

WHEEL_GCS_PATH="${WHEEL_GCS_PATH:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331/wheelhouse.tar}"
HF_GCS_PATH="${HF_GCS_PATH:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331/hf_cache.tar}"
WHEEL_TAR="${WHEEL_TAR:-runs/offline_tpu_assets_20260331_wheelhouse_v3/wheelhouse.tar}"
HF_TAR="${HF_TAR:-runs/offline_tpu_assets_20260331_train/hf_cache.tar}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

until gcloud storage ls "$WHEEL_GCS_PATH" >/dev/null 2>&1 && gcloud storage ls "$HF_GCS_PATH" >/dev/null 2>&1; do
  log "waiting for GCS assets"
  sleep 30
done

log "GCS assets ready; launching use1d full campaign"
tmux kill-session -t use1d_full_campaign 2>/dev/null || true
WHEEL_TAR="$WHEEL_TAR" \
HF_TAR="$HF_TAR" \
GCS_ASSET_PREFIX="$GCS_ASSET_PREFIX" \
bash scripts/run_use1d_full_campaign.sh
