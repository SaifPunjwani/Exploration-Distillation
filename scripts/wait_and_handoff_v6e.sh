#!/usr/bin/env bash
set -euo pipefail

TARGET_TPU_NAME="${TARGET_TPU_NAME:?TARGET_TPU_NAME is required}"
TARGET_ZONE="${TARGET_ZONE:?TARGET_ZONE is required}"
TARGET_ACCELERATOR_TYPE="${TARGET_ACCELERATOR_TYPE:-v6e-64}"
QUEUE_TAG="${QUEUE_TAG:-v6e_handoff_$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-20}"

while true; do
  state="$(
    gcloud alpha compute tpus tpu-vm list \
      --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
      --zone "$TARGET_ZONE" \
      --filter="name=${TARGET_TPU_NAME}" \
      --format='value(state)' 2>/dev/null || true
  )"
  printf '[%s] state=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${state:-NONE}"
  if [ "${state:-}" = "READY" ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
gcloud alpha compute tpus tpu-vm ssh "$TARGET_TPU_NAME" \
  --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
  --zone "$TARGET_ZONE" \
  --worker=0 \
  --tunnel-through-iap \
  --command 'echo READY && hostname'

LOCAL_HF_TAR="${LOCAL_HF_TAR:-/tmp/tmx_hf_cache.tar}" \
TARGET_TPU_NAME="$TARGET_TPU_NAME" \
TARGET_ZONE="$TARGET_ZONE" \
TARGET_ACCELERATOR_TYPE="$TARGET_ACCELERATOR_TYPE" \
QUEUE_TAG="$QUEUE_TAG" \
POLL_SECONDS=10 \
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/finalize_v6e_env_and_launch.sh"
