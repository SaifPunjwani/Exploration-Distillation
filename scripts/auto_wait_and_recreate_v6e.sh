#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_TPU_NAME="${TARGET_TPU_NAME:?TARGET_TPU_NAME is required}"
TARGET_ZONE="${TARGET_ZONE:?TARGET_ZONE is required}"
TARGET_ACCELERATOR_TYPE="${TARGET_ACCELERATOR_TYPE:-v6e-64}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"
POLL_SECONDS="${POLL_SECONDS:-20}"

qr_state() {
  gcloud alpha compute tpus queued-resources describe "$TARGET_TPU_NAME" \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$TARGET_ZONE" \
    --format='value(state.state)' 2>/dev/null || true
}

node_state() {
  gcloud alpha compute tpus tpu-vm list \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$TARGET_ZONE" \
    --filter="name=${TARGET_TPU_NAME}" \
    --format='value(state)' 2>/dev/null || true
}

node_health() {
  gcloud alpha compute tpus tpu-vm list \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$TARGET_ZONE" \
    --filter="name=${TARGET_TPU_NAME}" \
    --format='value(health)' 2>/dev/null || true
}

delete_qr() {
  gcloud alpha compute tpus queued-resources delete "$TARGET_TPU_NAME" \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
}

delete_node() {
  gcloud alpha compute tpus tpu-vm delete "$TARGET_TPU_NAME" \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
}

while true; do
  current_qr_state="$(qr_state)"
  current_node_state="$(node_state)"
  current_node_health="$(node_health)"

  printf '[%s] qr_state=%s node_state=%s node_health=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${current_qr_state:-NONE}" \
    "${current_node_state:-NONE}" \
    "${current_node_health:-NONE}"

  case "${current_qr_state:-}" in
    FAILED|SUSPENDED|SUSPENDING)
      delete_qr
      delete_node
      sleep 10
      continue
      ;;
  esac

  if [ -z "${current_qr_state:-}" ]; then
    bash scripts/recreate_v6e_qr.sh "$TARGET_TPU_NAME" "$TARGET_ZONE" "$RUNTIME_VERSION" || true
    sleep "$POLL_SECONDS"
    continue
  fi

  if [ "${current_node_state:-}" = "READY" ] && { [ -z "${current_node_health:-}" ] || [ "${current_node_health:-}" = "HEALTHY" ]; }; then
    if TARGET_TPU_NAME="$TARGET_TPU_NAME" \
      TARGET_ZONE="$TARGET_ZONE" \
      TARGET_ACCELERATOR_TYPE="$TARGET_ACCELERATOR_TYPE" \
      RUNTIME_VERSION="$RUNTIME_VERSION" \
      bash scripts/wait_and_handoff_v6e.sh
    then
      exit 0
    fi

    delete_qr
    delete_node
    sleep 10
    continue
  fi

  sleep "$POLL_SECONDS"
done
