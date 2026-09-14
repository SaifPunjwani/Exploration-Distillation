#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-us-east1-d}"
WORKER="${WORKER:-0}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"

WAIT_SECONDS="${WAIT_SECONDS:-60}"
BOOTSTRAP_ON_READY="${BOOTSTRAP_ON_READY:-1}"
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-0}"
SYNC_WORKERS="${SYNC_WORKERS:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
RESET_VENV="${RESET_VENV:-0}"

LOCAL_MONITOR_ROOT="${LOCAL_MONITOR_ROOT:-runs/aime_compare_${TPU_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOCAL_MONITOR_ROOT"
LOG_PATH="${LOCAL_MONITOR_ROOT}/waiter.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_PATH"
}

queued_state() {
  gcloud alpha compute tpus queued-resources list \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --format='value(name,state.state)' 2>/dev/null | awk -v want="$TPU_NAME" '$1==want{print $2}'
}

vm_state() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --format='value(state)' 2>/dev/null || true
}

vm_health() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project "$PROJECT" \
    --zone "$ZONE" \
    --format='value(health)' 2>/dev/null || true
}

start_aime_monitor() {
  log "Starting monitor_aime_compare_remote.sh on $TPU_NAME ($ZONE)"
  PROJECT="$PROJECT" \
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKER="$WORKER" \
  V6E_ZONE="$ZONE" \
  V6E_QUEUED_RESOURCE="$TPU_NAME" \
  LOCAL_MONITOR_ROOT="$LOCAL_MONITOR_ROOT" \
  bash scripts/monitor_aime_compare_remote.sh >> "$LOG_PATH" 2>&1
}

bootstrap_tpu() {
  log "Bootstrapping TPU worker(s): sync=$SYNC_WORKERS bootstrap=$BOOTSTRAP_WORKERS"
  ALLOW_DIRTY="$ALLOW_DIRTY" \
  RESET_VENV="$RESET_VENV" \
  TMX_CLOUDSDK_CONFIG="${TMX_CLOUDSDK_CONFIG:-}" \
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  SYNC_WORKERS="$SYNC_WORKERS" \
  BOOTSTRAP_WORKERS="$BOOTSTRAP_WORKERS" \
  bash scripts/bootstrap_tpu_slice.sh >> "$LOG_PATH" 2>&1
}

log "Waiting for TPU $TPU_NAME in $ZONE"
while true; do
  q_state="$(queued_state)"
  t_state="$(vm_state)"
  t_health="$(vm_health)"
  log "queued_state=${q_state:-missing} vm_state=${t_state:-missing} vm_health=${t_health:-unknown}"

  if [ "$t_state" = "READY" ]; then
    if [ "$BOOTSTRAP_ON_READY" = "1" ]; then
      bootstrap_tpu
    fi
    start_aime_monitor
    exit $?
  fi

  sleep "$WAIT_SECONDS"
done
