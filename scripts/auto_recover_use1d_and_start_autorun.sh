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

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-us-east1-d}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
DIRECT_SSH_HOST="${DIRECT_SSH_HOST:?set DIRECT_SSH_HOST to the ssh host alias of the TPU VM}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0+cpu}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.6.0}"
SSH_MAX_ATTEMPTS="${SSH_MAX_ATTEMPTS:-6}"
SSH_SLEEP_SECONDS="${SSH_SLEEP_SECONDS:-10}"
POLL_SECONDS="${POLL_SECONDS:-20}"
DELETE_POLL_SECONDS="${DELETE_POLL_SECONDS:-10}"
DELETE_MAX_POLLS="${DELETE_MAX_POLLS:-18}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
REMOTE_AUTORUN_LOG="${REMOTE_AUTORUN_LOG:-runs/use1d_remote_autorun_latest.log}"
REMOTE_AUTORUN_SESSION="${REMOTE_AUTORUN_SESSION:-use1d_remote_autorun}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

qr_state() {
  gcloud alpha compute tpus queued-resources describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(state.state)' 2>/dev/null || true
}

node_state() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(state)' 2>/dev/null || true
}

node_health() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(health)' 2>/dev/null || true
}

node_ip() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(networkEndpoints[0].ipAddress)' 2>/dev/null || true
}

delete_lane() {
  log "deleting queued resource / node for $TPU_NAME"
  gcloud alpha compute tpus queued-resources delete "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --force \
    --async \
    --quiet >/dev/null 2>&1 || true
  gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --async \
    --quiet >/dev/null 2>&1 || true

  local attempt=0
  while [ "$attempt" -lt "$DELETE_MAX_POLLS" ]; do
    attempt=$((attempt + 1))
    local qr state health
    qr="$(qr_state)"
    state="$(node_state)"
    health="$(node_health)"
    log "post-delete attempt=$attempt qr=${qr:-NONE} node=${state:-NONE} health=${health:-NONE}"
    if [ -z "${qr}" ] && [ -z "${state}" ]; then
      return 0
    fi
    sleep "$DELETE_POLL_SECONDS"
  done

  log "delete did not fully clear within poll budget; continuing"
}

recreate_lane() {
  log "recreating queued resource $TPU_NAME"
  bash scripts/recreate_v6e_qr.sh "$TPU_NAME" "$ZONE" v2-alpha-tpuv6e
}

wait_for_ready_healthy() {
  while true; do
    local qr state health ip
    qr="$(qr_state)"
    state="$(node_state)"
    health="$(node_health)"
    ip="$(node_ip)"
    log "qr=${qr:-NONE} node=${state:-NONE} health=${health:-NONE} ip=${ip:-NONE}"

    case "${qr:-}" in
      FAILED|SUSPENDED|SUSPENDING)
        return 1
        ;;
    esac

    if [ "$state" = "READY" ] && [ "$health" = "HEALTHY" ]; then
      return 0
    fi
    if [ "$health" = "UNHEALTHY_MAINTENANCE" ]; then
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_worker0_ssh() {
  local attempt=0
  while [ "$attempt" -lt "$SSH_MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    log "worker0 ssh check attempt=$attempt"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 \
      "$DIRECT_SSH_HOST" 'echo ok' >/tmp/use1d_worker0_ssh.out 2>/tmp/use1d_worker0_ssh.err
    then
      cat /tmp/use1d_worker0_ssh.out
      return 0
    fi
    tail -n 6 /tmp/use1d_worker0_ssh.err || true
    sleep "$SSH_SLEEP_SECONDS"
  done
  return 1
}

bootstrap_worker0() {
  log "bootstrapping worker0 from GCS assets"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  BOOTSTRAP_WORKERS=0 \
  SYNC_WORKERS=0 \
  DIRECT_SSH_HOST="$DIRECT_SSH_HOST" \
  GCS_ASSET_PREFIX="$GCS_ASSET_PREFIX" \
  RESET_VENV=1 \
  SKIP_INSTALL=0 \
  TORCH_VERSION="$TORCH_VERSION" \
  TORCH_XLA_VERSION="$TORCH_XLA_VERSION" \
  ALLOW_DIRTY=1 \
  bash scripts/bootstrap_tpu_slice.sh
}

start_remote_autorun() {
  local remote_cmd
  remote_cmd="$(cat <<INNER
cd $REMOTE_ROOT &&
tmux kill-session -t use1d_remote_autorun 2>/dev/null || true &&
tmux new-session -d -s use1d_remote_autorun 'bash -lc "cd $REMOTE_ROOT && bash scripts/run_use1d_remote_autorun.sh 2>&1 | tee runs/use1d_remote_autorun_latest.log"' &&
tmux ls | sed -n '/use1d_remote_autorun/p' &&
sleep 2 &&
tail -n 80 runs/use1d_remote_autorun_latest.log
INNER
)"
  log "starting remote autorun tmux session"
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$DIRECT_SSH_HOST" "$remote_cmd"
}

main() {
  while true; do
    if ! wait_for_ready_healthy; then
      delete_lane
      sleep 10
      recreate_lane
      continue
    fi

    if ! wait_for_worker0_ssh; then
      log "worker0 SSH never became ready; recycling lane"
      delete_lane
      sleep 10
      recreate_lane
      continue
    fi

    bootstrap_worker0
    start_remote_autorun
    log "remote autorun started successfully"
    exit 0
  done
}

main "$@"
