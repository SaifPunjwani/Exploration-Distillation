#!/usr/bin/env bash
# Keep TPU root disks from filling during large Orbax checkpoint saves.
#
# Safe defaults:
# - Always removes stale *.orbax-checkpoint-tmp dirs.
# - Clears pip/conda package caches only when free disk is below threshold.
# - Removes aged local step_* checkpoint dirs only when still below threshold.
#   Checkpoints are expected to be uploaded to HF by train.py; local removal is
#   only a disk-pressure relief valve for long-running TPU jobs.
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
MIN_FREE_GB="${MIN_FREE_GB:-35}"
LOCAL_CKPT_MIN_AGE_MIN="${LOCAL_CKPT_MIN_AGE_MIN:-45}"
DELETE_OLD_RUNS="${DELETE_OLD_RUNS:-0}"
OLD_RUN_MIN_AGE_MIN="${OLD_RUN_MIN_AGE_MIN:-120}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/two-model-exploration/runs}"
ALT_RUNS_ROOT="${ALT_RUNS_ROOT:-$HOME/tmx-jax/runs}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-$HOME/active_jax_run_name}"
GCS_BUCKET_ROOT="${GCS_BUCKET_ROOT:-}"
TMX_ALLOW_GCS_ARTIFACTS="${TMX_ALLOW_GCS_ARTIFACTS:-0}"

free_gb() {
  df -Pk "$HOME" | awk 'NR==2 { printf "%.0f", $4 / 1024 / 1024 }'
}

cleanup_temps() {
  for root in "$RUNS_ROOT" "$ALT_RUNS_ROOT"; do
    [ -d "$root" ] || continue
    find "$root" -type d -name "*.orbax-checkpoint-tmp" -mmin +2 -prune -print -exec rm -rf {} + 2>/dev/null || true
  done
}

cleanup_caches() {
  for path in "$HOME/.cache/pip" "$HOME/miniforge3/pkgs"; do
    [ -e "$path" ] || continue
    echo "[janitor] clearing cache: $path"
    rm -rf "$path" 2>/dev/null || true
    mkdir -p "$path" 2>/dev/null || true
  done
}

cleanup_aged_local_checkpoints() {
  local active_run active_dir
  active_run=""
  if [ -f "$ACTIVE_RUN_FILE" ]; then
    active_run="$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE" || true)"
  fi
  [ -n "$active_run" ] || return 0
  active_dir="$RUNS_ROOT/$active_run"
  [ -d "$active_dir" ] || return 0

  local gcs_base
  gcs_base="${TMX_GCS_CHECKPOINT_BASE:-$GCS_BUCKET_ROOT/$active_run}"

  checkpoint_uploaded() {
    local ckpt_name="$1"
    local candidate
    for candidate in \
      "${gcs_base%/}/$ckpt_name" \
      "${gcs_base%/}/$active_run/$ckpt_name" \
      "${GCS_BUCKET_ROOT%/}/$active_run/$ckpt_name"; do
      gsutil -q ls "$candidate/**" >/dev/null 2>&1 && return 0
    done
    return 1
  }

  # Delete local checkpoint dirs only after GCS visibility is confirmed. Never
  # remove temp dirs here and never anything younger than LOCAL_CKPT_MIN_AGE_MIN.
  find "$active_dir" -path "*/step_[0-9][0-9][0-9][0-9][0-9][0-9]" -type d \
    -mmin +"$LOCAL_CKPT_MIN_AGE_MIN" -print 2>/dev/null \
    | while IFS= read -r ckpt_dir; do
        [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && break
        if ! checkpoint_uploaded "$(basename "$ckpt_dir")"; then
          echo "[janitor] preserving local checkpoint not yet visible in GCS: $ckpt_dir"
          continue
        fi
        echo "[janitor] removing aged local checkpoint under disk pressure: $ckpt_dir"
        rm -rf "$ckpt_dir" 2>/dev/null || true
      done
}

cleanup_inactive_old_runs() {
  [ "$DELETE_OLD_RUNS" = "1" ] || return 0
  [ -d "$RUNS_ROOT" ] || return 0
  local active_run active_dir
  active_run=""
  if [ -f "$ACTIVE_RUN_FILE" ]; then
    active_run="$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE" || true)"
  fi
  active_dir="$RUNS_ROOT/$active_run"
  find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -mmin +"$OLD_RUN_MIN_AGE_MIN" -print 2>/dev/null \
    | while IFS= read -r run_dir; do
        [ "$(free_gb)" -ge "$MIN_FREE_GB" ] && break
        [ -n "$active_run" ] && [ "$run_dir" = "$active_dir" ] && continue
        echo "[janitor] removing inactive old run dir under disk pressure: $run_dir"
        rm -rf "$run_dir" 2>/dev/null || true
      done
}

run_once() {
  local free
  cleanup_temps
  free="$(free_gb)"
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "[janitor] low disk: ${free}GiB free below ${MIN_FREE_GB}GiB"
    cleanup_caches
    cleanup_aged_local_checkpoints
    cleanup_inactive_old_runs
  fi
  echo "[janitor] free disk: $(free_gb)GiB"
}

if [ "${JANITOR_ONCE:-0}" = "1" ]; then
  run_once
  exit 0
fi

while true; do
  run_once
  sleep "$INTERVAL_SECONDS"
done
