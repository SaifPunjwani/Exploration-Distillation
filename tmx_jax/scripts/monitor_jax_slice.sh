#!/usr/bin/env bash
set -euo pipefail

TPU_NAME="${TPU_NAME:?set TPU_NAME}"
ZONE="${ZONE:?set ZONE}"
RUN_NAME="${RUN_NAME:?set RUN_NAME}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3}"
LOG_FILE="${LOG_FILE:-$HOME/jax_monitor_${RUN_NAME}.log}"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "[$ts] run=$RUN_NAME"
    echo "--- worker0 pane ---"
    ssh_target="${TPU_NAME}"
    gcloud alpha compute tpus tpu-vm ssh "$ssh_target" --worker=0 --zone="$ZONE" --tunnel-through-iap \
      --command="tmux capture-pane -t '$RUN_NAME' -p -S -80 2>/dev/null | tail -40 || echo no_tmux" 2>&1 || true
    echo "--- train workers ---"
    for w in $TRAIN_WORKERS; do
      echo "[worker $w]"
      gcloud alpha compute tpus tpu-vm ssh "$ssh_target" --worker="$w" --zone="$ZONE" --tunnel-through-iap \
        --command="ps -eo pid,etime,%cpu,%mem,cmd | grep 'tmx_jax.train' | grep -v grep || echo no_train_proc" 2>&1 || true
    done
    echo "--- checkpoints worker0 ---"
    gcloud alpha compute tpus tpu-vm ssh "$ssh_target" --worker=0 --zone="$ZONE" --tunnel-through-iap \
      --command="find ~/runs/$RUN_NAME -maxdepth 1 -type d -name 'step_*' | sort | tail -5 || true" 2>&1 || true
    echo
  } | tee -a "$LOG_FILE"
  sleep "$INTERVAL_SECONDS"
done
