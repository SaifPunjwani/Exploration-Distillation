#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(expdis_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
RUN_ROOT_GLOB="${RUN_ROOT_GLOB:-runs/tpu_slice_*}"

expdis_gcloud alpha compute tpus tpu-vm ssh \
  "$TPU_NAME" \
  --worker=all \
  --batch-size=all \
  --zone="$ZONE" \
  "$TPU_TRANSPORT_FLAG" \
  --command="cd '$REMOTE_ROOT' && echo '== worker:' \$(hostname) '==' && tmux ls || true && ls -td $RUN_ROOT_GLOB 2>/dev/null | head -n 2 | xargs -r -I{} find {} -path '*/_slot_logs/*.log' -type f | sort | tail -n 4 | xargs -r tail -n 5"
