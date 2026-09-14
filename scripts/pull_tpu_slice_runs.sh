#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

. scripts/tpu_target_lib.sh
resolve_tpu_target_defaults
resolve_tpu_worker_env

TPU_NAME="${TPU_NAME}"
ZONE="${ZONE}"
TPU_TRANSPORT_FLAG="$(expdis_tpu_transport_flag)"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
RUN_ROOT_GLOB="${RUN_ROOT_GLOB:-main_recipe_*}"
PULL_WORKERS="${PULL_WORKERS:-${WORKERS}}"
LIST_ONLY="${LIST_ONLY:-0}"
LOCAL_DEST_ROOT="${LOCAL_DEST_ROOT:-$ROOT_DIR/runs_tpu_pull/${TPU_NAME}_$(date +%Y%m%d_%H%M%S)}"

if [ "${PULL_WORKERS}" = "all" ]; then
  PULL_WORKERS="${WORKERS}"
fi
PULL_WORKERS="${PULL_WORKERS//,/ }"

mkdir -p "${LOCAL_DEST_ROOT}"

list_worker_runs() {
  local worker="$1"
  expdis_gcloud alpha compute tpus tpu-vm ssh \
    "${TPU_NAME}" \
    --worker="${worker}" \
    --zone="${ZONE}" \
    "${TPU_TRANSPORT_FLAG}" \
    --command="python3 - <<'PY' '${REMOTE_ROOT}/runs' '${RUN_ROOT_GLOB}'
import glob
import os
import sys

root = sys.argv[1]
pattern = sys.argv[2]
os.chdir(root)
for path in sorted(glob.glob(pattern)):
    if os.path.isdir(path):
        print(path)
PY" \
    2>/dev/null || true
}

pull_worker_run() {
  local worker="$1"
  local run_name="$2"
  local local_worker_dir="${LOCAL_DEST_ROOT}/worker_${worker}"
  mkdir -p "${local_worker_dir}"
  echo "[pull] worker=${worker} run=${run_name} -> ${local_worker_dir}"
  expdis_gcloud alpha compute tpus tpu-vm scp \
    --recurse \
    "${TPU_NAME}:${REMOTE_ROOT}/runs/${run_name}" \
    "${local_worker_dir}/" \
    --worker="${worker}" \
    --zone="${ZONE}" \
    "${TPU_TRANSPORT_FLAG}"
}

total_runs=0
for worker in ${PULL_WORKERS}; do
  echo "[list] worker=${worker} pattern=${RUN_ROOT_GLOB}"
  run_names="$(list_worker_runs "${worker}")"
  if [ -z "${run_names}" ]; then
    echo "[list] worker=${worker} no matches"
    continue
  fi

  printf '%s\n' "${run_names}"
  if [ "${LIST_ONLY}" = "1" ]; then
    continue
  fi

  while IFS= read -r run_name; do
    [ -n "${run_name}" ] || continue
    pull_worker_run "${worker}" "${run_name}"
    total_runs=$((total_runs + 1))
  done <<< "${run_names}"
done

if [ "${LIST_ONLY}" = "1" ]; then
  echo "[done] listing complete"
else
  echo "[done] pulled ${total_runs} run directories into ${LOCAL_DEST_ROOT}"
fi
