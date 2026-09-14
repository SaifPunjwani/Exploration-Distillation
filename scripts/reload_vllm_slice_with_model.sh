#!/usr/bin/env bash
# Reload a full TPU inference slice with a model from GCS by restarting each worker's
# local vLLM server against the downloaded model bundle.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
. scripts/tpu_target_lib.sh

GCS_MODEL_PATH="${GCS_MODEL_PATH:?Must set GCS_MODEL_PATH}"
TPU_NAME="${TPU_NAME:?Must set TPU_NAME}"
ZONE="${ZONE:?Must set ZONE}"
WORKERS="${WORKERS:-all}"
PARALLELISM="${PARALLELISM:-16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-explorer_model}"
LOCAL_MODEL_DIR_BASENAME="${LOCAL_MODEL_DIR_BASENAME:-vllm_model}"

describe_json="$(expdis_gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format=json)"
worker_count="$(python3 - "$describe_json" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
eps = payload.get("networkEndpoints") or payload.get("workerNetworkEndpoints") or []
print(len(eps))
PY
)"

if [ "$WORKERS" = "all" ]; then
  worker_list="$(seq 0 $((worker_count - 1)))"
else
  worker_list="$WORKERS"
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

failures=0
active=0

# Pre-warm SSH key once (serial). Parallel gcloud SSH invocations race on
# ssh-keygen regeneration, causing 14/15 workers to fail with "key file
# already exists. Overwrite (y/n)?". A single pre-warm ensures the key is
# in place before fanning out, so parallel calls skip the regeneration.
expdis_gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
  --worker=0 \
  --zone="$ZONE" \
  "$(expdis_tpu_transport_flag)" \
  --command="true" >/dev/null 2>&1 || true

for worker in $worker_list; do
  (
    GCS_MODEL_PATH="$GCS_MODEL_PATH" \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    WORKER="$worker" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    LOCAL_MODEL_DIR_BASENAME="$LOCAL_MODEL_DIR_BASENAME" \
    PORT="${PORT:-8000}" \
    TP_SIZE="${TP_SIZE:-4}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}" \
    VLLM_VENV_DIR="${VLLM_VENV_DIR:?set VLLM_VENV_DIR to the vLLM virtualenv path on the serving TPU VM}" \
    REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}" \
    bash scripts/reload_vllm_with_model.sh >"$tmp_dir/worker_${worker}.log" 2>&1
  ) &
  active=$((active + 1))
  if [ "$active" -ge "$PARALLELISM" ]; then
    wait || failures=$((failures + 1))
    active=0
  fi
done
wait || failures=$((failures + 1))

for worker in $worker_list; do
  if [ -f "$tmp_dir/worker_${worker}.log" ]; then
    printf '[reload-slice] worker=%s %s\n' "$worker" "$(tail -n 2 "$tmp_dir/worker_${worker}.log" | tr '\n' ' ')" || true
  fi
done

if [ "$failures" -ne 0 ]; then
  echo "[reload-slice] ERROR: one or more worker reloads failed" >&2
  exit 1
fi

echo "[reload-slice] all requested workers reloaded successfully"
