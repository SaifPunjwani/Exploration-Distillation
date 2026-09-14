#!/usr/bin/env bash
# Legacy GCS artifact uploader. Disabled by default; use HF artifact uploads
# from the pipeline unless explicitly recovering an old GCS workflow.
set -euo pipefail

RUN_NAME="${RUN_NAME:?Must set RUN_NAME}"
RUNS_ROOT="${RUNS_ROOT:-$HOME/expdis/runs}"
ROOT="$RUNS_ROOT/$RUN_NAME"
INTERVAL_SECONDS="${EXPDIS_ARTIFACT_UPLOAD_INTERVAL_SECONDS:-300}"
GCS_BASE="${EXPDIS_GCS_CHECKPOINT_BASE:-}"
if [ "${EXPDIS_ALLOW_GCS_ARTIFACTS:-0}" != "1" ] || [ -z "$GCS_BASE" ]; then
  echo "[artifact-uploader] GCS upload disabled; exiting"
  exit 0
fi

step_from_jsonl() {
  local file="$1"
  python3 - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
last = -1
try:
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            for key in ("step", "explorer_step", "actual_step"):
                if key in row:
                    try:
                        last = max(last, int(row[key]))
                    except Exception:
                        pass
                    break
except FileNotFoundError:
    pass
print(last)
PY
}

upload_jsonl() {
  local file="$1"
  local kind="$2"
  [ -s "$file" ] || return 0
  local step
  step="$(step_from_jsonl "$file")"
  local gcs_dir="${GCS_BASE%/}/trajectories"
  local tmp="/tmp/$(basename "$file").$$"
  cp "$file" "$tmp"
  gsutil cp "$tmp" "$gcs_dir/${kind}_latest.jsonl" >/dev/null
  if [ "$step" -ge 0 ]; then
    gsutil cp "$tmp" "$gcs_dir/${kind}_step_$(printf '%06d' "$step").jsonl" >/dev/null
  fi
  rm -f "$tmp"
  echo "[artifact-upload] uploaded $kind step=$step -> $gcs_dir"
}

echo "[artifact-upload] watching $ROOT every ${INTERVAL_SECONDS}s -> ${GCS_BASE%/}/trajectories"
while true; do
  upload_jsonl "$ROOT/explorer/explorer_trajectories.jsonl" "explorer_trajectories" || true
  upload_jsonl "$ROOT/actual/grpo/explorer_trajectories.jsonl" "actual_grpo_trajectories" || true
  sleep "$INTERVAL_SECONDS"
done
