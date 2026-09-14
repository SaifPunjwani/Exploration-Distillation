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

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv_tpu/bin/python" ]; then
    PYTHON_BIN=".venv_tpu/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_TAG="${RUN_TAG:-novelty_full_tpu}"
RUN_ID="${RUN_ID:-4rk13lbq}"
OUTPUT_LOG="${OUTPUT_LOG:-wandb/run-20260217_025156-${RUN_ID}/files/output.log}"
POLL_SECS="${POLL_SECS:-30}"

OUT_DIR="${OUT_DIR:-runs/${RUN_TAG}_postprocess_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

STAMP_FILE="${OUT_DIR}/status.txt"
echo "watch_started=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" > "$STAMP_FILE"
echo "output_log=${OUTPUT_LOG}" >> "$STAMP_FILE"
echo "run_tag=${RUN_TAG}" >> "$STAMP_FILE"

wait_for_finish() {
  while true; do
    if [ -f "$OUTPUT_LOG" ]; then
      if rg -q "Done\\. Check '" "$OUTPUT_LOG"; then
        echo "run_finished_detected=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$STAMP_FILE"
        return 0
      fi
      if rg -q "Traceback \\(most recent call last\\)" "$OUTPUT_LOG"; then
        echo "run_failed_detected=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$STAMP_FILE"
        return 1
      fi
    fi
    sleep "$POLL_SECS"
  done
}

if wait_for_finish; then
  echo "postprocess_start=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$STAMP_FILE"

  "$PYTHON_BIN" scripts/plot_novelty_study.py \
    --root runs \
    --tag "$RUN_TAG" \
    --out-dir "${OUT_DIR}/aggregate" \
    > "${OUT_DIR}/aggregate.log" 2>&1 || true

  "$PYTHON_BIN" scripts/build_experiment_registry.py \
    --wandb-root wandb \
    --out-dir "${OUT_DIR}/registry" \
    > "${OUT_DIR}/registry.log" 2>&1 || true

  "$PYTHON_BIN" scripts/meeting_novelty_snapshot.py \
    --root runs \
    --registry-csv "${OUT_DIR}/registry/experiment_registry.csv" \
    --out-dir "${OUT_DIR}/meeting_snapshot" \
    > "${OUT_DIR}/meeting_snapshot.log" 2>&1 || true

  "$PYTHON_BIN" scripts/build_novelty_validation_pack.py \
    --out-dir "${OUT_DIR}/validation_pack" \
    --seed 13 \
    > "${OUT_DIR}/validation_pack.log" 2>&1 || true

  echo "postprocess_done=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$STAMP_FILE"
else
  echo "postprocess_skipped_due_to_failure=$(date -u +'%Y-%m-%d %H:%M:%S UTC')" >> "$STAMP_FILE"
  exit 1
fi

