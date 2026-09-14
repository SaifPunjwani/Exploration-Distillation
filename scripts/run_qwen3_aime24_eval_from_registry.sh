#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REGISTRY_CAMPAIGN_ROOT="${REGISTRY_CAMPAIGN_ROOT:-runs/qwen3_two_model_campaign}"
RAW_MANIFEST="${RAW_MANIFEST:-configs/qwen3_aime24_raw.tsv}"
EVAL_GROUP="${EVAL_GROUP:-qwen3_aime24_first_eval_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-tpu}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-8}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_TOP_K="${BENCHMARK_ROLLOUT_TOP_K:-20}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-8192}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-qwen3_aime24_first_eval}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$HOME/.venv_tpu_slice/bin/python" ]; then
    PYTHON_BIN="$HOME/.venv_tpu_slice/bin/python"
  elif [ -x ".venv_tpu_slice/bin/python" ]; then
    PYTHON_BIN=".venv_tpu_slice/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

REGISTRY_DIR="${REGISTRY_DIR:-$REGISTRY_CAMPAIGN_ROOT/registry}"
REGISTRY_JSON="${REGISTRY_JSON:-$REGISTRY_DIR/campaign_registry.json}"
TEMP_MANIFEST="$(mktemp /tmp/qwen3_aime24_eval_manifest.XXXXXX.tsv)"
trap 'rm -f "$TEMP_MANIFEST"' EXIT

if [ ! -f "$RAW_MANIFEST" ]; then
  echo "Missing raw manifest: $RAW_MANIFEST" >&2
  exit 1
fi

if [ ! -f "$REGISTRY_JSON" ]; then
  python3 scripts/build_campaign_artifact_registry.py --campaign-root "$REGISTRY_CAMPAIGN_ROOT" --out-dir "$REGISTRY_DIR" >/dev/null
fi

python3 - "$RAW_MANIFEST" "$REGISTRY_JSON" "$TEMP_MANIFEST" <<'PY'
import csv
import json
import sys
from pathlib import Path

raw_manifest_path = Path(sys.argv[1])
registry_json_path = Path(sys.argv[2])
out_manifest_path = Path(sys.argv[3])

rows = []

with raw_manifest_path.open("r", encoding="utf-8") as handle:
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        while len(row) < 5:
            row.append("")
        rows.append(row[:5])

with registry_json_path.open("r", encoding="utf-8") as handle:
    registry = json.load(handle)

for entry in registry:
    model_name = entry.get("model_name") or ""
    run_name = entry.get("run_name") or ""
    base_model_dir = entry.get("base_model_dir") or ""
    actual_model_dir = entry.get("actual_model_dir") or ""

    if base_model_dir:
        rows.append(
            [
                f"{run_name}_base",
                "local",
                base_model_dir,
                "base",
                f"campaign base bundle for {model_name}",
            ]
        )
    if actual_model_dir:
        rows.append(
            [
                f"{run_name}_actual",
                "local",
                actual_model_dir,
                "actual",
                f"campaign actual bundle for {model_name}",
            ]
        )

with out_manifest_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["# label", "source_type", "model_ref", "role_label", "notes"])
    for row in rows:
        writer.writerow(row)
PY

echo "[qwen3-aime24-eval] manifest: $TEMP_MANIFEST"
RUN_GROUP="$EVAL_GROUP" \
MANIFEST_PATH="$TEMP_MANIFEST" \
DEVICE="$DEVICE" \
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}" \
BENCHMARK_NUM_ROLLOUTS="$BENCHMARK_NUM_ROLLOUTS" \
BENCHMARK_ROLLOUT_TEMPERATURE="$BENCHMARK_ROLLOUT_TEMPERATURE" \
BENCHMARK_ROLLOUT_TOP_P="$BENCHMARK_ROLLOUT_TOP_P" \
BENCHMARK_ROLLOUT_TOP_K="$BENCHMARK_ROLLOUT_TOP_K" \
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
BENCHMARK_EVAL_BATCH_SIZE="$BENCHMARK_EVAL_BATCH_SIZE" \
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}" \
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}" \
USE_WANDB="$USE_WANDB" \
WANDB_GROUP="$WANDB_GROUP" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_aime_model_herd.sh
