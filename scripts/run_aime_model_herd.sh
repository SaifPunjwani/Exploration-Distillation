#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

MANIFEST_PATH="${MANIFEST_PATH:-configs/aime_model_herd.tsv}"
RUN_GROUP="${RUN_GROUP:-aime_model_herd_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
DEVICE="${DEVICE:-cpu}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-32}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
USE_WANDB="${USE_WANDB:-auto}"

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

OUTPUT_ROOT="runs/${RUN_GROUP}"
mkdir -p "$OUTPUT_ROOT"

RESULTS_JSON="${OUTPUT_ROOT}/herd_results.json"
RESULTS_CSV="${OUTPUT_ROOT}/herd_results.csv"

rows_json="[]"

resolve_model_ref() {
  local label="$1"
  local source_type="$2"
  local model_ref="$3"

  case "$source_type" in
    hf|local)
      printf '%s\n' "$model_ref"
      ;;
    wandb|wandb_artifact|artifact)
      local artifact_root="${OUTPUT_ROOT}/artifacts/${label}"
      mkdir -p "$artifact_root"
      "$PYTHON_BIN" - <<'PY' "$model_ref" "$artifact_root"
import os
import sys

artifact_ref, artifact_root = sys.argv[1:]
import wandb  # type: ignore

api = wandb.Api()
artifact = api.artifact(artifact_ref)
path = artifact.download(root=artifact_root)
print(os.path.abspath(path))
PY
      ;;
    *)
      echo "Unsupported source_type=$source_type for label=$label" >&2
      return 1
      ;;
  esac
}

while IFS=$'\t' read -r col1 col2 col3 col4 col5 _extra; do
  [ -n "${col1:-}" ] || continue
  case "$col1" in
    \#*) continue ;;
  esac

  label="$col1"
  source_type="hf"
  role_label="raw_base"
  model_ref=""
  notes=""

  case "${col2:-}" in
    hf|local|wandb|wandb_artifact|artifact)
      source_type="$col2"
      model_ref="${col3:-}"
      role_label="${col4:-raw_base}"
      notes="${col5:-}"
      ;;
    *)
      model_ref="${col2:-}"
      notes="${col3:-}"
      ;;
  esac
  [ -n "${model_ref:-}" ] || continue

  case "$role_name" in
    base|explorer|actual) ;;
    *) role_name="base" ;;
  esac

  resolved_model_ref="$(resolve_model_ref "$label" "$source_type" "$model_ref")"

  run_name="${RUN_GROUP}_${label}"
  echo "[herd] running label=${label} source=${source_type} role=${role_label} model=${resolved_model_ref}"

  summary_path="runs/${run_name}/analysis/benchmark_summary.json"
  run_status="success"
  error_message=""

  if ! DEVICE="$DEVICE" \
    MODEL_NAME="$resolved_model_ref" \
    RUN_NAME="$run_name" \
    BENCHMARK_DATASET_NAME="$BENCHMARK_DATASET_NAME" \
    BENCHMARK_DATASET_CONFIG_NAME="$BENCHMARK_DATASET_CONFIG_NAME" \
    MAX_BENCHMARK="$MAX_BENCHMARK" \
    BENCHMARK_NUM_ROLLOUTS="$BENCHMARK_NUM_ROLLOUTS" \
    BENCHMARK_ROLLOUT_TEMPERATURE="$BENCHMARK_ROLLOUT_TEMPERATURE" \
    BENCHMARK_ROLLOUT_TOP_P="$BENCHMARK_ROLLOUT_TOP_P" \
    BENCHMARK_ROLLOUT_PASS_K="$BENCHMARK_ROLLOUT_PASS_K" \
    BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
    BENCHMARK_EVAL_BATCH_SIZE="$BENCHMARK_EVAL_BATCH_SIZE" \
    MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
    MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
    USE_WANDB="$USE_WANDB" \
    bash scripts/run_aime_rollout_eval.sh
  then
    run_status="run_failed"
    error_message="run_aime_rollout_eval.sh exited non-zero"
  elif [ ! -f "$summary_path" ]; then
    run_status="missing_summary"
    error_message="benchmark_summary.json not found"
  fi

  row="$("$PYTHON_BIN" - <<'PY' "$summary_path" "$label" "$source_type" "$model_ref" "$resolved_model_ref" "$role_label" "$notes" "$run_name" "$run_status" "$error_message"
import json
import sys

summary_path, label, source_type, model_ref, resolved_model_ref, role_label, notes, run_name, run_status, error_message = sys.argv[1:]
summary = {}
if run_status == "success":
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

# run_aime_rollout_eval.sh always evaluates the loaded checkpoint under the
# "base" role, regardless of what the herd row labels it; role_label is a
# pass-through CSV column only.
role = "base"
protocol = summary.get("protocol") or {}
accuracies = summary.get("accuracies") or {}
pass_at_k = ((summary.get("pass_at_k") or {}).get(role) or {})
payload = {
    "label": label,
    "status": run_status,
    "error_message": error_message,
    "source_type": source_type,
    "role_label": role_label,
    "model_ref": model_ref,
    "resolved_model_ref": resolved_model_ref,
    "notes": notes,
    "run_name": run_name,
    "summary_path": summary_path,
    "dataset_name": summary.get("dataset_name"),
    "dataset_config_name": summary.get("dataset_config_name"),
    "eval_mode": protocol.get("eval_mode"),
    "num_rollouts": protocol.get("num_rollouts"),
    "rollout_temperature": protocol.get("rollout_temperature"),
    "rollout_top_p": protocol.get("rollout_top_p"),
    "rollout_max_completion_len": protocol.get("rollout_max_completion_len"),
    "primary_metric": (summary.get("primary_metrics") or {}).get(role),
    "avg_at_n": accuracies.get(role),
    "pass_at_1": pass_at_k.get("1"),
    "pass_at_4": pass_at_k.get("4"),
    "pass_at_8": pass_at_k.get("8"),
    "pass_at_16": pass_at_k.get("16"),
    "pass_at_32": pass_at_k.get("32"),
    "rollout_json_path": (summary.get("rollout_json_paths") or {}).get(role),
}
print(json.dumps(payload))
PY
)"

  rows_json="$("$PYTHON_BIN" - <<'PY' "$rows_json" "$row"
import json
import sys

rows = json.loads(sys.argv[1])
rows.append(json.loads(sys.argv[2]))
print(json.dumps(rows))
PY
)"
done < "$MANIFEST_PATH"

printf '%s\n' "$rows_json" > "$RESULTS_JSON"

"$PYTHON_BIN" - <<'PY' "$RESULTS_JSON" "$RESULTS_CSV"
import csv
import json
import sys

json_path, csv_path = sys.argv[1:]
with open(json_path, "r", encoding="utf-8") as f:
    rows = json.load(f)

fieldnames = [
    "label",
    "status",
    "error_message",
    "source_type",
    "role_label",
    "model_ref",
    "resolved_model_ref",
    "notes",
    "run_name",
    "dataset_name",
    "dataset_config_name",
    "eval_mode",
    "num_rollouts",
    "rollout_temperature",
    "rollout_top_p",
    "rollout_max_completion_len",
    "primary_metric",
    "avg_at_n",
    "pass_at_1",
    "pass_at_4",
    "pass_at_8",
    "pass_at_16",
    "pass_at_32",
    "summary_path",
    "rollout_json_path",
]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY

echo "[herd] complete"
echo "[herd] results json: $RESULTS_JSON"
echo "[herd] results csv:  $RESULTS_CSV"
