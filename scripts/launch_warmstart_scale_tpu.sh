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

. scripts/tpu_target_lib.sh

export TPU_PROJECT="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}"
export TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
export ZONE="${ZONE:-europe-west4-b}"
resolve_tpu_worker_env

SEEDS=(${SEEDS:-0 1 2 3})
RECIPE_MODE="${RECIPE_MODE:-stable_trace_warmstart}"
RUN_PROFILE="${RUN_PROFILE:-stable_trace}"
RECIPE_OFFSET="${RECIPE_OFFSET:-0}"
TRAIN_SELECTION_POLICY="${TRAIN_SELECTION_POLICY:-random}"
TRAIN_SELECTION_MANIFEST_PATH="${TRAIN_SELECTION_MANIFEST_PATH:-}"
QUEUE_TAG_BASE="${QUEUE_TAG_BASE:-warmstart_scale_$(date +%Y%m%d_%H%M%S)}"
SLOT_CHIPS="${SLOT_CHIPS:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-64}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-64}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-24}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-24}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-320}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
RESET_VENV="${RESET_VENV:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-1}"
FILTER_BUSY_WORKERS="${FILTER_BUSY_WORKERS:-0}"

workers=(${WORKERS})
chips=()
for chip in ${SLOT_CHIPS//,/ }; do
  [ -n "$chip" ] || continue
  chips+=("$chip")
done

if [ "${#workers[@]}" -eq 0 ] || [ "${#chips[@]}" -eq 0 ]; then
  echo "No worker/chip slots resolved."
  exit 1
fi

if [ "$FILTER_BUSY_WORKERS" = "1" ]; then
  declare -a available_workers=()
  for worker in "${workers[@]}"; do
    busy_output="$(
      tmx_gcloud alpha compute tpus tpu-vm ssh \
        "$TPU_NAME" \
        --worker="$worker" \
        --zone="$ZONE" \
        "$(tmx_tpu_transport_flag)" \
        --command="pgrep -af 'python .*main.py --device tpu' || true" 2>/dev/null || true
    )"
    if printf '%s\n' "$busy_output" | grep -q 'main.py --device tpu'; then
      echo "[launch] skipping busy worker=$worker"
      continue
    fi
    available_workers+=("$worker")
  done
  workers=("${available_workers[@]}")
  if [ "${#workers[@]}" -eq 0 ]; then
    echo "[launch] no free workers remained after busy-worker filtering."
    exit 1
  fi
fi

slot_workers=()
slot_chips=()
for worker in "${workers[@]}"; do
  for chip in "${chips[@]}"; do
    slot_workers+=("$worker")
    slot_chips+=("$chip")
  done
done

if [ "${#SEEDS[@]}" -gt "${#slot_workers[@]}" ]; then
  echo "Need ${#SEEDS[@]} slots but only ${#slot_workers[@]} available from workers=${WORKERS} chips=${SLOT_CHIPS}."
  exit 1
fi

echo "[launch] target TPU=$TPU_NAME zone=$ZONE workers=$WORKERS chips=$SLOT_CHIPS"
echo "[launch] model=$MODEL_NAME dataset=$DATASET_NAME seeds=${SEEDS[*]} recipe_mode=$RECIPE_MODE offset=$RECIPE_OFFSET"

if [ "$SKIP_BOOTSTRAP" != "1" ]; then
  ALLOW_DIRTY="$ALLOW_DIRTY" \
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  SKIP_INSTALL="$SKIP_INSTALL" \
  RESET_VENV="$RESET_VENV" \
  bash scripts/bootstrap_tpu_slice.sh
fi

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  worker="${slot_workers[$idx]}"
  chip="${slot_chips[$idx]}"
  queue_tag="${QUEUE_TAG_BASE}_s${seed}"
  session_prefix="warmstart_scale_s${seed}"
  echo "[launch] seed=$seed worker=$worker chip=$chip queue_tag=$queue_tag"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  SKIP_BOOTSTRAP=1 \
  WORKERS="$worker" \
  CHIPS="$chip" \
  SEED="$seed" \
  RECIPE_MODE="$RECIPE_MODE" \
  RUN_PROFILE="$RUN_PROFILE" \
  RUN_OFFSET="$RECIPE_OFFSET" \
  RUN_LIMIT=1 \
  TRAIN_SELECTION_POLICY="$TRAIN_SELECTION_POLICY" \
  TRAIN_SELECTION_MANIFEST_PATH="$TRAIN_SELECTION_MANIFEST_PATH" \
  QUEUE_TAG="$queue_tag" \
  SESSION_PREFIX="$session_prefix" \
  MODEL_NAME="$MODEL_NAME" \
  DATASET_NAME="$DATASET_NAME" \
  DATASET_CONFIG_NAME="$DATASET_CONFIG_NAME" \
  MAX_TRAIN_EXAMPLES="$MAX_TRAIN_EXAMPLES" \
  MAX_EVAL_EXAMPLES="$MAX_EVAL_EXAMPLES" \
  MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
  MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
  EXPLORATION_MAX_COMPLETION_LEN="$EXPLORATION_MAX_COMPLETION_LEN" \
  EVAL_MAX_COMPLETION_LEN="$EVAL_MAX_COMPLETION_LEN" \
  MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
  bash scripts/queue_gsm8k_recipe_search_tmux.sh
done

echo "[launch] complete"
