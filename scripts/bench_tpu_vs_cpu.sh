#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-bench_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/$RUN_ID}"
mkdir -p "$OUT_DIR"

MODEL_NAME="${MODEL_NAME:-sshleifer/tiny-gpt2}"
MAX_TRAIN="${MAX_TRAIN:-16}"
MAX_EVAL="${MAX_EVAL:-16}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-2}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-2}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-64}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-32}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-96}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-raw}"
ACTUAL_TRAIN_MODE="${ACTUAL_TRAIN_MODE:-sft_grpo}"

VENV_CPU="${VENV_CPU:-$ROOT_DIR/.venv}"
VENV_TPU="${VENV_TPU:-$ROOT_DIR/.venv_tpu}"

run_case() {
  local label="$1"
  local device="$2"
  local venv_dir="$3"
  local log_path="$OUT_DIR/${label}.log"

  if [ ! -x "$venv_dir/bin/python" ]; then
    echo "[ERROR] Missing venv: $venv_dir"
    exit 1
  fi

  # shellcheck disable=SC1091
  source "$venv_dir/bin/activate"

  if [ "$device" = "tpu" ]; then
    export PJRT_DEVICE=TPU
    export USE_TORCH_XLA=1
  else
    unset PJRT_DEVICE USE_TORCH_XLA TPU_LIBRARY_PATH
  fi

  local start_ts
  start_ts="$(date +%s)"
  python run_expdis_torch.py --device "$device" --no-wandb \
    --model-name "$MODEL_NAME" \
    --decoupling-mode decoupled \
    --actual-train-mode "$ACTUAL_TRAIN_MODE" \
    --base-eval-mode "$BASE_EVAL_MODE" \
    --max-train-examples "$MAX_TRAIN" \
    --max-eval-examples "$MAX_EVAL" \
    --grpo-max-steps "$GRPO_MAX_STEPS" \
    --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
    --grpo-generation-batch-size "$GRPO_GEN_BATCH" \
    --num-samples-per-prompt "$NUM_SAMPLES" \
    --max-prompt-len "$MAX_PROMPT_LEN" \
    --max-completion-len "$MAX_COMPLETION_LEN" \
    --max-total-len "$MAX_TOTAL_LEN" \
    --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS" \
    --distill-epochs "$DISTILL_EPOCHS" \
    >"$log_path" 2>&1
  local end_ts
  end_ts="$(date +%s)"
  local elapsed=$((end_ts - start_ts))

  local base_acc explorer_acc actual_acc
  base_acc="$(rg -m1 "Base accuracy" "$log_path" | sed -E 's/.*Base accuracy: ([0-9.]+)%.*/\1/' || true)"
  explorer_acc="$(rg -m1 "Explorer accuracy" "$log_path" | sed -E 's/.*Explorer accuracy: ([0-9.]+)%.*/\1/' || true)"
  actual_acc="$(rg -m1 "Actual accuracy" "$log_path" | sed -E 's/.*Actual accuracy: ([0-9.]+)%.*/\1/' || true)"

  echo "$label wall_time_sec=$elapsed base_acc_pct=$base_acc explorer_acc_pct=$explorer_acc actual_acc_pct=$actual_acc"
}

echo "Benchmark run: $RUN_ID"
echo "Model: $MODEL_NAME | train=$MAX_TRAIN eval=$MAX_EVAL | steps=$GRPO_MAX_STEPS | device modes: cpu,tpu"

run_case "cpu" "cpu" "$VENV_CPU"

if [ "${SKIP_TPU:-0}" != "1" ]; then
  run_case "tpu" "tpu" "$VENV_TPU"
else
  echo "SKIP_TPU=1 set; TPU run skipped."
fi

echo "Logs written to: $OUT_DIR"
