#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
SERVER_BASE_URL="${SERVER_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
RUN_PREFIX="${RUN_PREFIX:-use1d_qwen3_1p7b_baseline_suite_$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_RUN_NAME="${WAIT_FOR_RUN_NAME:-}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-use1d_qwen3_baseline_suite}"

run_eval() {
  local run_name="$1"
  local thinking_mode="$2"
  local temperature="$3"
  local top_p="$4"
  local top_k="$5"

  PYTHON_BIN="$VLLM_VENV_DIR/bin/python" \
  SERVER_BASE_URL="$SERVER_BASE_URL" \
  MODEL_NAME="$MODEL_NAME" \
  SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  TOKENIZER_NAME="$TOKENIZER_NAME" \
  RUN_NAME="$run_name" \
  ROLE_NAME="base" \
  BENCHMARK_DATASET_NAME="$BENCHMARK_DATASET_NAME" \
  BENCHMARK_DATASET_CONFIG_NAME="$BENCHMARK_DATASET_CONFIG_NAME" \
  MAX_BENCHMARK="$MAX_BENCHMARK" \
  BENCHMARK_NUM_ROLLOUTS="$BENCHMARK_NUM_ROLLOUTS" \
  BENCHMARK_ROLLOUT_TEMPERATURE="$temperature" \
  BENCHMARK_ROLLOUT_TOP_P="$top_p" \
  BENCHMARK_ROLLOUT_TOP_K="$top_k" \
  BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
  REQUEST_CONCURRENCY="$REQUEST_CONCURRENCY" \
  REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS" \
  MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
  THINKING_MODE="$thinking_mode" \
  USE_WANDB="$USE_WANDB" \
  WANDB_GROUP="$WANDB_GROUP" \
  bash scripts/run_vllm_aime_rollout_eval.sh
}

if [ ! -x "$VLLM_VENV_DIR/bin/python" ]; then
  VLLM_VENV_DIR="$VLLM_VENV_DIR" bash scripts/bootstrap_vllm_tpu_env.sh
fi

if [ -n "$WAIT_FOR_RUN_NAME" ]; then
  wait_path="runs/${WAIT_FOR_RUN_NAME}/analysis/benchmark_summary.json"
  echo "[qwen3-baseline-suite] waiting for $wait_path"
  until [ -f "$wait_path" ]; do
    sleep 60
  done
fi

echo "[qwen3-baseline-suite] running think baseline"
run_eval "${RUN_PREFIX}_think_full" "think" "0.6" "0.95" "20"

echo "[qwen3-baseline-suite] running no_think baseline"
run_eval "${RUN_PREFIX}_nothink_full" "no_think" "0.7" "0.8" "20"

echo "[qwen3-baseline-suite] complete"
