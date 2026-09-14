#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VLLM_VENV_DIR="${VLLM_VENV_DIR:-$HOME/vllm_tpu_env}"
SERVER_BASE_URL="${SERVER_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
RUN_PREFIX="${RUN_PREFIX:-use1d_qwen3_1p7b_fast_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_GROUP="${WANDB_GROUP:-use1d_qwen3_fast_suite}"
CALIBRATION_MAX_COMPLETION_LEN="${CALIBRATION_MAX_COMPLETION_LEN:-8192}"

run_eval() {
  local run_name="$1"
  local thinking_mode="$2"
  local temperature="$3"
  local top_p="$4"
  local top_k="$5"
  local num_rollouts="$6"
  local pass_k="$7"

  echo "[qwen3-fast-suite] starting ${run_name}"
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
  BENCHMARK_NUM_ROLLOUTS="$num_rollouts" \
  BENCHMARK_ROLLOUT_TEMPERATURE="$temperature" \
  BENCHMARK_ROLLOUT_TOP_P="$top_p" \
  BENCHMARK_ROLLOUT_TOP_K="$top_k" \
  BENCHMARK_ROLLOUT_PASS_K="$pass_k" \
  BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$CALIBRATION_MAX_COMPLETION_LEN" \
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

echo "[qwen3-fast-suite] running literature-style think pass@1"
run_eval "${RUN_PREFIX}_think_pass1" "think" "0.6" "0.95" "20" "1" "1"

echo "[qwen3-fast-suite] running literature-style no_think pass@1"
run_eval "${RUN_PREFIX}_nothink_pass1" "no_think" "0.7" "0.8" "20" "1" "1"

echo "[qwen3-fast-suite] running think avg@8 proxy"
run_eval "${RUN_PREFIX}_think_avg8" "think" "0.6" "0.95" "20" "8" "1,4,8"

echo "[qwen3-fast-suite] running no_think avg@8 proxy"
run_eval "${RUN_PREFIX}_nothink_avg8" "no_think" "0.7" "0.8" "20" "8" "1,4,8"

echo "[qwen3-fast-suite] complete"
