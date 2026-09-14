#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$HOME/vllm_tpu_env/bin/python" ]; then
    PYTHON_BIN="$HOME/vllm_tpu_env/bin/python"
  elif [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

SERVER_BASE_URL="${SERVER_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
RUN_NAME="${RUN_NAME:-vllm_aime_rollout_eval}"
ROLE_NAME="${ROLE_NAME:-base}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
BENCHMARK_DATASET_SPLIT_POLICY="${BENCHMARK_DATASET_SPLIT_POLICY:-first_available}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_TOP_K="${BENCHMARK_ROLLOUT_TOP_K:-0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-32}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
EVAL_RESPONSE_STYLE="${EVAL_RESPONSE_STYLE:-reasoning_final}"
ANSWER_CONSTRAINT_MODE="${ANSWER_CONSTRAINT_MODE:-none}"
THINKING_MODE="${THINKING_MODE:-auto}"
PROBLEM_SHARD_INDEX="${PROBLEM_SHARD_INDEX:-0}"
PROBLEM_SHARD_COUNT="${PROBLEM_SHARD_COUNT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
USE_WANDB="${USE_WANDB:-0}"

ARGS=(
  --server-base-url "$SERVER_BASE_URL"
  --model-name "$MODEL_NAME"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer-name "$TOKENIZER_NAME"
  --run-name "$RUN_NAME"
  --role-name "$ROLE_NAME"
  --dataset-name "$BENCHMARK_DATASET_NAME"
  --dataset-split-policy "$BENCHMARK_DATASET_SPLIT_POLICY"
  --max-benchmark-examples "$MAX_BENCHMARK"
  --num-rollouts "$BENCHMARK_NUM_ROLLOUTS"
  --rollout-temperature "$BENCHMARK_ROLLOUT_TEMPERATURE"
  --rollout-top-p "$BENCHMARK_ROLLOUT_TOP_P"
  --rollout-top-k "$BENCHMARK_ROLLOUT_TOP_K"
  --rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN"
  --pass-k "$BENCHMARK_ROLLOUT_PASS_K"
  --request-concurrency "$REQUEST_CONCURRENCY"
  --request-timeout-seconds "$REQUEST_TIMEOUT_SECONDS"
  --max-prompt-len "$MAX_PROMPT_LEN"
  --eval-response-style "$EVAL_RESPONSE_STYLE"
  --answer-constraint-mode "$ANSWER_CONSTRAINT_MODE"
  --thinking-mode "$THINKING_MODE"
  --problem-shard-index "$PROBLEM_SHARD_INDEX"
  --problem-shard-count "$PROBLEM_SHARD_COUNT"
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
)

if [ -n "$BENCHMARK_DATASET_CONFIG_NAME" ]; then
  ARGS+=(--dataset-config-name "$BENCHMARK_DATASET_CONFIG_NAME")
fi

if [ -n "$WANDB_GROUP" ]; then
  ARGS+=(--wandb-group "$WANDB_GROUP")
fi

case "$USE_WANDB" in
  1|true|TRUE|yes|YES)
    ARGS+=(--use-wandb)
    ;;
esac

"$PYTHON_BIN" scripts/run_vllm_aime_benchmark.py "${ARGS[@]}"
