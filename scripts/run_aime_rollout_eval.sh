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

DEVICE="${DEVICE:-cpu}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
RUN_NAME="${RUN_NAME:-aime_rollout_eval}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
BENCHMARK_DATASET_SPLIT_POLICY="${BENCHMARK_DATASET_SPLIT_POLICY:-first_available}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-32}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-16384}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-32}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
PLOTS_DIR="${PLOTS_DIR:-runs/${RUN_NAME}/plots}"
BENCHMARK_ROLLOUT_OUTPUT_PATH="${BENCHMARK_ROLLOUT_OUTPUT_PATH:-runs/${RUN_NAME}/analysis/aime_rollouts.json}"
WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-auto}"

if [ -z "$BENCHMARK_DATASET_CONFIG_NAME" ] && [ "$BENCHMARK_DATASET_NAME" = "aime_2025" ]; then
  BENCHMARK_DATASET_CONFIG_NAME="all"
fi

WANT_WANDB=0
case "$USE_WANDB" in
  1|true|TRUE|yes|YES)
    WANT_WANDB=1
    ;;
  0|false|FALSE|no|NO)
    WANT_WANDB=0
    ;;
  auto|AUTO)
    if [ -n "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE:-}" = "offline" ]; then
      WANT_WANDB=1
    fi
    ;;
  *)
    echo "Unsupported USE_WANDB value: $USE_WANDB" >&2
    exit 2
    ;;
esac

WANDB_ARGS=(--no-wandb)
if [ "$WANT_WANDB" = "1" ]; then
  WANDB_ARGS=(--use-wandb --wandb-run-name "$RUN_NAME")
  if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
  fi
  if [ -n "${WANDB_ENTITY:-}" ]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

BENCHMARK_ARGS=(
  --benchmark-dataset-name "$BENCHMARK_DATASET_NAME"
  --benchmark-dataset-split-policy "$BENCHMARK_DATASET_SPLIT_POLICY"
  --benchmark-eval-mode sampled
  --benchmark-num-rollouts "$BENCHMARK_NUM_ROLLOUTS"
  --benchmark-rollout-temperature "$BENCHMARK_ROLLOUT_TEMPERATURE"
  --benchmark-rollout-top-p "$BENCHMARK_ROLLOUT_TOP_P"
  --benchmark-rollout-pass-k "$BENCHMARK_ROLLOUT_PASS_K"
  --benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN"
  --benchmark-eval-batch-size "$BENCHMARK_EVAL_BATCH_SIZE"
  --benchmark-rollout-output-path "$BENCHMARK_ROLLOUT_OUTPUT_PATH"
)
if [ -n "$BENCHMARK_DATASET_CONFIG_NAME" ]; then
  BENCHMARK_ARGS+=(--benchmark-dataset-config-name "$BENCHMARK_DATASET_CONFIG_NAME")
fi

"$PYTHON_BIN" main.py \
  --device "$DEVICE" \
  --model-name "$MODEL_NAME" \
  --dataset-name gsm8k \
  --dataset-config-name main \
  --dataset-split-policy auto \
  --pipeline-mode base_only \
  --base-eval-mode raw \
  --max-train-examples 1 \
  --max-eval-examples 1 \
  --max-benchmark-examples "$MAX_BENCHMARK" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --eval-max-completion-len "$MAX_COMPLETION_LEN" \
  --eval-batch-size 1 \
  --trajectory-batch-size 1 \
  --answer-constraint-mode none \
  --eval-response-style reasoning_final \
  --eval-answer-constraint-mode none \
  --plots-dir "$PLOTS_DIR" \
  --no-save-actual-model \
  "${BENCHMARK_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
