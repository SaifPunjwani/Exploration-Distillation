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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-all}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
BENCHMARK_DATASET_CONFIG_NAME="${BENCHMARK_DATASET_CONFIG_NAME:-}"
BENCHMARK_DATASET_SPLIT_EVAL="${BENCHMARK_DATASET_SPLIT_EVAL:-}"
BENCHMARK_DATASET_SPLIT_POLICY="${BENCHMARK_DATASET_SPLIT_POLICY:-first_available}"
BENCHMARK_EVAL_MODE="${BENCHMARK_EVAL_MODE:-auto}"
BENCHMARK_NUM_ROLLOUTS="${BENCHMARK_NUM_ROLLOUTS:-1}"
BENCHMARK_ROLLOUT_TEMPERATURE="${BENCHMARK_ROLLOUT_TEMPERATURE:-1.0}"
BENCHMARK_ROLLOUT_TOP_P="${BENCHMARK_ROLLOUT_TOP_P:-1.0}"
BENCHMARK_ROLLOUT_PASS_K="${BENCHMARK_ROLLOUT_PASS_K:-1,4,8,16,32}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-}"
BENCHMARK_ROLLOUT_OUTPUT_PATH="${BENCHMARK_ROLLOUT_OUTPUT_PATH:-}"
PIPELINE_MODE="${PIPELINE_MODE:-two_model}"
DECOUPLING_MODE="${DECOUPLING_MODE:-decoupled}"
BASE_EVAL_MODE="${BASE_EVAL_MODE:-grpo}"
ACTUAL_TRAIN_MODE="${ACTUAL_TRAIN_MODE:-sft_grpo}"
ACTUAL_REWARD_SOURCE="${ACTUAL_REWARD_SOURCE:-correctness}"
TRAIN_SELECTION_POLICY="${TRAIN_SELECTION_POLICY:-random}"
TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-16}"
TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-2}"
TRAIN_PROBE_TEMPERATURE="${TRAIN_PROBE_TEMPERATURE:-0.8}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
EXPLORATION_RESPONSE_STYLE="${EXPLORATION_RESPONSE_STYLE:-answer_only}"
EVAL_RESPONSE_STYLE="${EVAL_RESPONSE_STYLE:-answer_only}"
EXPLORATION_ANSWER_CONSTRAINT_MODE="${EXPLORATION_ANSWER_CONSTRAINT_MODE:-auto}"
EVAL_ANSWER_CONSTRAINT_MODE="${EVAL_ANSWER_CONSTRAINT_MODE:-auto}"

MAX_TRAIN="${MAX_TRAIN:-128}"
MAX_EVAL="${MAX_EVAL:-64}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-32}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-32}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-384}"

GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-16}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-2}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"

DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
ACTUAL_MIN_ACCEPTED_TRAJECTORIES="${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-8}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_balanced}"
ACTUAL_MAX_INCORRECT_RATIO="${ACTUAL_MAX_INCORRECT_RATIO:-1.0}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.03}"
NOVELTY_METRIC="${NOVELTY_METRIC:-sqrt_mse}"
NOVELTY_TEXT_CONTRACT="${NOVELTY_TEXT_CONTRACT:-full_raw}"
ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-canonical}"
REWARD_SHAPE="${REWARD_SHAPE:-linear}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:--1.0}"

RUN_NAME="${RUN_NAME:-cpu_dapo_aime_benchmark}"
PLOTS_DIR="${PLOTS_DIR:-runs/${RUN_NAME}/plots}"
TRAJ_PATH="${TRAJ_PATH:-runs/${RUN_NAME}/explorer_trajectories.jsonl}"
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
  --benchmark-eval-mode "$BENCHMARK_EVAL_MODE"
  --benchmark-num-rollouts "$BENCHMARK_NUM_ROLLOUTS"
  --benchmark-rollout-temperature "$BENCHMARK_ROLLOUT_TEMPERATURE"
  --benchmark-rollout-top-p "$BENCHMARK_ROLLOUT_TOP_P"
  --benchmark-rollout-pass-k "$BENCHMARK_ROLLOUT_PASS_K"
)
if [ -n "$BENCHMARK_DATASET_CONFIG_NAME" ]; then
  BENCHMARK_ARGS+=(--benchmark-dataset-config-name "$BENCHMARK_DATASET_CONFIG_NAME")
fi
if [ -n "$BENCHMARK_DATASET_SPLIT_EVAL" ]; then
  BENCHMARK_ARGS+=(--benchmark-dataset-split-eval "$BENCHMARK_DATASET_SPLIT_EVAL")
fi
if [ -n "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" ]; then
  BENCHMARK_ARGS+=(--benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN")
fi
if [ -n "$BENCHMARK_EVAL_BATCH_SIZE" ]; then
  BENCHMARK_ARGS+=(--benchmark-eval-batch-size "$BENCHMARK_EVAL_BATCH_SIZE")
fi
if [ -n "$BENCHMARK_ROLLOUT_OUTPUT_PATH" ]; then
  BENCHMARK_ARGS+=(--benchmark-rollout-output-path "$BENCHMARK_ROLLOUT_OUTPUT_PATH")
fi

"$PYTHON_BIN" main.py \
  --device cpu \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --dataset-config-name "$DATASET_CONFIG_NAME" \
  --train-selection-policy "$TRAIN_SELECTION_POLICY" \
  --train-probe-pool-size "$TRAIN_PROBE_POOL_SIZE" \
  --train-probe-samples-per-prompt "$TRAIN_PROBE_SAMPLES_PER_PROMPT" \
  --train-probe-temperature "$TRAIN_PROBE_TEMPERATURE" \
  --num-samples-per-prompt "$NUM_SAMPLES_PER_PROMPT" \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --max-benchmark-examples "$MAX_BENCHMARK" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --exploration-max-completion-len "$EXPLORATION_MAX_COMPLETION_LEN" \
  --eval-max-completion-len "$EVAL_MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN" \
  --exploration-response-style "$EXPLORATION_RESPONSE_STYLE" \
  --eval-response-style "$EVAL_RESPONSE_STYLE" \
  --exploration-answer-constraint-mode "$EXPLORATION_ANSWER_CONSTRAINT_MODE" \
  --eval-answer-constraint-mode "$EVAL_ANSWER_CONSTRAINT_MODE" \
  --pipeline-mode "$PIPELINE_MODE" \
  --base-eval-mode "$BASE_EVAL_MODE" \
  --decoupling-mode "$DECOUPLING_MODE" \
  --actual-train-mode "$ACTUAL_TRAIN_MODE" \
  --actual-target-contract "$ACTUAL_TARGET_CONTRACT" \
  --actual-reward-source "$ACTUAL_REWARD_SOURCE" \
  --actual-min-accepted-trajectories "$ACTUAL_MIN_ACCEPTED_TRAJECTORIES" \
  --actual-dataset-policy "$ACTUAL_DATASET_POLICY" \
  --actual-max-incorrect-ratio "$ACTUAL_MAX_INCORRECT_RATIO" \
  --novelty-feature-source multilayer \
  --novelty-metric "$NOVELTY_METRIC" \
  --novelty-text-contract "$NOVELTY_TEXT_CONTRACT" \
  --reward-shape "$REWARD_SHAPE" \
  --reward-correct "$REWARD_CORRECT" \
  --reward-incorrect "$REWARD_INCORRECT" \
  --answer-constraint-mode auto \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-batch-size "$GRPO_BATCH_SIZE" \
  --grpo-generation-batch-size "$GRPO_GEN_BATCH" \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-grad-accum "$GRPO_GRAD_ACCUM" \
  --distill-epochs "$DISTILL_EPOCHS" \
  --actual-grpo-epochs "$ACTUAL_GRPO_EPOCHS" \
  --lambda-novelty "$LAMBDA_NOVELTY" \
  --plots-dir "$PLOTS_DIR" \
  --explorer-traj-path "$TRAJ_PATH" \
  "${BENCHMARK_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
