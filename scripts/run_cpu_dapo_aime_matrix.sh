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

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PREFIX="${PREFIX:-cpu_dapo_aime_matrix_${STAMP}}"

MAX_TRAIN="${MAX_TRAIN:-32}"
MAX_EVAL="${MAX_EVAL:-16}"
MAX_BENCHMARK="${MAX_BENCHMARK:-16}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-384}"

GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-4}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-1}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}"
ACTUAL_MIN_ACCEPTED_TRAJECTORIES="${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-8}"
ACTUAL_DATASET_POLICY="${ACTUAL_DATASET_POLICY:-accepted_balanced}"
ACTUAL_MAX_INCORRECT_RATIO="${ACTUAL_MAX_INCORRECT_RATIO:-1.0}"
USE_WANDB="${USE_WANDB:-auto}"

run_case() {
  local run_name="$1"
  shift
  echo
  echo "================================================================"
  echo "Launching ${run_name}"
  echo "================================================================"
  env \
    RUN_NAME="$run_name" \
    MAX_TRAIN="$MAX_TRAIN" \
    MAX_EVAL="$MAX_EVAL" \
    MAX_BENCHMARK="$MAX_BENCHMARK" \
    MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
    MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
    MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
    GRPO_MAX_STEPS="$GRPO_MAX_STEPS" \
    GRPO_BATCH_SIZE="$GRPO_BATCH_SIZE" \
    GRPO_GEN_BATCH="$GRPO_GEN_BATCH" \
    GRPO_NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
    GRPO_GRAD_ACCUM="$GRPO_GRAD_ACCUM" \
    NUM_SAMPLES_PER_PROMPT="$NUM_SAMPLES_PER_PROMPT" \
    DISTILL_EPOCHS="$DISTILL_EPOCHS" \
    ACTUAL_GRPO_EPOCHS="$ACTUAL_GRPO_EPOCHS" \
    ACTUAL_MIN_ACCEPTED_TRAJECTORIES="$ACTUAL_MIN_ACCEPTED_TRAJECTORIES" \
    ACTUAL_DATASET_POLICY="$ACTUAL_DATASET_POLICY" \
    ACTUAL_MAX_INCORRECT_RATIO="$ACTUAL_MAX_INCORRECT_RATIO" \
    USE_WANDB="$USE_WANDB" \
    "$@" \
    bash scripts/run_cpu_dapo_aime_benchmark.sh
}

run_case "${PREFIX}_base_grpo" \
  PIPELINE_MODE=base_only \
  BASE_EVAL_MODE=grpo \
  LAMBDA_NOVELTY=0.0

run_case "${PREFIX}_two_canonical_l010" \
  PIPELINE_MODE=two_model \
  ACTUAL_TARGET_CONTRACT=canonical \
  LAMBDA_NOVELTY=0.10 \
  NOVELTY_METRIC=sqrt_mse

run_case "${PREFIX}_two_full_trace_l010" \
  PIPELINE_MODE=two_model \
  ACTUAL_TARGET_CONTRACT=full_trace \
  LAMBDA_NOVELTY=0.10 \
  NOVELTY_METRIC=sqrt_mse

run_case "${PREFIX}_two_canonical_l000" \
  PIPELINE_MODE=two_model \
  ACTUAL_TARGET_CONTRACT=canonical \
  LAMBDA_NOVELTY=0.0 \
  NOVELTY_METRIC=sqrt_mse
