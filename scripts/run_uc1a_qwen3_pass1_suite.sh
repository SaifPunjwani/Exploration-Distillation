#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_PREFIX="${RUN_PREFIX:-uc1a_qwen3_pass1_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-tpu}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
BENCHMARK_DATASET_NAME="${BENCHMARK_DATASET_NAME:-aime_2024}"
MAX_BENCHMARK="${MAX_BENCHMARK:-30}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-512}"
BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="${BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN:-8192}"
BENCHMARK_EVAL_BATCH_SIZE="${BENCHMARK_EVAL_BATCH_SIZE:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

run_one() {
  local name="$1"
  local thinking_mode="$2"
  local temp="$3"
  local top_p="$4"

  PYTHON_BIN=".venv_tpu_slice/bin/python" \
  DEVICE="$DEVICE" \
  MODEL_NAME="$MODEL_NAME" \
  RUN_NAME="$name" \
  BENCHMARK_DATASET_NAME="$BENCHMARK_DATASET_NAME" \
  MAX_BENCHMARK="$MAX_BENCHMARK" \
  BENCHMARK_NUM_ROLLOUTS="1" \
  BENCHMARK_ROLLOUT_TEMPERATURE="$temp" \
  BENCHMARK_ROLLOUT_TOP_P="$top_p" \
  BENCHMARK_ROLLOUT_PASS_K="1" \
  BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
  BENCHMARK_EVAL_BATCH_SIZE="$BENCHMARK_EVAL_BATCH_SIZE" \
  MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
  MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
  USE_WANDB="$USE_WANDB" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  TPU_SKIP_MDS_QUERY=1 \
  TMX_ALLOW_SINGLE_WORKER_TPU=1 \
  PJRT_DEVICE=TPU \
  TPU_VISIBLE_CHIPS=0 \
  TPU_PROCESS_BOUNDS=1,1,1 \
  TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
  TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 \
  TPU_HOST_BOUNDS=1,1,1 \
  TPU_WORKER_ID=0 \
  TPU_WORKER_HOSTNAMES=localhost \
  .venv_tpu_slice/bin/python main.py \
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
    --eval-thinking-mode "$thinking_mode" \
    --benchmark-dataset-name "$BENCHMARK_DATASET_NAME" \
    --benchmark-dataset-split-policy first_available \
    --benchmark-eval-mode sampled \
    --benchmark-num-rollouts 1 \
    --benchmark-rollout-temperature "$temp" \
    --benchmark-rollout-top-p "$top_p" \
    --benchmark-rollout-pass-k 1 \
    --benchmark-rollout-max-completion-len "$BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN" \
    --benchmark-eval-batch-size "$BENCHMARK_EVAL_BATCH_SIZE" \
    --benchmark-rollout-output-path "runs/${name}/analysis/benchmark_rollouts_base.json" \
    --plots-dir "runs/${name}/plots" \
    --no-save-actual-model \
    --use-wandb \
    --wandb-run-name "$name" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY"
}

run_one "${RUN_PREFIX}_think_pass1" "think" "0.6" "0.95"
run_one "${RUN_PREFIX}_nothink_pass1" "no_think" "0.7" "0.8"
