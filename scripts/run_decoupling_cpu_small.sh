#!/usr/bin/env bash
set -euo pipefail

# CPU decoupling study (small scale) for quick end-to-end verification.

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_GROUP="${WANDB_GROUP:-decoupling_cpu_small_$(date +%Y%m%d_%H%M%S)}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-decoupling_cpu_small}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
MAX_TRAIN="${MAX_TRAIN:-200}"
MAX_EVAL="${MAX_EVAL:-50}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-30}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMP_SCHEDULE="${TEMP_SCHEDULE:-1.0}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-96}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-288}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

EXPDIS_WANDB_MINIMAL="${EXPDIS_WANDB_MINIMAL:-1}" \
"$PYTHON_BIN" run_expdis_torch.py --device cpu --use-wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --model-name "$MODEL_NAME" \
  --decoupling-sweep "coupled,decoupled" \
  --lambda-sweep "0.0,0.1" \
  --loss-type-sweep "dr_grpo" \
  --novelty-metric-sweep "mse" \
  --temperature-schedule "$TEMP_SCHEDULE" \
  --num-samples-per-prompt "$NUM_SAMPLES" \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN"
