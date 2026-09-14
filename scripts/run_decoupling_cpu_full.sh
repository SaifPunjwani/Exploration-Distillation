#!/usr/bin/env bash
set -euo pipefail

# CPU decoupling study (full run). Override via env vars if needed.
#
# Examples:
#   WANDB_GROUP=cpu_full_$(date +%Y%m%d_%H%M%S) bash scripts/run_decoupling_cpu_full.sh
#   MAX_TRAIN=1000 MAX_EVAL=256 GRPO_MAX_STEPS=100 bash scripts/run_decoupling_cpu_full.sh

export WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_GROUP="${WANDB_GROUP:-cpu_full_$(date +%Y%m%d_%H%M%S)}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-decoupling_cpu_full}"
export EXPDIS_WANDB_MINIMAL="${EXPDIS_WANDB_MINIMAL:-1}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
MAX_TRAIN="${MAX_TRAIN:-7473}"
MAX_EVAL="${MAX_EVAL:-1319}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-200}"
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

"$PYTHON_BIN" run_expdis_torch.py --device cpu --use-wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --decoupling-sweep "coupled,decoupled" \
  --lambda-sweep "0.0,0.05,0.1,0.3" \
  --loss-type-sweep "dr_grpo" \
  --novelty-metric-sweep "mse" \
  --temperature-schedule "1.2,1.0,0.8" \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --max-total-len "$MAX_TOTAL_LEN"
