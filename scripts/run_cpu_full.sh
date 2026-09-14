#!/usr/bin/env bash
set -euo pipefail

# CPU full-scale run (single config).
# Edit defaults here or override via env vars.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
MAX_TRAIN="${MAX_TRAIN:-7473}"   # GSM8K train size
MAX_EVAL="${MAX_EVAL:-1319}"     # GSM8K test size

GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-200}"
GRPO_LR="${GRPO_LR:-5e-6}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-4}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-2}"

LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.1}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
NOVELTY_METRIC="${NOVELTY_METRIC:-mse}"
NOVELTY_CLIP="${NOVELTY_CLIP:-}"

DECOUPLING_MODE="${DECOUPLING_MODE:-decoupled}"
TEMP_SCHEDULE="${TEMP_SCHEDULE:-1.2,1.0,0.8}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-cpu_full_run}"
WANDB_PROJECT="${WANDB_PROJECT:-expdis}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
USE_WANDB="${USE_WANDB:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

WANDB_ARGS=()
if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS+=(--use-wandb --wandb-run-name "$WANDB_RUN_NAME" --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY")
else
  WANDB_ARGS+=(--no-wandb)
fi

NOVELTY_CLIP_ARGS=()
if [ -n "$NOVELTY_CLIP" ]; then
  NOVELTY_CLIP_ARGS+=(--novelty-clip "$NOVELTY_CLIP")
fi

"$PYTHON_BIN" run_expdis_torch.py --device cpu \
  --model-name "$MODEL_NAME" \
  --decoupling-mode "$DECOUPLING_MODE" \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-lr "$GRPO_LR" \
  --grpo-batch-size "$GRPO_BATCH_SIZE" \
  --grpo-generation-batch-size "$GRPO_GEN_BATCH" \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-grad-accum "$GRPO_GRAD_ACCUM" \
  --lambda-novelty "$LAMBDA_NOVELTY" \
  --grpo-loss-type "$GRPO_LOSS_TYPE" \
  --novelty-metric "$NOVELTY_METRIC" \
  --temperature-schedule "$TEMP_SCHEDULE" \
  "${NOVELTY_CLIP_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
