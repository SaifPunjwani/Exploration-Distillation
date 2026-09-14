#!/usr/bin/env bash
set -euo pipefail

# TPU smoke test runner.
# Edit defaults here or override via env vars.

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
MAX_TRAIN="${MAX_TRAIN:-8}"
MAX_EVAL="${MAX_EVAL:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-192}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-96}"

GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-2}"
GRPO_GEN_BATCH="${GRPO_GEN_BATCH:-2}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"

LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.1}"
GRPO_LOSS_TYPE="${GRPO_LOSS_TYPE:-dr_grpo}"
NOVELTY_METRIC="${NOVELTY_METRIC:-mse}"
NOVELTY_FEATURE_SOURCE="${NOVELTY_FEATURE_SOURCE:-multilayer}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-tpu_test}"
WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

TPU_STRICT="${TPU_STRICT:-1}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PJRT_DEVICE=TPU
export TOKENIZERS_PARALLELISM=false
export TMX_XLA_SAFE_GENERATE="${TMX_XLA_SAFE_GENERATE:-1}"
export TMX_XLA_MARK_STEP_EVERY="${TMX_XLA_MARK_STEP_EVERY:-1}"
export TMX_XLA_FIXED_PADDING="${TMX_XLA_FIXED_PADDING:-1}"

if [ -n "${TPU_VISIBLE_DEVICES:-}" ]; then
  export TPU_VISIBLE_DEVICES
fi

if [ "$TPU_STRICT" = "1" ]; then
  "$PYTHON_BIN" - <<'PY'
import os
paths = [f"/dev/accel{i}" for i in range(4)]
ok = False
for p in paths:
    try:
        fd = os.open(p, os.O_RDWR)
        os.close(fd)
        ok = True
        break
    except OSError:
        pass
if not ok:
    raise SystemExit("TPU device not openable. Free /dev/accel* or set TPU_STRICT=0.")
PY
fi

"$PYTHON_BIN" main.py --device tpu \
  --model-name "$MODEL_NAME" \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --max-prompt-len "$MAX_PROMPT_LEN" \
  --max-completion-len "$MAX_COMPLETION_LEN" \
  --grpo-batch-size "$GRPO_BATCH_SIZE" \
  --grpo-generation-batch-size "$GRPO_GEN_BATCH" \
  --grpo-num-generations "$GRPO_NUM_GENERATIONS" \
  --grpo-grad-accum "$GRPO_GRAD_ACCUM" \
  --lambda-novelty "$LAMBDA_NOVELTY" \
  --grpo-loss-type "$GRPO_LOSS_TYPE" \
  --novelty-metric "$NOVELTY_METRIC" \
  --novelty-feature-source "$NOVELTY_FEATURE_SOURCE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --use-wandb
