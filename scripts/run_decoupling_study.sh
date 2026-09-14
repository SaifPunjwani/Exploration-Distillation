#!/usr/bin/env bash
set -euo pipefail

# TPU + W&B decoupling study grid.
# Adjust model, dataset sizes, and sweep ranges for larger runs.

export WANDB_PROJECT=expdis
export WANDB_ENTITY="${WANDB_ENTITY:-}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

"$PYTHON_BIN" run_expdis_torch.py --device tpu --use-wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --decoupling-sweep "coupled,decoupled" \
  --loss-type-sweep "dr_grpo,grpo,dapo" \
  --lambda-sweep "0.0,0.05,0.1,0.3" \
  --novelty-metric-sweep "mse,sqrt_mse,l2" \
  --novelty-clip-sweep "none,5.0,10.0" \
  --temperature-schedule "1.2,1.0,0.8" \
  --traj-novelty-train \
  --grpo-num-generations 2 \
  --grpo-generation-batch-size 4 \
  --grpo-grad-accum 2 \
  --max-prompt-len 64 \
  --max-completion-len 32 \
  --max-total-len 96
