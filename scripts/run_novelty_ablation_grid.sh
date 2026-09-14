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
    PYTHON_BIN="python"
  fi
fi

DEVICE="${DEVICE:-tpu}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
SEED="${SEED:-13}"
RUN_TAG="${RUN_TAG:-novelty_grid}"
MAX_TRAIN="${MAX_TRAIN:-2048}"
MAX_EVAL="${MAX_EVAL:-512}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-80}"
REWARD_CORRECT="${REWARD_CORRECT:-1.0}"
REWARD_INCORRECT="${REWARD_INCORRECT:-0.0}"

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"

"$PYTHON_BIN" main.py \
  --device "$DEVICE" \
  --use-wandb \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --decoupling-mode decoupled \
  --actual-train-mode sft_grpo \
  --base-eval-mode grpo \
  --novelty-feature-source-sweep "last,multilayer" \
  --reward-shape linear \
  --novelty-metric-sweep "mse,sqrt_mse,l2" \
  --lambda-sweep "0.0,0.05,0.1,0.2" \
  --novelty-clip-sweep "none,5.0" \
  --novelty-layers "4,8,12,16" \
  --novelty-layer-pool mean \
  --novelty-layer-agg mean \
  --novelty-zscore batch \
  --correctness-zscore off \
  --max-train-examples "$MAX_TRAIN" \
  --max-eval-examples "$MAX_EVAL" \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-num-generations 2 \
  --grpo-generation-batch-size 4 \
  --num-samples-per-prompt 2 \
  --reward-correct "$REWARD_CORRECT" \
  --reward-incorrect "$REWARD_INCORRECT" \
  --seed "$SEED" \
  --wandb-run-name "$RUN_TAG" \
  --plots-dir "runs/${RUN_TAG}/plots" \
  --explorer-traj-path "runs/${RUN_TAG}/explorer_trajectories.jsonl" \
  --novelty-study-tag "$RUN_TAG"

echo "Ablation grid complete. Run aggregation with:"
echo "  python scripts/plot_novelty_study.py --root runs --tag ${RUN_TAG}"
