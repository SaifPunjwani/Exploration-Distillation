#!/usr/bin/env bash
# Launch the Explorer GRPO training on TPU.
#
# Usage:
#   TMX_VLLM_SERVER_URLS=http://ip:8000/v1,http://ip2:8000/v1 \
#   RUN_NAME=explore_lambda0 \
#   bash tmx_jax/scripts/launch_jax.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_DIR="${ENV_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-tmx_jax}"
source "$ENV_DIR/bin/activate" "$ENV_NAME"

RUN_NAME="${RUN_NAME:-jax_lambda0_$(date +%Y%m%d_%H%M%S)}"
OUT="runs/$RUN_NAME"
mkdir -p "$OUT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-20000}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-500}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-50}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-16}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
TMX_GCS_CHECKPOINT_BASE="${TMX_GCS_CHECKPOINT_BASE:-}"
TMX_DEBUG_COMPARE_LOGPROB_PATHS="${TMX_DEBUG_COMPARE_LOGPROB_PATHS:-0}"

VLLM_URLS="${TMX_VLLM_SERVER_URLS:-${VLLM_SERVER_URLS:-}}"
if [ -z "$VLLM_URLS" ]; then
  echo "ERROR: set TMX_VLLM_SERVER_URLS=http://ip:8000/v1,..." >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-two-model-explore}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export TPU_VISIBLE_CHIPS="${TPU_VISIBLE_CHIPS:-0,1,2,3}"
export TPU_PROCESS_BOUNDS="${TPU_PROCESS_BOUNDS:-1,1,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
export TMX_GCS_CHECKPOINT_BASE
export TMX_DEBUG_COMPARE_LOGPROB_PATHS

python -m tmx_jax.train \
  --model-name "$MODEL_NAME" \
  --dataset-name dapo_math_17k \
  --max-train-examples "$MAX_TRAIN_EXAMPLES" \
  --max-prompt-len 2048 \
  --max-completion-len 16384 \
  --max-total-len 18432 \
  --grpo-batch-size 1 \
  --grpo-grad-accum 2 \
  --grpo-num-generations 8 \
  --grpo-max-steps "$GRPO_MAX_STEPS" \
  --grpo-loss-type dr_grpo \
  --optimizer adamw \
  --grpo-lr 5e-6 \
  --grpo-clip-epsilon 0.2 \
  --grpo-kl-beta 0.0 \
  --trajectory-batch-size 2 \
  --vllm-server-urls "$VLLM_URLS" \
  --vllm-concurrency "$VLLM_CONCURRENCY" \
  --vllm-fanout-per-prompt "$VLLM_FANOUT_PER_PROMPT" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --enable-thinking auto \
  --reward-correct 1.0 \
  --reward-incorrect 0.0 \
  --format-penalty-weight 0.1 \
  --lambda-novelty "${LAMBDA_NOVELTY:-0.0}" \
  --novelty-feature-max-length 10240 \
  --distill-epochs 2 \
  --distill-batch-size 1 \
  --max-distill-examples 20000 \
  --actual-online-grpo-max-steps 250 \
  --actual-grpo-lr 1e-6 \
  --actual-grpo-kl-coef 0.10 \
  --benchmark-num-rollouts 32 \
  --benchmark-rollout-max-completion-len 32768 \
  --rescore-chunk-size 1 \
  --logprob-chunk-tokens 64 \
  --output-dir "$OUT" \
  --wandb-run-name "$RUN_NAME" \
  --save-every-steps "$SAVE_EVERY_STEPS" \
  --seed 0 \
  2>&1 | tee -a "$OUT/console.log"
