#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-us-central1-a}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
WORKER="${WORKER:-1}"
RUN_TAG="${RUN_TAG:-uc1a_canary_20260331w1j}"
RUN_NAME="${RUN_NAME:-${RUN_TAG}_two_trace_ws_distill_dl5e6}"
RUN_LOG="${RUN_LOG:-runs/${RUN_TAG}/${RUN_NAME}/console.log}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WATCH_LOG="${WATCH_LOG:-runs/${RUN_TAG}_watch.log}"

mkdir -p "$(dirname "$WATCH_LOG")"

remote() {
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone "$ZONE" \
    --worker "$WORKER" \
    --tunnel-through-iap \
    --command "$1"
}

launch_run() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launch ${RUN_TAG}" | tee -a "$WATCH_LOG"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$WORKER" \
  REMOTE_WORKERS="$WORKER" \
  CHIPS=0 \
  LOCAL_CHIPS=1 \
  SKIP_BOOTSTRAP=1 \
  ALLOW_DIRTY=1 \
  TPU_ACCELERATOR_TYPE=v5litepod-64 \
  HF_HOME="$REMOTE_ROOT/.hf" \
  HF_HUB_CACHE="$REMOTE_ROOT/.hf/hub" \
  HF_DATASETS_CACHE="$REMOTE_ROOT/.hf/datasets" \
  TRANSFORMERS_CACHE="$REMOTE_ROOT/.hf/hub" \
  HF_HUB_OFFLINE=0 \
  HF_DATASETS_OFFLINE=0 \
  TRANSFORMERS_OFFLINE=0 \
  TMX_GRPO_GRAD_CKPT=0 \
  TMX_XLA_SAFE_GENERATE=1 \
  RUN_PROFILE=smoke \
  RECIPE_MODE=stable_trace_warmstart \
  TRAIN_SELECTION_POLICY=random \
  TRAIN_SELECTION_MANIFEST_PATH="runs/uc1a_canary_20260331w1j/train_selection_seed0.json" \
  ANSWER_CONSTRAINT_MODE=none \
  EXPLORATION_RESPONSE_STYLE=reasoning_final \
  EVAL_RESPONSE_STYLE=reasoning_final \
  EXPLORATION_ANSWER_CONSTRAINT_MODE=none \
  EVAL_ANSWER_CONSTRAINT_MODE=none \
  ACTUAL_TARGET_CONTRACT=full_trace \
  BASE_EVAL_MODE=grpo \
  RUN_LIMIT=1 \
  RETRY_ON_FAILURE=0 \
  QUEUE_TAG="$RUN_TAG" \
  SESSION_PREFIX=uc1aw1j \
  MODEL_NAME='Qwen/Qwen2.5-Math-1.5B-Instruct' \
  DATASET_NAME='gsm8k' \
  MAX_TRAIN_EXAMPLES=64 \
  MAX_EVAL_EXAMPLES=64 \
  TRAIN_PROBE_POOL_SIZE=64 \
  TRAIN_PROBE_SAMPLES_PER_PROMPT=2 \
  TRAIN_PROBE_BATCH_SIZE=8 \
  GRPO_MAX_STEPS=4 \
  GRPO_NUM_GENERATIONS=2 \
  GRPO_GENERATION_BATCH_SIZE=1 \
  NUM_SAMPLES_PER_PROMPT=2 \
  MAX_PROMPT_LEN=96 \
  MAX_COMPLETION_LEN=24 \
  EXPLORATION_MAX_COMPLETION_LEN=12 \
  EVAL_MAX_COMPLETION_LEN=16 \
  MAX_TOTAL_LEN=144 \
  NOVELTY_LAYERS=21 \
  DISTILL_EPOCHS=1 \
  MAX_DISTILL_EXAMPLES=64 \
  MAX_ACTUAL_EXAMPLES=64 \
  bash scripts/queue_gsm8k_recipe_search_tmux.sh | tee -a "$WATCH_LOG"
}

while true; do
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  active="$(
    remote "ps -eo cmd | grep -F '$RUN_NAME' | grep -v grep || true" 2>/dev/null || true
  )"
  status_tail="$(
    remote "tail -n 8 '$RUN_LOG' 2>/dev/null || true" 2>/dev/null || true
  )"

  if printf '%s\n' "$status_tail" | grep -q '\[run\] success'; then
    echo "[$timestamp] success ${RUN_NAME}" | tee -a "$WATCH_LOG"
    exit 0
  fi

  if [ -n "$active" ]; then
    echo "[$timestamp] active ${RUN_NAME}" | tee -a "$WATCH_LOG"
    sleep "$POLL_SECONDS"
    continue
  fi

  echo "[$timestamp] inactive ${RUN_NAME}; relaunching" | tee -a "$WATCH_LOG"
  launch_run
  sleep 20
done
