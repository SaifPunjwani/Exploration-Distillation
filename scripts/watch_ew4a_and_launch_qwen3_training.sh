#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-a}"
POLL_SECONDS="${POLL_SECONDS:-30}"

QUEUE_TAG="${QUEUE_TAG:-ew4a_qwen3_1p7b_two_model_resume}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/ew4a_qwen3_two_model_resume}"

log() {
  printf '[ew4a-qwen3-watch] %s\n' "$*"
}

queued_state() {
  gcloud alpha compute tpus queued-resources describe "$TPU_NAME" \
    --zone="$ZONE" \
    --format='value(state)' 2>/dev/null | sed 's/^state=//' || true
}

node_health() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" \
    --format='value(state,health)' 2>/dev/null || true
}

ssh_ready() {
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --worker=0 \
    --zone="$ZONE" \
    --command='echo ok' >/dev/null 2>&1
}

log "waiting for $TPU_NAME in $ZONE"
while true; do
  q_state="$(queued_state)"
  n_state="$(node_health)"
  if [ "$q_state" = "READY" ] && printf '%s' "$n_state" | grep -q 'READY' && printf '%s' "$n_state" | grep -q 'HEALTHY' && ssh_ready; then
    log "target is reachable: queued_state=$q_state node=$n_state"
    break
  fi
  log "not ready yet: queued_state=${q_state:-missing} node=${n_state:-missing}"
  sleep "$POLL_SECONDS"
done

log "launching Qwen3 two-model checkpoint run on $TPU_NAME"
TMX_DISABLE_BF16="${TMX_DISABLE_BF16:-1}" \
TMX_SKIP_INTERMEDIATE_EVALS="${TMX_SKIP_INTERMEDIATE_EVALS:-0}" \
RUN_PROFILE="${RUN_PROFILE:-smoke}" \
RECIPE_MODE="${RECIPE_MODE:-stable_trace_warmstart}" \
TPU_NAME="$TPU_NAME" \
ZONE="$ZONE" \
WORKERS="${WORKERS:-0}" \
SYNC_WORKERS="${SYNC_WORKERS:-0}" \
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-0}" \
CHIPS="${CHIPS:-0}" \
QUEUE_TAG="$QUEUE_TAG" \
SESSION_PREFIX="${SESSION_PREFIX:-ew4a_q3train}" \
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}" \
DATASET_NAME="${DATASET_NAME:-dapo_math_17k}" \
RUN_ARTIFACT_GCS_PREFIX="$RUN_ARTIFACT_GCS_PREFIX" \
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-16}" \
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-8}" \
TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-32}" \
TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-1}" \
TRAIN_PROBE_BATCH_SIZE="${TRAIN_PROBE_BATCH_SIZE:-2}" \
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-2}" \
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}" \
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-1}" \
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-2}" \
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-64}" \
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-32}" \
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-16}" \
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-24}" \
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-96}" \
DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}" \
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-32}" \
ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-1}" \
ACTUAL_GRPO_BATCH_SIZE="${ACTUAL_GRPO_BATCH_SIZE:-1}" \
MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-32}" \
bash scripts/launch_uc1a_qwen3_two_model_training.sh

log "launch command completed"
