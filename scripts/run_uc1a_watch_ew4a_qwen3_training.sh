#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_TPU_NAME="${TARGET_TPU_NAME:?set TARGET_TPU_NAME to the target TPU VM (slice) name}"
TARGET_ZONE="${TARGET_ZONE:-europe-west4-a}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WORKERS="${WORKERS:-0}"
SYNC_WORKERS="${SYNC_WORKERS:-$WORKERS}"
BOOTSTRAP_WORKERS="${BOOTSTRAP_WORKERS:-$WORKERS}"
CHIPS="${CHIPS:-}"
LOCAL_CHIPS="${LOCAL_CHIPS:-1}"

QUEUE_TAG="${QUEUE_TAG:-ew4a_qwen3_1p7b_sftfulltrace}"
SESSION_PREFIX="${SESSION_PREFIX:-ew4a_q3sft}"
RUN_ROOT="${RUN_ROOT:-runs/${QUEUE_TAG}}"
RUN_ARTIFACT_GCS_PREFIX="${RUN_ARTIFACT_GCS_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/ew4a_qwen3_1p7b_sftfulltrace}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:?set SAVED_MODEL_ROOT to the saved-model directory on the TPU VM}"

log() {
  printf '[ew4a-watch] %s\n' "$*"
}

queued_state() {
  gcloud alpha compute tpus queued-resources describe "$TARGET_TPU_NAME" \
    --zone="$TARGET_ZONE" \
    --format='value(state)' 2>/dev/null | sed 's/^state=//' || true
}

node_state() {
  gcloud alpha compute tpus tpu-vm describe "$TARGET_TPU_NAME" \
    --zone="$TARGET_ZONE" \
    --format='value(state,health)' 2>/dev/null || true
}

target_ready() {
  local qr_state node
  qr_state="$(queued_state)"
  node="$(node_state)"
  [ "$qr_state" = "READY" ] || [ "$qr_state" = "ACTIVE" ] || return 1
  printf '%s' "$node" | grep -q 'READY' || return 1
  printf '%s' "$node" | grep -q 'HEALTHY' || return 1
  gcloud alpha compute tpus tpu-vm ssh "$TARGET_TPU_NAME" \
    --worker=0 \
    --zone="$TARGET_ZONE" \
    --command='echo ready' >/dev/null 2>&1
}

remote_training_active() {
  local worker
  for worker in $WORKERS; do
    if gcloud alpha compute tpus tpu-vm ssh "$TARGET_TPU_NAME" \
      --worker="$worker" \
      --zone="$TARGET_ZONE" \
      --command="if pgrep -af 'python.*main.py.*${QUEUE_TAG}' | grep -v 'pgrep -af' >/dev/null; then exit 0; fi; found=1; for pane_pid in \$(tmux list-panes -a -F '#{session_name} #{pane_pid}' 2>/dev/null | awk '\$1 ~ /^${SESSION_PREFIX}_/ {print \$2}'); do if pgrep -P \"\$pane_pid\" >/dev/null 2>&1; then found=0; break; fi; done; exit \$found" \
      >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

saved_model_present() {
  if [ -d "$SAVED_MODEL_ROOT" ] && find "$SAVED_MODEL_ROOT" -name artifact_metadata.json -print -quit | grep -q .; then
    return 0
  fi
  gcloud storage ls "${RUN_ARTIFACT_GCS_PREFIX}/**/artifact_metadata.json" >/dev/null 2>&1
}

launch_once() {
  log "target is healthy; launching Qwen3 two-model training"
  TPU_NAME="$TARGET_TPU_NAME" \
  ZONE="$TARGET_ZONE" \
  QUEUE_TAG="$QUEUE_TAG" \
  SESSION_PREFIX="$SESSION_PREFIX" \
  RUN_ROOT="$RUN_ROOT" \
  RUN_ARTIFACT_GCS_PREFIX="$RUN_ARTIFACT_GCS_PREFIX" \
  SAVED_MODEL_ROOT="$SAVED_MODEL_ROOT" \
  WORKERS="$WORKERS" \
  SYNC_WORKERS="$SYNC_WORKERS" \
  BOOTSTRAP_WORKERS="$BOOTSTRAP_WORKERS" \
  CHIPS="$CHIPS" \
  LOCAL_CHIPS="$LOCAL_CHIPS" \
  ALLOW_DIRTY=1 \
  MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}" \
  DATASET_NAME="${DATASET_NAME:-dapo_math_17k}" \
  RUN_PROFILE="${RUN_PROFILE:-stable_trace}" \
  RECIPE_MODE="${RECIPE_MODE:-stable_trace_baseinit}" \
  RUN_OFFSET="${RUN_OFFSET:-0}" \
  RUN_LIMIT="${RUN_LIMIT:-1}" \
  MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-512}" \
  MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-256}" \
  TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-512}" \
  TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-2}" \
  TRAIN_PROBE_BATCH_SIZE="${TRAIN_PROBE_BATCH_SIZE:-16}" \
  GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-48}" \
  GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-4}" \
  GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-4}" \
  NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}" \
  MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}" \
  MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-192}" \
  EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-96}" \
  EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-128}" \
  MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-448}" \
  DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}" \
  MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-512}" \
  ACTUAL_GRPO_EPOCHS="${ACTUAL_GRPO_EPOCHS:-2}" \
  ACTUAL_GRPO_BATCH_SIZE="${ACTUAL_GRPO_BATCH_SIZE:-1}" \
  ACTUAL_MIN_ACCEPTED_TRAJECTORIES="${ACTUAL_MIN_ACCEPTED_TRAJECTORIES:-8}" \
  MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-512}" \
  TMX_GRPO_GRAD_CKPT="${TMX_GRPO_GRAD_CKPT:-1}" \
  TMX_XLA_SAFE_GENERATE="${TMX_XLA_SAFE_GENERATE:-1}" \
  TMX_DISABLE_BF16="${TMX_DISABLE_BF16:-1}" \
  TMX_XLA_SKIP_LOADER_MARK_STEP="${TMX_XLA_SKIP_LOADER_MARK_STEP:-1}" \
  TMX_XLA_INCREMENTAL_GENERATE="${TMX_XLA_INCREMENTAL_GENERATE:-0}" \
  TMX_RESUME_TRAINING="${TMX_RESUME_TRAINING:-1}" \
  TMX_SKIP_INTERMEDIATE_EVALS="${TMX_SKIP_INTERMEDIATE_EVALS:-0}" \
  TMX_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE="${TMX_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE:-1}" \
  TMX_GRPO_SAVE_STRATEGY="${TMX_GRPO_SAVE_STRATEGY:-steps}" \
  TMX_GRPO_SAVE_STEPS="${TMX_GRPO_SAVE_STEPS:-4}" \
  TMX_GRPO_SAVE_TOTAL_LIMIT="${TMX_GRPO_SAVE_TOTAL_LIMIT:-3}" \
  TMX_GRPO_LOGGING_STRATEGY="${TMX_GRPO_LOGGING_STRATEGY:-steps}" \
  TMX_GRPO_LOGGING_STEPS="${TMX_GRPO_LOGGING_STEPS:-1}" \
  TMX_USE_XLA_WORLD_SIZE="${TMX_USE_XLA_WORLD_SIZE:-}" \
  ANSWER_CONSTRAINT_MODE="${ANSWER_CONSTRAINT_MODE:-}" \
  EXPLORATION_RESPONSE_STYLE="${EXPLORATION_RESPONSE_STYLE:-}" \
  EVAL_RESPONSE_STYLE="${EVAL_RESPONSE_STYLE:-}" \
  EXPLORATION_ANSWER_CONSTRAINT_MODE="${EXPLORATION_ANSWER_CONSTRAINT_MODE:-}" \
  EVAL_ANSWER_CONSTRAINT_MODE="${EVAL_ANSWER_CONSTRAINT_MODE:-}" \
  ACTUAL_TARGET_CONTRACT="${ACTUAL_TARGET_CONTRACT:-}" \
  BASE_EVAL_MODE="${BASE_EVAL_MODE:-}" \
  bash scripts/launch_uc1a_qwen3_two_model_training.sh
}

log "watching $TARGET_TPU_NAME in $TARGET_ZONE until a saved model lands"
while true; do
  if saved_model_present; then
    log "saved model detected; watcher exiting"
    exit 0
  fi
  if ! target_ready; then
    log "target not ready yet: queued_state=$(queued_state) node=$(node_state)"
    sleep "$POLL_SECONDS"
    continue
  fi
  if remote_training_active; then
    log "remote training still active on $TARGET_TPU_NAME"
    sleep "$POLL_SECONDS"
    continue
  fi
  launch_once || log "launch failed; will retry"
  sleep "$POLL_SECONDS"
done
