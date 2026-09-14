#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-us-east1-d}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
CANARY_QUEUE_TAG="${CANARY_QUEUE_TAG:-use1d_remote_smoke_$(date +%Y%m%d_%H%M%S)}"
CANARY_SESSION_PREFIX="${CANARY_SESSION_PREFIX:-use1d_remote_smoke}"
CANARY_MODEL_NAME="${CANARY_MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
CANARY_DATASET_NAME="${CANARY_DATASET_NAME:-gsm8k}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-5400}"
CANARY_POLL_SECONDS="${CANARY_POLL_SECONDS:-20}"
POST_SMOKE_CAMPAIGN_TAG="${POST_SMOKE_CAMPAIGN_TAG:-use1d_post_smoke_$(date +%Y%m%d_%H%M%S)}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

slot_log_path() {
  printf '%s/runs/%s/_slot_logs/%s_w0_c0.log' \
    "$REMOTE_ROOT" "$CANARY_QUEUE_TAG" "$CANARY_SESSION_PREFIX"
}

launch_canary() {
  log "launching lean worker-0 canary queue tag=$CANARY_QUEUE_TAG"
  env \
    TMX_DISABLE_BF16=1 \
    TMX_XLA_SAFE_GENERATE=1 \
    TMX_DEBUG_GENERATE="${TMX_DEBUG_GENERATE:-1}" \
    TMX_MASK_TRUNCATED_COMPLETIONS=0 \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    WORKERS=0 \
    LOCAL_CHIPS=1 \
    CHIPS=0 \
    SKIP_BOOTSTRAP=1 \
    SESSION_PREFIX="$CANARY_SESSION_PREFIX" \
    QUEUE_TAG="$CANARY_QUEUE_TAG" \
    RUN_PROFILE=stable_trace \
    RECIPE_MODE=stable_trace_warmstart \
    TRAIN_SELECTION_POLICY=random \
    RUN_LIMIT=1 \
    SEED=0 \
    MODEL_NAME="$CANARY_MODEL_NAME" \
    DATASET_NAME="$CANARY_DATASET_NAME" \
    MAX_PROMPT_LEN=64 \
    MAX_COMPLETION_LEN=32 \
    EXPLORATION_MAX_COMPLETION_LEN=8 \
    EVAL_MAX_COMPLETION_LEN=16 \
    MAX_TOTAL_LEN=96 \
    GRPO_NUM_GENERATIONS=2 \
    GRPO_GENERATION_BATCH_SIZE=2 \
    NUM_SAMPLES_PER_PROMPT=2 \
    MAX_TRAIN_EXAMPLES=64 \
    MAX_EVAL_EXAMPLES=64 \
    GRPO_MAX_STEPS=4 \
    MAX_DISTILL_EXAMPLES=64 \
    MAX_ACTUAL_EXAMPLES=96 \
    RND_HIDDEN=64 \
    NOVELTY_LAYERS=21 \
    NOVELTY_FEATURE_MAX_LENGTH=32 \
    bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

wait_for_canary_success() {
  local slot_log
  local deadline
  slot_log="$(slot_log_path)"
  deadline=$((SECONDS + CANARY_TIMEOUT_SECONDS))

  log "waiting for canary success at $slot_log"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ -f "$slot_log" ]; then
      if grep -q '\[run\] success' "$slot_log" || grep -q '\[run\] retry-success' "$slot_log"; then
        log "canary succeeded"
        tail -n 80 "$slot_log" || true
        return 0
      fi
      if grep -q '\[slot\] finished' "$slot_log"; then
        log "canary finished without success"
        tail -n 120 "$slot_log" || true
        return 1
      fi
    fi
    sleep "$CANARY_POLL_SECONDS"
  done

  log "canary timed out without success"
  [ -f "$slot_log" ] && tail -n 120 "$slot_log" || true
  return 1
}

main() {
  launch_canary
  if ! wait_for_canary_success; then
    exit 1
  fi

  log "starting post-smoke dispatch with campaign tag=$POST_SMOKE_CAMPAIGN_TAG"
  env \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    CAMPAIGN_TAG="$POST_SMOKE_CAMPAIGN_TAG" \
    bash scripts/run_use1d_post_smoke_dispatch.sh
}

main "$@"
