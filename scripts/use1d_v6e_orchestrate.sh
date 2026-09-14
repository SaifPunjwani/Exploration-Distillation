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
ASSET_ROOT="${ASSET_ROOT:-runs/use1d_train_assets_20260331}"
WHEEL_TAR="${WHEEL_TAR:-$ASSET_ROOT/wheelhouse.tar}"
HF_TAR="${HF_TAR:-$ASSET_ROOT/hf_cache.tar}"
REMOTE_SSH_HOST="${REMOTE_SSH_HOST:?set REMOTE_SSH_HOST to the ssh host alias of the TPU VM}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

wait_for_file() {
  local path="$1"
  until [ -f "$path" ]; do
    log "waiting for asset: $path"
    sleep 30
  done
}

bootstrap_group() {
  local workers="$1"
  log "bootstrapping workers: $workers"
  local direct_ssh_host=""
  if [ "$workers" = "0" ]; then
    direct_ssh_host="$REMOTE_SSH_HOST"
  fi
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$workers" \
  BOOTSTRAP_WORKERS="$workers" \
  SYNC_WORKERS="$workers" \
  DIRECT_SSH_HOST="$direct_ssh_host" \
  GCS_ASSET_PREFIX="$GCS_ASSET_PREFIX" \
  ALLOW_DIRTY=1 \
  LOCAL_WHEELHOUSE_TAR="$WHEEL_TAR" \
  LOCAL_HF_TAR="$HF_TAR" \
  RESET_VENV=1 \
  SKIP_INSTALL=0 \
  SKIP_TPU_VALIDATE=0 \
  bash scripts/bootstrap_tpu_slice.sh
}

launch_queue() {
  local workers="$1"
  local local_chips="$2"
  local chips="$3"
  local session_prefix="$4"
  local queue_tag="$5"
  local recipe_mode="$6"
  local seed="$7"
  local run_limit="$8"
  shift 8

  log "launching queue tag=$queue_tag workers=[$workers] recipe=$recipe_mode seed=$seed"
  env \
    TMX_DISABLE_BF16=1 \
    TMX_XLA_SAFE_GENERATE=1 \
    TMX_MASK_TRUNCATED_COMPLETIONS=0 \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    WORKERS="$workers" \
    LOCAL_CHIPS="$local_chips" \
    CHIPS="$chips" \
    SKIP_BOOTSTRAP=1 \
    SESSION_PREFIX="$session_prefix" \
    QUEUE_TAG="$queue_tag" \
    RUN_PROFILE=stable_trace \
    RECIPE_MODE="$recipe_mode" \
    SEED="$seed" \
    RUN_LIMIT="$run_limit" \
    MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct" \
    DATASET_NAME="gsm8k" \
    MAX_PROMPT_LEN=64 \
    MAX_COMPLETION_LEN=32 \
    EXPLORATION_MAX_COMPLETION_LEN=8 \
    EVAL_MAX_COMPLETION_LEN=16 \
    MAX_TOTAL_LEN=96 \
    GRPO_NUM_GENERATIONS=2 \
    GRPO_GENERATION_BATCH_SIZE=2 \
    NUM_SAMPLES_PER_PROMPT=2 \
    GRPO_LOSS_TYPE=grpo \
    RND_HIDDEN=64 \
    NOVELTY_LAYERS=21 \
    NOVELTY_FEATURE_MAX_LENGTH=32 \
    TRAIN_PROBE_POOL_SIZE=256 \
    MAX_TRAIN_EXAMPLES=256 \
    MAX_EVAL_EXAMPLES=128 \
    GRPO_MAX_STEPS=24 \
    MAX_DISTILL_EXAMPLES=256 \
    MAX_ACTUAL_EXAMPLES=384 \
    "$@" \
    bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

check_canary_health() {
  local queue_tag="$1"
  local session_prefix="$2"
  local slot_log="$REMOTE_ROOT/runs/$queue_tag/_slot_logs/${session_prefix}_w0_c0.log"
  local run_name="${queue_tag}_two_trace_ws_distill_dl5e6"
  local console_log="$REMOTE_ROOT/runs/$queue_tag/$run_name/console.log"
  local attempt=0
  local saw_grpo=0

  while [ "$attempt" -lt 30 ]; do
    out="$(
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_SSH_HOST" \
        "test -f '$slot_log' && tail -n 60 '$slot_log'; echo '__TMX_SPLIT__'; test -f '$console_log' && tail -n 120 '$console_log'" 2>/dev/null || true
    )"
    if printf '%s' "$out" | grep -q '\[run\] failure'; then
      log "canary reported failure"
      printf '%s\n' "$out"
      return 1
    fi
    if printf '%s' "$out" | grep -q 'Starting GRPO training (Explorer)'; then
      saw_grpo=1
    fi
    if [ "$saw_grpo" -eq 1 ] && ! printf '%s' "$out" | grep -q '\[run\] failure'; then
      log "canary reached Explorer GRPO without immediate failure"
      printf '%s\n' "$out"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 20
  done

  log "canary did not fail, but no Explorer GRPO signal arrived in time"
  return 1
}

main() {
  log "starting use1d orchestration on $TPU_NAME ($ZONE)"
  wait_for_file "$WHEEL_TAR"
  wait_for_file "$HF_TAR"

  local canary_tag="use1d_canary_20260331a"
  local canary_prefix="use1d_canary"

  bootstrap_group "0"
  launch_queue "0" "1" "0" "$canary_prefix" "$canary_tag" "stable_trace_warmstart" "0" "1" \
    MAX_TRAIN_EXAMPLES=128 MAX_EVAL_EXAMPLES=64 GRPO_MAX_STEPS=12 MAX_DISTILL_EXAMPLES=128 MAX_ACTUAL_EXAMPLES=192

  if ! check_canary_health "$canary_tag" "$canary_prefix"; then
    log "stopping after canary; investigate remote logs before scaling"
    exit 1
  fi

  bootstrap_group "0 1"
  bootstrap_group "2 3"
  bootstrap_group "4 5"
  bootstrap_group "6 7"

  launch_queue "0 1" "4" "" "use1d_ws_s0" "use1d_ws_s0_20260331" "stable_trace_warmstart" "0" "6"
  launch_queue "2 3" "4" "" "use1d_ws_s1" "use1d_ws_s1_20260331" "stable_trace_warmstart" "1" "6"
  launch_queue "4 5" "4" "" "use1d_mr_s0" "use1d_mr_s0_20260331" "stable_trace_multi_round" "0" "5"
  launch_queue "6 7" "4" "" "use1d_mr_s1" "use1d_mr_s1_20260331" "stable_trace_multi_round" "1" "5"

  log "launched warm-start and multi-round queues on use1d"
}

main "$@"
