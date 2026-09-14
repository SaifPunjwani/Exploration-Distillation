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
ZONE="${ZONE:-us-central1-a}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-64}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"

CANARY_QUEUE_TAG="${CANARY_QUEUE_TAG:-uc1a_remote_smoke_$(date +%Y%m%d_%H%M%S)}"
CANARY_SESSION_PREFIX="${CANARY_SESSION_PREFIX:-uc1a_remote_smoke}"
CANARY_MODEL_NAME="${CANARY_MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
CANARY_DATASET_NAME="${CANARY_DATASET_NAME:-gsm8k}"
CANARY_TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-5400}"
CANARY_POLL_SECONDS="${CANARY_POLL_SECONDS:-20}"

RAW_AIME24_GROUP_PREFIX="${RAW_AIME24_GROUP_PREFIX:-uc1a_aime24_raw}"
RAW_AIME24_MAX_EXAMPLES="${RAW_AIME24_MAX_EXAMPLES:-30}"
RAW_AIME_NUM_ROLLOUTS="${RAW_AIME_NUM_ROLLOUTS:-32}"
RAW_AIME_TEMPERATURE="${RAW_AIME_TEMPERATURE:-1.0}"
RAW_AIME_TOP_P="${RAW_AIME_TOP_P:-1.0}"
RAW_AIME_PASS_K="${RAW_AIME_PASS_K:-1,4,8,16,32}"
RAW_AIME_MAX_COMPLETION_LEN="${RAW_AIME_MAX_COMPLETION_LEN:-16384}"
RAW_AIME_EVAL_BATCH_SIZE="${RAW_AIME_EVAL_BATCH_SIZE:-1}"
RAW_AIME_MAX_PROMPT_LEN="${RAW_AIME_MAX_PROMPT_LEN:-1024}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

slot_log_path() {
  printf '%s/runs/%s/_slot_logs/%s_w0_c0.log' \
    "$REMOTE_ROOT" "$CANARY_QUEUE_TAG" "$CANARY_SESSION_PREFIX"
}

launch_canary() {
  log "launching uc1a canary tag=$CANARY_QUEUE_TAG"
  env \
    TMX_DISABLE_BF16=1 \
    TMX_GRPO_GRAD_CKPT=1 \
    TMX_XLA_SAFE_GENERATE=1 \
    TMX_DEBUG_GENERATE="${TMX_DEBUG_GENERATE:-1}" \
    TMX_MASK_TRUNCATED_COMPLETIONS=0 \
    TPU_NAME="$TPU_NAME" \
    ZONE="$ZONE" \
    WORKERS=0 \
    LOCAL_CHIPS=4 \
    CHIPS=0 \
    SKIP_BOOTSTRAP=1 \
    TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE" \
    SESSION_PREFIX="$CANARY_SESSION_PREFIX" \
    QUEUE_TAG="$CANARY_QUEUE_TAG" \
    RUN_PROFILE=stable_trace \
    RECIPE_MODE=stable_trace_warmstart \
    TRAIN_SELECTION_POLICY=random \
    RUN_LIMIT=1 \
    SEED=0 \
    MODEL_NAME="$CANARY_MODEL_NAME" \
    DATASET_NAME="$CANARY_DATASET_NAME" \
    MAX_PROMPT_LEN=48 \
    MAX_COMPLETION_LEN=24 \
    EXPLORATION_MAX_COMPLETION_LEN=4 \
    EVAL_MAX_COMPLETION_LEN=8 \
    MAX_TOTAL_LEN=72 \
    GRPO_NUM_GENERATIONS=2 \
    GRPO_GENERATION_BATCH_SIZE=2 \
    NUM_SAMPLES_PER_PROMPT=2 \
    MAX_TRAIN_EXAMPLES=32 \
    MAX_EVAL_EXAMPLES=32 \
    GRPO_MAX_STEPS=2 \
    MAX_DISTILL_EXAMPLES=32 \
    MAX_ACTUAL_EXAMPLES=48 \
    RND_HIDDEN=32 \
    NOVELTY_LAYERS=21 \
    NOVELTY_FEATURE_MAX_LENGTH=24 \
    SAVE_BASE_MODEL=1 \
    TRAIN_INLINE_AIME_BENCHMARK=0 \
    bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

launch_raw_shard() {
  local chip="$1"
  local shard_name="$2"
  local manifest_path="$3"
  local session="uc1a_${shard_name}_c${chip}"
  local run_group="${RAW_AIME24_GROUP_PREFIX}_${shard_name}_$(date +%Y%m%d_%H%M%S)"
  local run_root="${REMOTE_ROOT}/runs/${run_group}"
  local remote_script="${REMOTE_ROOT}/tmp_${session}.sh"

  log "launching raw AIME24 shard=$shard_name chip=$chip manifest=$manifest_path"
  tmux kill-session -t "$session" 2>/dev/null || true
  cat > "$remote_script" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_ROOT"
. .venv_tpu_slice/bin/activate
mkdir -p "$run_root"
export PYTHONUNBUFFERED=1
export PYTHON_BIN=".venv_tpu_slice/bin/python"
export DEVICE="tpu"
export PJRT_DEVICE="TPU"
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TMX_ALLOW_SINGLE_WORKER_TPU=1
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE"
export TPU_VISIBLE_CHIPS="$chip"
export TPU_PROCESS_BOUNDS="1,1,1"
export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
export TPU_CHIPS_PER_HOST_BOUNDS="1,1,1"
export TPU_HOST_BOUNDS="1,1,1"
export TPU_WORKER_ID="0"
export TPU_WORKER_HOSTNAMES="localhost"
export MANIFEST_PATH="$manifest_path"
export RUN_GROUP="$run_group"
export USE_WANDB=1
export BENCHMARK_DATASET_NAME="aime_2024"
export MAX_BENCHMARK="$RAW_AIME24_MAX_EXAMPLES"
export BENCHMARK_NUM_ROLLOUTS="$RAW_AIME_NUM_ROLLOUTS"
export BENCHMARK_ROLLOUT_TEMPERATURE="$RAW_AIME_TEMPERATURE"
export BENCHMARK_ROLLOUT_TOP_P="$RAW_AIME_TOP_P"
export BENCHMARK_ROLLOUT_PASS_K="$RAW_AIME_PASS_K"
export BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$RAW_AIME_MAX_COMPLETION_LEN"
export BENCHMARK_EVAL_BATCH_SIZE="$RAW_AIME_EVAL_BATCH_SIZE"
export MAX_PROMPT_LEN="$RAW_AIME_MAX_PROMPT_LEN"
bash scripts/run_aime_model_herd.sh > "$run_root/console.log" 2>&1
EOF
  chmod +x "$remote_script"
  tmux new-session -d -s "$session" "$remote_script"
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
  launch_raw_shard 1 shard_a configs/aime_six_model_raw_shard_a.tsv
  launch_raw_shard 2 shard_b configs/aime_six_model_raw_shard_b.tsv
  launch_raw_shard 3 shard_c configs/aime_six_model_raw_shard_c.tsv
  launch_canary
  wait_for_canary_success
}

main "$@"
