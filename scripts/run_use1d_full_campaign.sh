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
REMOTE_SSH_HOST="${REMOTE_SSH_HOST:?set REMOTE_SSH_HOST to the ssh host alias of the TPU VM}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-use1d_six_model_campaign_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/${CAMPAIGN_TAG}}"
SAVED_MODEL_ROOT="${SAVED_MODEL_ROOT:-$REMOTE_ROOT/$RUN_ROOT/saved_models}"
RAW_HERD_MANIFEST="${RAW_HERD_MANIFEST:-configs/aime_six_model_raw.tsv}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-configs/use1d_six_model_campaign.tsv}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331}"
LOCAL_WHEELHOUSE_TAR="${LOCAL_WHEELHOUSE_TAR:-runs/offline_tpu_assets_20260331_wheelhouse_v3/wheelhouse.tar}"
LOCAL_HF_TAR="${LOCAL_HF_TAR:-runs/offline_tpu_assets_20260331_train/hf_cache.tar}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0+cpu}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.6.0}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-dapo_math_17k}"
AIME24_MAX_EXAMPLES="${AIME24_MAX_EXAMPLES:-30}"
AIME_NUM_ROLLOUTS="${AIME_NUM_ROLLOUTS:-32}"
AIME_TEMPERATURE="${AIME_TEMPERATURE:-1.0}"
AIME_TOP_P="${AIME_TOP_P:-1.0}"
AIME_PASS_K="${AIME_PASS_K:-1,4,8,16,32}"
AIME_MAX_COMPLETION_LEN="${AIME_MAX_COMPLETION_LEN:-16384}"
AIME_EVAL_BATCH_SIZE="${AIME_EVAL_BATCH_SIZE:-1}"
TRAIN_MAX_PROMPT_LEN="${TRAIN_MAX_PROMPT_LEN:-64}"
TRAIN_MAX_COMPLETION_LEN="${TRAIN_MAX_COMPLETION_LEN:-32}"
TRAIN_EXPLORATION_MAX_COMPLETION_LEN="${TRAIN_EXPLORATION_MAX_COMPLETION_LEN:-8}"
TRAIN_EVAL_MAX_COMPLETION_LEN="${TRAIN_EVAL_MAX_COMPLETION_LEN:-16}"
TRAIN_MAX_TOTAL_LEN="${TRAIN_MAX_TOTAL_LEN:-96}"
TRAIN_GRPO_NUM_GENERATIONS="${TRAIN_GRPO_NUM_GENERATIONS:-2}"
TRAIN_GRPO_GENERATION_BATCH_SIZE="${TRAIN_GRPO_GENERATION_BATCH_SIZE:-2}"
TRAIN_NUM_SAMPLES_PER_PROMPT="${TRAIN_NUM_SAMPLES_PER_PROMPT:-2}"
TRAIN_MAX_TRAIN_EXAMPLES="${TRAIN_MAX_TRAIN_EXAMPLES:-256}"
TRAIN_MAX_EVAL_EXAMPLES="${TRAIN_MAX_EVAL_EXAMPLES:-128}"
TRAIN_GRPO_MAX_STEPS="${TRAIN_GRPO_MAX_STEPS:-24}"
TRAIN_MAX_DISTILL_EXAMPLES="${TRAIN_MAX_DISTILL_EXAMPLES:-256}"
TRAIN_MAX_ACTUAL_EXAMPLES="${TRAIN_MAX_ACTUAL_EXAMPLES:-384}"
TRAIN_RND_HIDDEN="${TRAIN_RND_HIDDEN:-64}"
TRAIN_NOVELTY_LAYERS="${TRAIN_NOVELTY_LAYERS:-21}"
TRAIN_NOVELTY_FEATURE_MAX_LENGTH="${TRAIN_NOVELTY_FEATURE_MAX_LENGTH:-32}"
TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8}"
RAW_HERD_WORKER="${RAW_HERD_WORKER:-8}"
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-64}"
RESUME_AFTER_CANARY="${RESUME_AFTER_CANARY:-0}"
BOOTSTRAP_CONCURRENCY="${BOOTSTRAP_CONCURRENCY:-1}"
DISPATCH_ONLY="${DISPATCH_ONLY:-0}"
TRAIN_INLINE_AIME_BENCHMARK="${TRAIN_INLINE_AIME_BENCHMARK:-0}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_gcs_assets() {
  local wheel_gcs="${GCS_ASSET_PREFIX%/}/$(basename "$LOCAL_WHEELHOUSE_TAR")"
  local hf_gcs="${GCS_ASSET_PREFIX%/}/$(basename "$LOCAL_HF_TAR")"
  until gcloud storage ls "$wheel_gcs" >/dev/null 2>&1 && gcloud storage ls "$hf_gcs" >/dev/null 2>&1; do
    log "waiting for GCS assets under $GCS_ASSET_PREFIX"
    sleep 20
  done
}

require_use1d_ready() {
  local state health
  state="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" --zone="$ZONE" --format='value(state)' 2>/dev/null || true)"
  health="$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" --zone="$ZONE" --format='value(health)' 2>/dev/null || true)"
  if [ "$state" != "READY" ] || [ "$health" != "HEALTHY" ]; then
    echo "TPU not ready: state=${state:-unknown} health=${health:-unknown}" >&2
    exit 1
  fi
}

bootstrap_one_worker() {
  local worker="$1"
  local direct_ssh_host=""
  if [ "$worker" = "0" ]; then
    direct_ssh_host="$REMOTE_SSH_HOST"
  fi
  log "bootstrapping worker: $worker"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$worker" \
  BOOTSTRAP_WORKERS="$worker" \
  SYNC_WORKERS="$worker" \
  DIRECT_SSH_HOST="$direct_ssh_host" \
  GCS_ASSET_PREFIX="$GCS_ASSET_PREFIX" \
  LOCAL_WHEELHOUSE_TAR="$LOCAL_WHEELHOUSE_TAR" \
  LOCAL_HF_TAR="$LOCAL_HF_TAR" \
  TORCH_VERSION="$TORCH_VERSION" \
  TORCH_XLA_VERSION="$TORCH_XLA_VERSION" \
  RESET_VENV=1 \
  SKIP_INSTALL=0 \
  ALLOW_DIRTY=1 \
  bash scripts/bootstrap_tpu_slice.sh
}

bootstrap_workers() {
  local workers="$1"
  local worker
  local failed=0
  local -a worker_list=()
  local -a pids=()
  local i

  for worker in $workers; do
    worker_list+=("$worker")
  done

  for ((i = 0; i < ${#worker_list[@]}; i += BOOTSTRAP_CONCURRENCY)); do
    pids=()
    for worker in "${worker_list[@]:i:BOOTSTRAP_CONCURRENCY}"; do
      bootstrap_one_worker "$worker" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
  done
  if [ "$failed" -ne 0 ]; then
    echo "One or more worker bootstraps failed." >&2
    return 1
  fi
}

launch_training_queue() {
  local worker="$1"
  local queue_tag="$2"
  local session_prefix="$3"
  local model_name="$4"
  local dataset_name="$5"
  local recipe_mode="$6"
  local run_offset="$7"
  local seed="$8"

  log "launching training queue: worker=$worker model=$model_name recipe=$recipe_mode offset=$run_offset seed=$seed"
  TMX_DISABLE_BF16='1' \
  TMX_XLA_SAFE_GENERATE='1' \
  TMX_MASK_TRUNCATED_COMPLETIONS='0' \
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$worker" \
  LOCAL_CHIPS=1 \
  CHIPS=0 \
  SKIP_BOOTSTRAP=1 \
  SESSION_PREFIX="$session_prefix" \
  QUEUE_TAG="$queue_tag" \
  RUN_PROFILE=stable_trace \
  RECIPE_MODE="$recipe_mode" \
  RUN_OFFSET="$run_offset" \
  RUN_LIMIT=1 \
  SEED="$seed" \
  MODEL_NAME="$model_name" \
  DATASET_NAME="$dataset_name" \
  MAX_PROMPT_LEN="$TRAIN_MAX_PROMPT_LEN" \
  MAX_COMPLETION_LEN="$TRAIN_MAX_COMPLETION_LEN" \
  EXPLORATION_MAX_COMPLETION_LEN="$TRAIN_EXPLORATION_MAX_COMPLETION_LEN" \
  EVAL_MAX_COMPLETION_LEN="$TRAIN_EVAL_MAX_COMPLETION_LEN" \
  MAX_TOTAL_LEN="$TRAIN_MAX_TOTAL_LEN" \
  GRPO_NUM_GENERATIONS="$TRAIN_GRPO_NUM_GENERATIONS" \
  GRPO_GENERATION_BATCH_SIZE="$TRAIN_GRPO_GENERATION_BATCH_SIZE" \
  NUM_SAMPLES_PER_PROMPT="$TRAIN_NUM_SAMPLES_PER_PROMPT" \
  MAX_TRAIN_EXAMPLES="$TRAIN_MAX_TRAIN_EXAMPLES" \
  MAX_EVAL_EXAMPLES="$TRAIN_MAX_EVAL_EXAMPLES" \
  GRPO_MAX_STEPS="$TRAIN_GRPO_MAX_STEPS" \
  MAX_DISTILL_EXAMPLES="$TRAIN_MAX_DISTILL_EXAMPLES" \
  MAX_ACTUAL_EXAMPLES="$TRAIN_MAX_ACTUAL_EXAMPLES" \
  RND_HIDDEN="$TRAIN_RND_HIDDEN" \
  NOVELTY_LAYERS="$TRAIN_NOVELTY_LAYERS" \
  NOVELTY_FEATURE_MAX_LENGTH="$TRAIN_NOVELTY_FEATURE_MAX_LENGTH" \
  SAVE_BASE_MODEL=1 \
  SAVED_MODEL_ROOT="$SAVED_MODEL_ROOT" \
  BENCHMARK_DATASET_NAME="$(
    if [ "$TRAIN_INLINE_AIME_BENCHMARK" = "1" ]; then
      printf 'aime_2024'
    fi
  )" \
  MAX_BENCHMARK="$AIME24_MAX_EXAMPLES" \
  BENCHMARK_EVAL_MODE=sampled \
  BENCHMARK_NUM_ROLLOUTS="$AIME_NUM_ROLLOUTS" \
  BENCHMARK_ROLLOUT_TEMPERATURE="$AIME_TEMPERATURE" \
  BENCHMARK_ROLLOUT_TOP_P="$AIME_TOP_P" \
  BENCHMARK_ROLLOUT_PASS_K="$AIME_PASS_K" \
  BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN="$AIME_MAX_COMPLETION_LEN" \
  BENCHMARK_EVAL_BATCH_SIZE="$AIME_EVAL_BATCH_SIZE" \
  bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

launch_raw_aime24_herd() {
  local worker="$1"
  local session="aime24_raw_herd_${CAMPAIGN_TAG}"
  local remote_script="/tmp/${session}.sh"
  local remote_run_root="$REMOTE_ROOT/runs/${session}"

  log "launching raw AIME24 herd eval on worker $worker"
  gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone="$ZONE" \
    --worker="$worker" \
    --command="cat > '$remote_script' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
. .venv_tpu_slice/bin/activate
mkdir -p '$remote_run_root'
export PYTHONUNBUFFERED=1
export PYTHON_BIN='.venv_tpu_slice/bin/python'
export DEVICE='tpu'
export PJRT_DEVICE='TPU'
export TMX_TPU_SLICE_LAUNCHED=1
export TMX_REQUIRE_TPU=1
export TMX_ALLOW_SINGLE_WORKER_TPU=1
export TPU_SKIP_MDS_QUERY=1
export TPU_ACCELERATOR_TYPE='$TPU_ACCELERATOR_TYPE'
export TPU_VISIBLE_CHIPS='0'
export TPU_PROCESS_BOUNDS='1,1,1'
export TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1'
export TPU_CHIPS_PER_HOST_BOUNDS='1,1,1'
export TPU_HOST_BOUNDS='1,1,1'
export TPU_WORKER_ID='$worker'
export TPU_WORKER_HOSTNAMES='localhost'
export MANIFEST_PATH='$RAW_HERD_MANIFEST'
export RUN_GROUP='${session}'
export BENCHMARK_DATASET_NAME='aime_2024'
export MAX_BENCHMARK='${AIME24_MAX_EXAMPLES}'
export BENCHMARK_NUM_ROLLOUTS='${AIME_NUM_ROLLOUTS}'
export BENCHMARK_ROLLOUT_TEMPERATURE='${AIME_TEMPERATURE}'
export BENCHMARK_ROLLOUT_TOP_P='${AIME_TOP_P}'
export BENCHMARK_ROLLOUT_PASS_K='${AIME_PASS_K}'
export BENCHMARK_ROLLOUT_MAX_COMPLETION_LEN='${AIME_MAX_COMPLETION_LEN}'
export BENCHMARK_EVAL_BATCH_SIZE='${AIME_EVAL_BATCH_SIZE}'
bash scripts/run_aime_model_herd.sh > '${remote_run_root}/console.log' 2>&1
EOF
chmod +x '$remote_script'
tmux kill-session -t '$session' 2>/dev/null || true
tmux new-session -d -s '$session' '$remote_script'"
}

launch_raw_aime24_herd_best_effort() {
  local worker="$1"
  local attempts="${2:-3}"
  local attempt=1
  while [ "$attempt" -le "$attempts" ]; do
    if launch_raw_aime24_herd "$worker"; then
      return 0
    fi
    log "raw AIME24 herd launch failed on worker $worker (attempt $attempt/$attempts); retrying"
    sleep 20
    attempt=$((attempt + 1))
  done
  log "raw AIME24 herd launch failed after $attempts attempts; continuing without blocking training dispatch"
  return 0
}

check_canary_health() {
  local queue_tag="$1"
  local session_prefix="$2"
  local run_name="${queue_tag}_mr_trace_ws_distill_dl5e6_r4_topic"
  local slot_log="$REMOTE_ROOT/runs/$queue_tag/_slot_logs/${session_prefix}_w0_c0.log"
  local console_log="$REMOTE_ROOT/runs/$queue_tag/$run_name/console.log"
  local attempt=0

  while [ "$attempt" -lt 30 ]; do
    out="$(
      ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE_SSH_HOST" \
        "test -f '$slot_log' && tail -n 80 '$slot_log'; echo '__TMX_SPLIT__'; test -f '$console_log' && tail -n 120 '$console_log'" 2>/dev/null || true
    )"
    if printf '%s' "$out" | grep -q '\[run\] failure'; then
      log "canary reported failure"
      printf '%s\n' "$out"
      return 1
    fi
    if printf '%s' "$out" | grep -q 'Starting GRPO training (Explorer)'; then
      log "canary reached Explorer GRPO"
      printf '%s\n' "$out"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 20
  done

  log "canary timed out before Explorer GRPO signal"
  return 1
}

launch_full_campaign() {
  local control_worker=6
  local next_multi_worker=0
  local manifest_line=""
  local -a manifest_lines=()

  mapfile -t manifest_lines < "$TRAIN_MANIFEST"
  for manifest_line in "${manifest_lines[@]}"; do
    IFS=$'\t' read -r label model_name dataset_name eval_order single_round_control _size_tier notes <<<"$manifest_line"
    [ -n "${label:-}" ] || continue
    case "$label" in
      \#*) continue ;;
    esac

    local mr_queue_tag="${CAMPAIGN_TAG}_${label}_mr"
    local mr_session_prefix="${CAMPAIGN_TAG}_${label}_mr"
    launch_training_queue "$next_multi_worker" "$mr_queue_tag" "$mr_session_prefix" "$model_name" "$dataset_name" "stable_trace_multi_round" "1" "0"
    next_multi_worker=$((next_multi_worker + 1))

    if [ "${single_round_control:-no}" = "yes" ]; then
      local sr_queue_tag="${CAMPAIGN_TAG}_${label}_sr"
      local sr_session_prefix="${CAMPAIGN_TAG}_${label}_sr"
      launch_training_queue "$control_worker" "$sr_queue_tag" "$sr_session_prefix" "$model_name" "$dataset_name" "stable_trace_warmstart" "0" "0"
      control_worker=$((control_worker + 1))
    fi
  done
}

main() {
  local canary_tag="${CAMPAIGN_TAG}_canary"
  local canary_prefix="${CAMPAIGN_TAG}_canary"

  require_file "$RAW_HERD_MANIFEST"
  require_file "$TRAIN_MANIFEST"
  require_file "$LOCAL_WHEELHOUSE_TAR"
  require_file "$LOCAL_HF_TAR"
  require_use1d_ready
  require_gcs_assets

  if [ "$RESUME_AFTER_CANARY" != "1" ]; then
    bootstrap_workers "0"
    launch_training_queue "0" "$canary_tag" "$canary_prefix" "Qwen/Qwen2.5-Math-1.5B-Instruct" "gsm8k" "stable_trace_multi_round" "1" "0"
    if ! check_canary_health "$canary_tag" "$canary_prefix"; then
      echo "Canary failed. Inspect remote logs under runs/${canary_tag}" >&2
      exit 1
    fi
  else
    log "RESUME_AFTER_CANARY=1; skipping canary bootstrap/launch."
  fi

  if [ "$DISPATCH_ONLY" = "1" ]; then
    log "DISPATCH_ONLY=1; skipping worker bootstrap and dispatching directly."
  else
    bootstrap_workers "$TRAIN_WORKERS"
  fi
  launch_full_campaign
  launch_raw_aime24_herd_best_effort "$RAW_HERD_WORKER"

  log "full campaign launched"
  log "saved model root: $SAVED_MODEL_ROOT"
  log "raw AIME24 herd session: aime24_raw_herd_${CAMPAIGN_TAG}"
  log "AIME25 remains gated on AIME24 completion and trained checkpoint selection."
}

main "$@"
