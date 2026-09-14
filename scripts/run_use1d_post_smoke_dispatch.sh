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
TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-64}"
GCS_ASSET_PREFIX="${GCS_ASSET_PREFIX:-gs://llmvae-checkpoints/two-model-exploration/use1d_r6_assets_20260331}"
LOCAL_WHEELHOUSE_TAR="${LOCAL_WHEELHOUSE_TAR:-runs/offline_tpu_assets_20260331_wheelhouse_v3/wheelhouse.tar}"
LOCAL_HF_TAR="${LOCAL_HF_TAR:-runs/offline_tpu_assets_20260331_train/hf_cache.tar}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0+cpu}"
TORCH_XLA_VERSION="${TORCH_XLA_VERSION:-2.6.0}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-use1d_post_smoke_$(date +%Y%m%d_%H%M%S)}"
MULTI_WORKERS="${MULTI_WORKERS:-1 2 3 4 5 6}"
CONTROL_WORKERS="${CONTROL_WORKERS:-7 8}"
RAW_HERD_WORKER="${RAW_HERD_WORKER:-9}"
BOOTSTRAP_WORKERS_SET="${BOOTSTRAP_WORKERS_SET:-1 2 3 4 5 6 7 8 9}"
RAW_HERD_MANIFEST="${RAW_HERD_MANIFEST:-configs/aime_six_model_raw.tsv}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-configs/use1d_six_model_campaign.tsv}"
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

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  [ -f "$path" ] || { echo "Missing required file: $path" >&2; exit 1; }
}

bootstrap_one_worker() {
  local worker="$1"
  log "bootstrapping worker=$worker"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$worker" \
  BOOTSTRAP_WORKERS="$worker" \
  SYNC_WORKERS="$worker" \
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

launch_training_queue() {
  local worker="$1"
  local queue_tag="$2"
  local session_prefix="$3"
  local model_name="$4"
  local dataset_name="$5"
  local recipe_mode="$6"
  local run_offset="$7"
  local seed="$8"

  log "queue worker=$worker model=$model_name recipe=$recipe_mode"
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
  TRAIN_INLINE_AIME_BENCHMARK=0 \
  bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

launch_raw_aime24_herd() {
  local worker="$1"
  local session="aime24_raw_herd_${CAMPAIGN_TAG}"
  local remote_script="/tmp/${session}.sh"
  local remote_run_root="$REMOTE_ROOT/runs/${session}"

  log "launching raw AIME24 herd on worker=$worker"
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
export TPU_ACCELERATOR_TYPE='${TPU_ACCELERATOR_TYPE}'
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

main() {
  local -a multi_workers=()
  local -a control_workers=()
  local -a manifest_lines=()
  local manifest_line=""
  local worker=""
  local control_idx=0
  local multi_idx=0

require_file "$TRAIN_MANIFEST"
require_file "$RAW_HERD_MANIFEST"
  if [ -z "$GCS_ASSET_PREFIX" ]; then
    require_file "$LOCAL_WHEELHOUSE_TAR"
    require_file "$LOCAL_HF_TAR"
  fi

  for worker in $BOOTSTRAP_WORKERS_SET; do
    bootstrap_one_worker "$worker"
  done

  for worker in $MULTI_WORKERS; do
    multi_workers+=("$worker")
  done
  for worker in $CONTROL_WORKERS; do
    control_workers+=("$worker")
  done

  mapfile -t manifest_lines < "$TRAIN_MANIFEST"
  for manifest_line in "${manifest_lines[@]}"; do
    IFS=$'\t' read -r label model_name dataset_name eval_order single_round_control size_tier notes <<<"$manifest_line"
    [ -n "${label:-}" ] || continue
    case "$label" in
      \#*) continue ;;
    esac

    launch_training_queue "${multi_workers[$multi_idx]}" \
      "${CAMPAIGN_TAG}_${label}_mr" \
      "${CAMPAIGN_TAG}_${label}_mr" \
      "$model_name" "$dataset_name" "stable_trace_multi_round" "1" "0"
    multi_idx=$((multi_idx + 1))

    if [ "${single_round_control:-no}" = "yes" ]; then
      launch_training_queue "${control_workers[$control_idx]}" \
        "${CAMPAIGN_TAG}_${label}_sr" \
        "${CAMPAIGN_TAG}_${label}_sr" \
        "$model_name" "$dataset_name" "stable_trace_warmstart" "0" "0"
      control_idx=$((control_idx + 1))
    fi
  done

  launch_raw_aime24_herd "$RAW_HERD_WORKER"
  log "post-smoke dispatch launched"
}

main "$@"
