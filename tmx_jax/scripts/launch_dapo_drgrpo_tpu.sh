#!/usr/bin/env bash
# Self-contained DAPO/Dr.GRPO TPU launcher for the JAX repo.
#
# Default mode uses only the existing TRC v5litepod-256, split into train and
# rollout worker sets. If TRAIN_TPU_NAME/TRAIN_TPU_PROJECT/TRAIN_ZONE are set to
# an owned nanny-created train TPU, that TPU is used for JAX train and the TRC
# v5litepod-256 is used for rollout/vLLM.
set -euo pipefail
trap 'rc=$?; [ "$rc" = "0" ] || echo "[dapo-launch] EXIT rc=$rc line=$LINENO" >&2' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

bool_enabled() {
  case "${1:-0}" in
    1|true|True|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

join_by_comma() {
  local first=1
  for item in "$@"; do
    if [ "$first" = "1" ]; then
      printf '%s' "$item"
      first=0
    else
      printf ',%s' "$item"
    fi
  done
}

worker_ips() {
  local project="$1"
  local zone="$2"
  local tpu_name="$3"
  gcloud --project="$project" alpha compute tpus tpu-vm describe "$tpu_name" \
    --zone="$zone" \
    --format='get(networkEndpoints[].ipAddress)'
}

tpu_region_from_zone() {
  local zone="$1"
  printf '%s\n' "${zone%-*}"
}

assert_tpu_ready() {
  local project="$1"
  local zone="$2"
  local tpu_name="$3"
  local expected="${4:-}"
  local desc state health accelerator workers
  desc="$(
    gcloud --project="$project" alpha compute tpus tpu-vm describe "$tpu_name" \
      --zone="$zone" \
      --format=json 2>/dev/null || true
  )"
  if [ -z "$desc" ]; then
    echo "[dapo-launch] ERROR: cannot describe $project/$zone/$tpu_name" >&2
    return 2
  fi
  read -r state health accelerator workers < <(
    TPU_DESC_JSON="$desc" python3 - <<'PY'
import json
import os
raw = json.loads(os.environ["TPU_DESC_JSON"])
print(
    raw.get("state") or "UNKNOWN",
    raw.get("health") or "UNKNOWN",
    raw.get("acceleratorType") or "UNKNOWN",
    len(raw.get("networkEndpoints") or []),
)
PY
  )
  if [ "$state" != "READY" ] || [ "$health" != "HEALTHY" ]; then
    echo "[dapo-launch] ERROR: $project/$zone/$tpu_name state=$state health=$health; expected READY/HEALTHY" >&2
    return 2
  fi
  if [ -n "$expected" ] && [ "$accelerator" != "$expected" ]; then
    echo "[dapo-launch] ERROR: $project/$zone/$tpu_name accelerator=$accelerator; expected $expected" >&2
    return 2
  fi
  echo "[dapo-launch] OK $project/$zone/$tpu_name accelerator=$accelerator workers=$workers"
}

clean_workers() {
  local project="$1"
  local zone="$2"
  local tpu_name="$3"
  local workers="$4"
  local label="$5"
  local worker_count
  worker_count="$(printf '%s\n' "$workers" | wc -w | tr -d ' ')"
  echo "[dapo-launch] cleaning $label workers=[$workers] on $project/$zone/$tpu_name"
  if [ "${TMX_FAST_CLEAN_SLICE:-0}" = "1" ] \
    && [ "$worker_count" -ge 16 ] \
    && [ "$project/$zone/$tpu_name" != "$TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME" ]; then
    gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --zone="$zone" \
      --worker=all \
      --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="
        pkill -9 -f '[p]ython -m tmx_jax' 2>/dev/null || true
        pkill -9 -f '[V]LLM::EngineCore|[v]llm.entrypoints.openai.api_server|[v]llm serve' 2>/dev/null || true
        pkill -9 -f '[t]mx_vllm_proxy_' 2>/dev/null || true
        tmux ls 2>/dev/null | while IFS=: read -r s rest; do
          case \"\$s\" in *jax_*|*dapo*|*pipeline*|*vllm*) tmux kill-session -t \"\$s\" 2>/dev/null || true ;; esac
        done
      " >/dev/null || true
    return 0
  fi
  for worker in $workers; do
    gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --zone="$zone" \
      --worker="$worker" \
      --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --ssh-flag="-o ServerAliveInterval=${GCLOUD_SSH_SERVER_ALIVE_INTERVAL:-10}" \
      --ssh-flag="-o ServerAliveCountMax=${GCLOUD_SSH_SERVER_ALIVE_COUNT_MAX:-3}" \
      --command="
        pkill -9 -f '[p]ython -m tmx_jax' 2>/dev/null || true
        pkill -9 -f '[V]LLM::EngineCore|[v]llm.entrypoints.openai.api_server|[v]llm serve' 2>/dev/null || true
        pkill -9 -f '[t]mx_vllm_proxy_' 2>/dev/null || true
        tmux ls 2>/dev/null | while IFS=: read -r s rest; do
          case \"\$s\" in *jax_*|*dapo*|*pipeline*|*vllm*) tmux kill-session -t \"\$s\" 2>/dev/null || true ;; esac
        done
      " >/dev/null || true
  done
}

healthy_workers() {
  local project="$1"
  local zone="$2"
  local tpu_name="$3"
  local workers="$4"
  local healthy=""
  for worker in $workers; do
    if gcloud --project="$project" alpha compute tpus tpu-vm ssh "$tpu_name" \
      --zone="$zone" \
      --worker="$worker" \
      --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=${GCLOUD_SSH_CONNECT_TIMEOUT_SECONDS:-30}" \
      --command="curl -fsS --max-time 3 http://127.0.0.1:${PORT}/health >/dev/null" \
      >/dev/null 2>&1; then
      healthy="$healthy $worker"
    fi
  done
  printf '%s\n' "$healthy" | xargs || true
}

TRC2_PROJECT="${TRC2_PROJECT:?set TRC2_PROJECT to the GCP project id of the TRC allocation}"
TRC2_ZONE="${TRC2_ZONE:-europe-west4-b}"
TRC2_TPU_NAME="${TRC2_TPU_NAME:-trc2-v5lite256-ew4b-r1}"

SOURCE_RUN="${SOURCE_RUN:-dapo16k_drgrpo_novelty05_hf_resume49_20260429_174651_wandbresume_parallelvllm}"
SOURCE_STEP="${SOURCE_STEP:-000100}"
LAMBDA_NOVELTY="${LAMBDA_NOVELTY:-0.5}"
RUN_NAME="${RUN_NAME:-dapo16k_drgrpo_novelty${LAMBDA_NOVELTY/./}_jaxrepo_resume${SOURCE_STEP}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-${DATASET:-dapo_math_17k}}"
REMOTE_REPO="${REMOTE_REPO:?set REMOTE_REPO to the absolute repo path on every TPU worker}"
RUNS_ROOT="${RUNS_ROOT:?set RUNS_ROOT to the absolute runs directory on the TPU VM}"

TRAIN_TPU_PROJECT="${TRAIN_TPU_PROJECT:-$TRC2_PROJECT}"
TRAIN_ZONE="${TRAIN_ZONE:-$TRC2_ZONE}"
TRAIN_TPU_NAME="${TRAIN_TPU_NAME:-$TRC2_TPU_NAME}"
SERVE_TPU_PROJECT="${SERVE_TPU_PROJECT:-$TRC2_PROJECT}"
SERVE_ZONE="${SERVE_ZONE:-$TRC2_ZONE}"
SERVE_TPU_NAME="${SERVE_TPU_NAME:-$TRC2_TPU_NAME}"

train_is_trc=0
serve_is_trc=0
if [ "$TRAIN_TPU_PROJECT/$TRAIN_ZONE/$TRAIN_TPU_NAME" = "$TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME" ]; then
  train_is_trc=1
fi
if [ "$SERVE_TPU_PROJECT/$SERVE_ZONE/$SERVE_TPU_NAME" = "$TRC2_PROJECT/$TRC2_ZONE/$TRC2_TPU_NAME" ]; then
  serve_is_trc=1
fi
if [ "$train_is_trc" = "1" ] || [ "$serve_is_trc" = "1" ]; then
  export TRC2_PROJECT TRC2_ZONE TRC2_TPU_NAME
  "$SCRIPT_DIR/trc2_v5lite256_guard.sh" "$@"
fi
if [ "$(tpu_region_from_zone "$TRAIN_ZONE")" != "$(tpu_region_from_zone "$SERVE_ZONE")" ] && ! bool_enabled "${TMX_ALLOW_CROSS_REGION_TPU:-0}"; then
  echo "[dapo-launch] ERROR: refusing cross-region train/serve topology: train=$TRAIN_ZONE serve=$SERVE_ZONE" >&2
  exit 2
fi

if [ "$train_is_trc" = "1" ]; then
  TRAIN_WORKERS="${TRAIN_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
  TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5litepod-256}"
  TMX_USE_MANUAL_TPU_TOPOLOGY="${TMX_USE_MANUAL_TPU_TOPOLOGY:-1}"
  TMX_TPU_HOST_BOUNDS="${TMX_TPU_HOST_BOUNDS:-4,4,1}"
else
  TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"
  TMX_USE_MANUAL_TPU_TOPOLOGY="${TMX_USE_MANUAL_TPU_TOPOLOGY:-1}"
  TMX_TPU_HOST_BOUNDS="${TMX_TPU_HOST_BOUNDS:-1,1,1}"
  TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
fi
if [ "$serve_is_trc" = "1" ]; then
  if [ "$train_is_trc" = "1" ]; then
    SERVE_WORKERS="${SERVE_WORKERS:-16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63}"
    MIN_HEALTHY_SERVE_WORKERS="${MIN_HEALTHY_SERVE_WORKERS:-40}"
  else
    SERVE_WORKERS="${SERVE_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63}"
    MIN_HEALTHY_SERVE_WORKERS="${MIN_HEALTHY_SERVE_WORKERS:-48}"
  fi
else
  SERVE_WORKERS="${SERVE_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"
  MIN_HEALTHY_SERVE_WORKERS="${MIN_HEALTHY_SERVE_WORKERS:-12}"
fi

assert_tpu_ready "$TRAIN_TPU_PROJECT" "$TRAIN_ZONE" "$TRAIN_TPU_NAME" "${TMX_EXPECTED_TRAIN_ACCELERATOR:-}"
assert_tpu_ready "$SERVE_TPU_PROJECT" "$SERVE_ZONE" "$SERVE_TPU_NAME" "${TMX_EXPECTED_SERVE_ACCELERATOR:-}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-16384}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-18432}"
SOFT_OVERLONG_EXPECTED_LEN="${SOFT_OVERLONG_EXPECTED_LEN:-13107}"
SOFT_OVERLONG_CACHE_LEN="${SOFT_OVERLONG_CACHE_LEN:-3277}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-16}"
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10}"
PORT="${PORT:-8000}"

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MAX_TOTAL_LEN}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-$MAX_TOTAL_LEN}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-4}"
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-192}"
VLLM_FANOUT_PER_PROMPT="${VLLM_FANOUT_PER_PROMPT:-16}"
TMX_VLLM_RANDOMIZE_SEEDS="${TMX_VLLM_RANDOMIZE_SEEDS:-1}"
TMX_VLLM_SEED_MODE="${TMX_VLLM_SEED_MODE:-prompt_nonce}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-hf://SaifPunjwani/two-model-exploration-checkpoints/${SOURCE_RUN}/explorer/checkpoints/step_${SOURCE_STEP}}"

if bool_enabled "${DRY_RUN:-0}"; then
  echo "[dapo-launch] DRY_RUN=1"
  echo "[dapo-launch] run=$RUN_NAME"
  echo "[dapo-launch] train=$TRAIN_TPU_PROJECT/$TRAIN_ZONE/$TRAIN_TPU_NAME workers=[$TRAIN_WORKERS] accelerator=$TPU_ACCELERATOR_TYPE host_bounds=$TMX_TPU_HOST_BOUNDS"
  echo "[dapo-launch] serve=$SERVE_TPU_PROJECT/$SERVE_ZONE/$SERVE_TPU_NAME workers=[$SERVE_WORKERS]"
  echo "[dapo-launch] checkpoint=$INIT_CHECKPOINT"
  echo "[dapo-launch] dataset=$DATASET_NAME"
  echo "[dapo-launch] contract=lambda=$LAMBDA_NOVELTY G=$GRPO_NUM_GENERATIONS max_completion=$MAX_COMPLETION_LEN Dr.GRPO clip=0.2/0.28 KL=0"
  exit 0
fi

if bool_enabled "${CLEAN_SERVE_SLICE:-1}"; then
  clean_workers "$SERVE_TPU_PROJECT" "$SERVE_ZONE" "$SERVE_TPU_NAME" "$SERVE_WORKERS" "serve"
fi

if bool_enabled "${BOOTSTRAP_SERVE_VLLM:-1}"; then
  echo "[dapo-launch] bootstrapping serve vLLM on $SERVE_TPU_PROJECT/$SERVE_ZONE/$SERVE_TPU_NAME workers=[$SERVE_WORKERS]"
  (
    cd "$REPO_ROOT"
    TPU_PROJECT="$SERVE_TPU_PROJECT" \
    TPU_NAME="$SERVE_TPU_NAME" \
    ZONE="$SERVE_ZONE" \
    WORKERS="$SERVE_WORKERS" \
    MODEL="$MODEL_NAME" \
    PORT="$PORT" \
    TP="$VLLM_TP_SIZE" \
    MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    VLLM_TPU_BUCKET_PADDING_GAP="${VLLM_TPU_BUCKET_PADDING_GAP:-4096}" \
    VLLM_TPU_MOST_MODEL_LEN="${VLLM_TPU_MOST_MODEL_LEN:-$MAX_TOTAL_LEN}" \
    VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
    VLLM_V1_USE_PREFILL_DECODE_ATTENTION="${VLLM_V1_USE_PREFILL_DECODE_ATTENTION:-1}" \
    VLLM_XLA_CACHE_TAG="${VLLM_XLA_CACHE_TAG:-trc256_tp${VLLM_TP_SIZE}_pda1_mlen${VLLM_MAX_MODEL_LEN}_mbt${VLLM_MAX_NUM_BATCHED_TOKENS}}" \
    TMX_VLLM_SUPERVISE="${TMX_VLLM_SUPERVISE:-1}" \
    TMX_VLLM_SKIP_FAILED_WORKERS="${TMX_VLLM_SKIP_FAILED_WORKERS:-1}" \
    TMX_VLLM_BATCHED_SLICE_BOOTSTRAP="${TMX_VLLM_BATCHED_SLICE_BOOTSTRAP:-1}" \
    TMX_VLLM_BATCH_SIZE="${TMX_VLLM_BATCH_SIZE:-4}" \
    bash tmx_jax/scripts/bootstrap_vllm_slice.sh
  )
  echo "[dapo-launch] serve vLLM bootstrap command completed"
fi

sleep "${TMX_VLLM_POST_BOOTSTRAP_WARMUP_SECONDS:-5}"
echo "[dapo-launch] checking serve vLLM health min=$MIN_HEALTHY_SERVE_WORKERS workers=[$SERVE_WORKERS]"
health_deadline=$(( $(date +%s) + ${TMX_VLLM_HEALTH_WAIT_SECONDS:-900} ))
selected_workers=""
selected_count=0
while true; do
  selected_workers="$(healthy_workers "$SERVE_TPU_PROJECT" "$SERVE_ZONE" "$SERVE_TPU_NAME" "$SERVE_WORKERS")"
  selected_count="$(printf '%s\n' "$selected_workers" | wc -w | tr -d ' ')"
  if [ "$selected_count" -ge "$MIN_HEALTHY_SERVE_WORKERS" ]; then
    break
  fi
  if [ "$(date +%s)" -ge "$health_deadline" ]; then
    echo "[dapo-launch] ERROR: healthy vLLM workers [$selected_workers] count=$selected_count, need $MIN_HEALTHY_SERVE_WORKERS" >&2
    exit 3
  fi
  echo "[dapo-launch] waiting for serve vLLM health count=$selected_count need=$MIN_HEALTHY_SERVE_WORKERS workers=[$selected_workers]"
  sleep "${TMX_VLLM_HEALTH_POLL_SECONDS:-15}"
done
echo "[dapo-launch] serve vLLM healthy workers=[$selected_workers] count=$selected_count"

serve_ips_raw="$(worker_ips "$SERVE_TPU_PROJECT" "$SERVE_ZONE" "$SERVE_TPU_NAME")"
IFS=';' read -r -a serve_ips <<< "$serve_ips_raw"
urls=()
for worker in $selected_workers; do
  urls+=("http://${serve_ips[$worker]}:${PORT}/v1")
done
TMX_VLLM_SERVER_URLS="$(join_by_comma "${urls[@]}")"

if bool_enabled "${CLEAN_TRAIN_SLICE:-1}"; then
  clean_workers "$TRAIN_TPU_PROJECT" "$TRAIN_ZONE" "$TRAIN_TPU_NAME" "$TRAIN_WORKERS" "train"
fi

echo "[dapo-launch] run=$RUN_NAME"
echo "[dapo-launch] train=$TRAIN_TPU_PROJECT/$TRAIN_ZONE/$TRAIN_TPU_NAME workers=[$TRAIN_WORKERS]"
echo "[dapo-launch] serve=$SERVE_TPU_PROJECT/$SERVE_ZONE/$SERVE_TPU_NAME workers=[$selected_workers]"
echo "[dapo-launch] contract=lambda=$LAMBDA_NOVELTY G=$GRPO_NUM_GENERATIONS max_completion=$MAX_COMPLETION_LEN Dr.GRPO clip=0.2/0.28 KL=0"

cd "$REPO_ROOT"
env \
  CLOUDSDK_CORE_PROJECT="$TRAIN_TPU_PROJECT" \
  TPU_NAME="$TRAIN_TPU_NAME" \
  ZONE="$TRAIN_ZONE" \
  TRAIN_WORKERS="$TRAIN_WORKERS" \
  SERVE_WORKERS="" \
  RUN_NAME="$RUN_NAME" \
  RUNS_ROOT="$RUNS_ROOT" \
  REMOTE_REPO="$REMOTE_REPO" \
  MODEL_NAME="$MODEL_NAME" \
  DATASET_NAME="$DATASET_NAME" \
  INIT_CHECKPOINT="$INIT_CHECKPOINT" \
  TPU_ACCELERATOR_TYPE="$TPU_ACCELERATOR_TYPE" \
  TMX_USE_MANUAL_TPU_TOPOLOGY="$TMX_USE_MANUAL_TPU_TOPOLOGY" \
  TMX_TPU_HOST_BOUNDS="$TMX_TPU_HOST_BOUNDS" \
  JAX_INITIALIZATION_TIMEOUT="${JAX_INITIALIZATION_TIMEOUT:-1200}" \
  JAX_HEARTBEAT_TIMEOUT_SECONDS="${JAX_HEARTBEAT_TIMEOUT_SECONDS:-900}" \
  TMX_PARALLEL_TRAIN_LAUNCH="${TMX_PARALLEL_TRAIN_LAUNCH:-0}" \
  SYNC_LOCAL_PACKAGE="${SYNC_LOCAL_PACKAGE:-1}" \
  MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
  MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
  MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
  NOVELTY_FEATURE_MAX_LENGTH="${NOVELTY_FEATURE_MAX_LENGTH:-$MAX_TOTAL_LEN}" \
  LOGPROB_CHUNK_TOKENS="${LOGPROB_CHUNK_TOKENS:-256}" \
  GRPO_NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
  GRPO_BATCH_SIZE="$GRPO_BATCH_SIZE" \
  GRPO_GRAD_ACCUM="$GRPO_GRAD_ACCUM" \
  GRPO_MAX_STEPS="$GRPO_MAX_STEPS" \
  GRPO_UPDATES_PER_ROLLOUT="${GRPO_UPDATES_PER_ROLLOUT:-1}" \
  GRPO_LR="${GRPO_LR:-5e-6}" \
  GRPO_CLIP_EPSILON=0.2 \
  GRPO_CLIP_EPSILON_HIGH=0.28 \
  GRPO_KL_BETA=0.0 \
  GRPO_ADVANTAGE_NORMALIZATION=none \
  DYNAMIC_SAMPLING=true \
  DYNAMIC_SAMPLING_MAX_ATTEMPTS="${DYNAMIC_SAMPLING_MAX_ATTEMPTS:-8}" \
  DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT="${DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT:-1}" \
  MASK_TRUNCATED_COMPLETIONS=false \
  SOFT_OVERLONG_EXPECTED_LEN="$SOFT_OVERLONG_EXPECTED_LEN" \
  SOFT_OVERLONG_CACHE_LEN="$SOFT_OVERLONG_CACHE_LEN" \
  REWARD_CORRECT=1.0 \
  REWARD_INCORRECT=-1.0 \
  FORMAT_PENALTY_WEIGHT=0.0 \
  NONTERMINATION_PENALTY=0.0 \
  INVALID_ANSWER_PENALTY=0.0 \
  LAMBDA_NOVELTY="$LAMBDA_NOVELTY" \
  INCORRECT_NOVELTY_SCALE="${INCORRECT_NOVELTY_SCALE:-1.0}" \
  NOVELTY_FEATURE_SOURCE=multilayer \
  NOVELTY_LAYERS="${NOVELTY_LAYERS:-7,14,21}" \
  NOVELTY_LAYER_POOL=mean \
  NOVELTY_LAYER_AGG=mean \
  NOVELTY_METRIC=sqrt_mse \
  NOVELTY_TEXT_CONTRACT=full_raw \
  GATE_NOVELTY_BY_QUALITY=false \
  TMX_DAPO_DRGRPO_FULL_CONTRACT=1 \
  TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
  TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN="$SOFT_OVERLONG_EXPECTED_LEN" \
  TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN="$SOFT_OVERLONG_CACHE_LEN" \
  TMX_ALLOW_LONG_COMPLETION_TRAIN=1 \
  VLLM_CONCURRENCY="$VLLM_CONCURRENCY" \
  VLLM_FANOUT_PER_PROMPT="$VLLM_FANOUT_PER_PROMPT" \
  TMX_VLLM_RANDOMIZE_SEEDS="$TMX_VLLM_RANDOMIZE_SEEDS" \
  TMX_VLLM_SEED_MODE="$TMX_VLLM_SEED_MODE" \
  TMX_VLLM_SERVER_URLS="$TMX_VLLM_SERVER_URLS" \
  VLLM_SERVER_URLS="$TMX_VLLM_SERVER_URLS" \
  VLLM_RELOAD_TPU_NAME="$SERVE_TPU_NAME" \
  VLLM_RELOAD_TPU_PROJECT="$SERVE_TPU_PROJECT" \
  VLLM_RELOAD_ZONE="$SERVE_ZONE" \
  VLLM_RELOAD_WORKERS="$selected_workers" \
  TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS="${TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS:-10}" \
  TMX_TRAIN_VLLM_RELOAD_FIRST_STEP="${TMX_TRAIN_VLLM_RELOAD_FIRST_STEP:-110}" \
  TMX_TRAIN_VLLM_RELOAD_ON_RESUME=1 \
  TMX_TRAIN_VLLM_RELOAD_MODE="${TMX_TRAIN_VLLM_RELOAD_MODE:-external}" \
  TMX_VLLM_RELOAD_TRANSPORT="${TMX_VLLM_RELOAD_TRANSPORT:-hf}" \
  TMX_HF_MIRROR_VLLM_EXPORTS="${TMX_HF_MIRROR_VLLM_EXPORTS:-1}" \
  TMX_EXTERNAL_VLLM_RELOAD_REQUIRED="${TMX_EXTERNAL_VLLM_RELOAD_REQUIRED:-1}" \
  TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL="${TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL:-1}" \
  TMX_ARTIFACT_BACKEND=hf \
  TMX_ALLOW_GCS_ARTIFACTS=0 \
  TMX_HF_CHECKPOINT_REPO="${TMX_HF_CHECKPOINT_REPO:-SaifPunjwani/two-model-exploration-checkpoints}" \
  TMX_HF_ARTIFACT_PREFIX="${TMX_HF_ARTIFACT_PREFIX:-$RUN_NAME}" \
  TMX_HF_CHECKPOINT_EVERY_STEPS="${TMX_HF_CHECKPOINT_EVERY_STEPS:-10}" \
  TMX_HF_CHECKPOINT_FIRST_STEP="${TMX_HF_CHECKPOINT_FIRST_STEP:-110}" \
  SAVE_EVERY_STEPS="$SAVE_EVERY_STEPS" \
  TMX_SAVE_OPT_STATE=0 \
  TMX_RESTORE_OPT_STATE=0 \
  TMX_WANDB_SINGLE_RUN=1 \
  TMX_WANDB_COMPACT=1 \
  TMX_WANDB_MINIMAL=1 \
  TMX_AIME_PROBE_EVERY_STEPS="${TMX_AIME_PROBE_EVERY_STEPS:-10}" \
  TMX_AIME_PROBE_NUM_PROBLEMS="${TMX_AIME_PROBE_NUM_PROBLEMS:-4}" \
  TMX_AIME_PROBE_NUM_ROLLOUTS="${TMX_AIME_PROBE_NUM_ROLLOUTS:-4}" \
  TMX_AIME_PROBE_MAX_TOKENS="${TMX_AIME_PROBE_MAX_TOKENS:-16384}" \
  TMX_AIME_PROBE_CONCURRENCY="${TMX_AIME_PROBE_CONCURRENCY:-32}" \
  bash tmx_jax/scripts/launch_jax_pipeline_multihost_v5lite.sh
