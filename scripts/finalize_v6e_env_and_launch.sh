#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_TAR="${LOCAL_TAR:-/tmp/tmx_venv_tpu_slice.tar}"
LOCAL_HF_TAR="${LOCAL_HF_TAR:-}"
TARGET_TPU_NAME="${TARGET_TPU_NAME:?TARGET_TPU_NAME is required}"
TARGET_ZONE="${TARGET_ZONE:?TARGET_ZONE is required}"
TARGET_WORKER="${TARGET_WORKER:-0}"
TARGET_ACCELERATOR_TYPE="${TARGET_ACCELERATOR_TYPE:-v6e-64}"
TARGET_REMOTE_ROOT="${TARGET_REMOTE_ROOT:?set TARGET_REMOTE_ROOT to the absolute repo checkout path on the target TPU VM}"
TARGET_REMOTE_TAR="${TARGET_REMOTE_TAR:-/tmp/tmx_venv_tpu_slice.tar}"
TARGET_REMOTE_HF_TAR="${TARGET_REMOTE_HF_TAR:-/tmp/tmx_hf_cache.tar}"
QUEUE_TAG="${QUEUE_TAG:-v6e_wsanswer_qwen25_gsm8k_$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-15}"
STABLE_POLLS="${STABLE_POLLS:-2}"
RUN_PROFILE="${RUN_PROFILE:-stable_trace}"
RECIPE_MODE="${RECIPE_MODE:-stable_trace_warmstart}"
TRAIN_SELECTION_POLICY="${TRAIN_SELECTION_POLICY:-random}"
RUN_LIMIT="${RUN_LIMIT:-1}"
RETRY_ON_FAILURE="${RETRY_ON_FAILURE:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-512}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-256}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-128}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-96}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-128}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-448}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-48}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-2}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-2}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-2}"
MAX_DISTILL_EXAMPLES="${MAX_DISTILL_EXAMPLES:-512}"
MAX_ACTUAL_EXAMPLES="${MAX_ACTUAL_EXAMPLES:-512}"

ssh_target() {
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm ssh "$TARGET_TPU_NAME" \
    --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone "$TARGET_ZONE" \
    --worker "$TARGET_WORKER" \
    --tunnel-through-iap \
    --command "$1"
}

scp_to_target() {
  local local_path="$1"
  local remote_path="$2"
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm scp \
    --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone "$TARGET_ZONE" \
    --worker "$TARGET_WORKER" \
    --tunnel-through-iap \
    "$local_path" \
    "${TARGET_TPU_NAME}:${remote_path}"
}

remote_file_size() {
  local remote_path="$1"
  ssh_target "python3 - <<'PY'
import os
p = '$remote_path'
print(os.path.getsize(p) if os.path.exists(p) else 0)
PY" 2>/dev/null | tail -n 1 | tr -d '\r'
}

wait_for_remote_file() {
  local local_path="$1"
  local remote_path="$2"
  local label="$3"
  local local_size
  local last_remote=-1
  local stable_count=0

  local_size="$(stat -f%z "$local_path")"
  echo "[finalize] ${label} local size=$local_size"

  while true; do
    local remote_size
    remote_size="$(remote_file_size "$remote_path" || echo 0)"
    echo "[finalize] ${label} remote size=$remote_size"
    if [ "$remote_size" = "$local_size" ] && [ "$remote_size" = "$last_remote" ]; then
      stable_count=$((stable_count + 1))
      if [ "$stable_count" -ge "$STABLE_POLLS" ]; then
        break
      fi
    else
      stable_count=0
    fi
    last_remote="$remote_size"
    sleep "$POLL_SECONDS"
  done
}

scp_to_target "$LOCAL_TAR" "$TARGET_REMOTE_TAR"
wait_for_remote_file "$LOCAL_TAR" "$TARGET_REMOTE_TAR" "venv"

ssh_target "mkdir -p '$TARGET_REMOTE_ROOT' && cd '$TARGET_REMOTE_ROOT' && rm -rf .venv_tpu_slice && tar -xf '$TARGET_REMOTE_TAR' -C '$TARGET_REMOTE_ROOT'"

if [ -n "$LOCAL_HF_TAR" ] && [ -f "$LOCAL_HF_TAR" ]; then
  scp_to_target "$LOCAL_HF_TAR" "$TARGET_REMOTE_HF_TAR"
  wait_for_remote_file "$LOCAL_HF_TAR" "$TARGET_REMOTE_HF_TAR" "hf"
  ssh_target "mkdir -p '$TARGET_REMOTE_ROOT' && cd '$TARGET_REMOTE_ROOT' && rm -rf .hf && tar -xf '$TARGET_REMOTE_HF_TAR' -C '$TARGET_REMOTE_ROOT'"
fi

TPU_NAME="$TARGET_TPU_NAME" \
ZONE="$TARGET_ZONE" \
SYNC_WORKERS="$TARGET_WORKER" \
ALLOW_DIRTY=1 \
bash scripts/sync_tpu_slice_repo.sh

TPU_NAME="$TARGET_TPU_NAME" \
ZONE="$TARGET_ZONE" \
BOOTSTRAP_WORKERS="$TARGET_WORKER" \
SKIP_SYNC=1 \
SKIP_INSTALL=1 \
bash scripts/bootstrap_tpu_slice.sh

ssh_target "cd '$TARGET_REMOTE_ROOT' && . .venv_tpu_slice/bin/activate && python - <<'PY'
import importlib.util as u
mods = ['torch', 'torch_xla', 'transformers', 'datasets', 'wandb']
print({m: (u.find_spec(m) is not None) for m in mods})
PY"

ssh_target "export TPU_ACCELERATOR_TYPE='$TARGET_ACCELERATOR_TYPE' PJRT_DEVICE=TPU TMX_TPU_SLICE_LAUNCHED=1 TMX_REQUIRE_TPU=1 TPU_SKIP_MDS_QUERY=1 TPU_VISIBLE_CHIPS=0 TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost; cd '$TARGET_REMOTE_ROOT' && . .venv_tpu_slice/bin/activate && python - <<'PY'
from tmx.devices import resolve_device
d, is_xla, label = resolve_device('tpu')
print(d, is_xla, label)
PY"

TPU_NAME="$TARGET_TPU_NAME" \
ZONE="$TARGET_ZONE" \
WORKERS="$TARGET_WORKER" \
REMOTE_WORKERS="$TARGET_WORKER" \
LOCAL_CHIPS=1 \
CHIPS=0 \
SKIP_BOOTSTRAP=1 \
TPU_ACCELERATOR_TYPE="$TARGET_ACCELERATOR_TYPE" \
TMX_GRPO_GRAD_CKPT="${TMX_GRPO_GRAD_CKPT:-0}" \
TMX_XLA_SAFE_GENERATE=1 \
HF_HOME="$TARGET_REMOTE_ROOT/.hf" \
HF_HUB_CACHE="$TARGET_REMOTE_ROOT/.hf/hub" \
HF_DATASETS_CACHE="$TARGET_REMOTE_ROOT/.hf/datasets" \
TRANSFORMERS_CACHE="$TARGET_REMOTE_ROOT/.hf/transformers" \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
RUN_PROFILE="$RUN_PROFILE" \
RECIPE_MODE="$RECIPE_MODE" \
TRAIN_SELECTION_POLICY="$TRAIN_SELECTION_POLICY" \
RUN_LIMIT="$RUN_LIMIT" \
RETRY_ON_FAILURE="$RETRY_ON_FAILURE" \
QUEUE_TAG="$QUEUE_TAG" \
MODEL_NAME="$MODEL_NAME" \
DATASET_NAME="$DATASET_NAME" \
MAX_TRAIN_EXAMPLES="$MAX_TRAIN_EXAMPLES" \
MAX_EVAL_EXAMPLES="$MAX_EVAL_EXAMPLES" \
MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
EXPLORATION_MAX_COMPLETION_LEN="$EXPLORATION_MAX_COMPLETION_LEN" \
EVAL_MAX_COMPLETION_LEN="$EVAL_MAX_COMPLETION_LEN" \
MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
GRPO_MAX_STEPS="$GRPO_MAX_STEPS" \
GRPO_NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
GRPO_GENERATION_BATCH_SIZE="$GRPO_GENERATION_BATCH_SIZE" \
NUM_SAMPLES_PER_PROMPT="$NUM_SAMPLES_PER_PROMPT" \
DISTILL_EPOCHS="$DISTILL_EPOCHS" \
MAX_DISTILL_EXAMPLES="$MAX_DISTILL_EXAMPLES" \
MAX_ACTUAL_EXAMPLES="$MAX_ACTUAL_EXAMPLES" \
bash scripts/queue_gsm8k_recipe_search_tmux.sh
