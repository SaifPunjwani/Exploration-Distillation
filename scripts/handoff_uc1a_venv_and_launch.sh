#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOCAL_TAR="${LOCAL_TAR:-/tmp/tmx_venv_tpu_slice.tar}"
MIN_READY_BYTES="${MIN_READY_BYTES:-2000000000}"
STABLE_POLLS="${STABLE_POLLS:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"

TARGET_TPU_NAME="${TARGET_TPU_NAME:?set TARGET_TPU_NAME to the target TPU VM (slice) name}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-a}"
TARGET_WORKER="${TARGET_WORKER:-0}"
TARGET_REMOTE_ROOT="${TARGET_REMOTE_ROOT:?set TARGET_REMOTE_ROOT to the absolute repo checkout path on the target TPU VM}"
TARGET_REMOTE_TAR="${TARGET_REMOTE_TAR:-/tmp/tmx_venv_tpu_slice.tar}"
TARGET_ACCELERATOR_TYPE="${TARGET_ACCELERATOR_TYPE:-v5litepod-64}"

QUEUE_TAG="${QUEUE_TAG:-uc1a_wsanswer_qwen25_gsm8k_20260326a}"

wait_for_stable_tar() {
  local stable_count=0
  local last_size=-1
  while true; do
    local size=0
    if [ -f "$LOCAL_TAR" ]; then
      size="$(stat -f%z "$LOCAL_TAR" 2>/dev/null || echo 0)"
    fi
    printf '[%s] local tar size=%s bytes\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$size"
    if [ "$size" -ge "$MIN_READY_BYTES" ] && [ "$size" -eq "$last_size" ]; then
      stable_count=$((stable_count + 1))
      if [ "$stable_count" -ge "$STABLE_POLLS" ]; then
        break
      fi
    else
      stable_count=0
    fi
    last_size="$size"
    sleep "$POLL_SECONDS"
  done
}

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
  CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
  gcloud alpha compute tpus tpu-vm scp \
    --project "${TPU_PROJECT:?set TPU_PROJECT to the GCP project id of the TPU}" \
    --zone "$TARGET_ZONE" \
    --worker "$TARGET_WORKER" \
    --tunnel-through-iap \
    "$LOCAL_TAR" \
    "${TARGET_TPU_NAME}:${TARGET_REMOTE_TAR}"
}

wait_for_stable_tar

ssh_target "pkill -f 'bootstrap_tpu_env.sh|python -m pip install --upgrade pip|pip download --progress-bar' || true"
scp_to_target

ssh_target "cd '$TARGET_REMOTE_ROOT' && rm -rf .venv_tpu_slice && tar -xf '$TARGET_REMOTE_TAR' -C '$TARGET_REMOTE_ROOT'"

ssh_target "cd '$TARGET_REMOTE_ROOT' && . .venv_tpu_slice/bin/activate && python - <<'PY'
import importlib.util as u
mods=['torch','torch_xla','transformers','datasets','wandb']
print({m:(u.find_spec(m) is not None) for m in mods})
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
RUN_PROFILE=smoke \
RECIPE_MODE=stable_answer_warmstart \
TRAIN_SELECTION_POLICY=random \
RUN_LIMIT=1 \
RETRY_ON_FAILURE=0 \
QUEUE_TAG="$QUEUE_TAG" \
MODEL_NAME='Qwen/Qwen2.5-Math-1.5B-Instruct' \
DATASET_NAME='gsm8k' \
MAX_TRAIN_EXAMPLES=256 \
MAX_EVAL_EXAMPLES=128 \
MAX_PROMPT_LEN=224 \
MAX_COMPLETION_LEN=96 \
EXPLORATION_MAX_COMPLETION_LEN=48 \
EVAL_MAX_COMPLETION_LEN=64 \
MAX_TOTAL_LEN=320 \
GRPO_MAX_STEPS=16 \
GRPO_NUM_GENERATIONS=2 \
GRPO_GENERATION_BATCH_SIZE=2 \
NUM_SAMPLES_PER_PROMPT=2 \
DISTILL_EPOCHS=2 \
MAX_DISTILL_EXAMPLES=256 \
MAX_ACTUAL_EXAMPLES=256 \
bash scripts/queue_gsm8k_recipe_search_tmux.sh
