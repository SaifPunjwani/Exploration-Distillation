#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TPU_NAME="${TPU_NAME:?set TPU_NAME to the TPU VM (slice) name}"
ZONE="${ZONE:-europe-west4-b}"
PROJECT="${PROJECT:?set PROJECT to the GCP project id of the TPU}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-64}"
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"
TRANSPORT_FLAG="${TRANSPORT_FLAG:---tunnel-through-iap}"
POLL_SECONDS="${POLL_SECONDS:-20}"
SSH_DEAD_POLLS="${SSH_DEAD_POLLS:-10}"
NODE_MISSING_POLLS="${NODE_MISSING_POLLS:-6}"
UNHEALTHY_POLLS="${UNHEALTHY_POLLS:-4}"
TARGET_WORKER="${TARGET_WORKER:-0}"
TARGET_WORKERS="${TARGET_WORKERS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"

QUEUE_TAG="${QUEUE_TAG:-ew4b_canary_$(date +%Y%m%d_%H%M%S)}"
SESSION_PREFIX="${SESSION_PREFIX:-ew4bcanary}"
RUN_PROFILE="${RUN_PROFILE:-stable_trace}"
RECIPE_MODE="${RECIPE_MODE:-stable_trace_warmstart}"
RUN_OFFSET="${RUN_OFFSET:-0}"
RUN_LIMIT="${RUN_LIMIT:-1}"
SEED="${SEED:-0}"

LOCAL_WHEELHOUSE_TAR="${LOCAL_WHEELHOUSE_TAR:-runs/offline_tpu_assets_20260331_wheelhouse_v3/wheelhouse.tar}"
LOCAL_HF_TAR="${LOCAL_HF_TAR:-runs/offline_tpu_assets_20260331_train/hf_cache.tar}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to the absolute repo checkout path on the TPU VM}"
TRAIN_SELECTION_MANIFEST_PATH="${TRAIN_SELECTION_MANIFEST_PATH:-runs/ew4b_overnight_20260331/train_selection_gsm8k.json}"

MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-256}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-128}"
TRAIN_PROBE_POOL_SIZE="${TRAIN_PROBE_POOL_SIZE:-256}"
TRAIN_PROBE_SAMPLES_PER_PROMPT="${TRAIN_PROBE_SAMPLES_PER_PROMPT:-2}"
TRAIN_PROBE_BATCH_SIZE="${TRAIN_PROBE_BATCH_SIZE:-16}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-32}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_GENERATION_BATCH_SIZE="${GRPO_GENERATION_BATCH_SIZE:-2}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-2}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-256}"
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-160}"
EXPLORATION_MAX_COMPLETION_LEN="${EXPLORATION_MAX_COMPLETION_LEN:-64}"
EVAL_MAX_COMPLETION_LEN="${EVAL_MAX_COMPLETION_LEN:-96}"
MAX_TOTAL_LEN="${MAX_TOTAL_LEN:-352}"

HF_HOME="${HF_HOME:-$REMOTE_ROOT/.hf}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$REMOTE_ROOT/.hf/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REMOTE_ROOT/.hf/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$REMOTE_ROOT/.hf/transformers}"

qr_state() {
  gcloud alpha compute tpus queued-resources describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(state.state)' 2>/dev/null || true
}

node_state() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(state)' 2>/dev/null || true
}

node_health() {
  gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(health)' 2>/dev/null || true
}

ssh_ready() {
  local worker="$1"
  python3 - "$TPU_NAME" "$PROJECT" "$ZONE" "$worker" "$TRANSPORT_FLAG" <<'PY'
import subprocess
import sys

tpu_name, project, zone, worker, transport_flag = sys.argv[1:]
cmd = [
    "gcloud",
    "alpha",
    "compute",
    "tpus",
    "tpu-vm",
    "ssh",
    tpu_name,
    f"--project={project}",
    f"--zone={zone}",
    f"--worker={worker}",
    transport_flag,
    "--ssh-flag=-o ConnectTimeout=5",
    "--ssh-flag=-o ConnectionAttempts=1",
    "--command=echo ok",
]
try:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=12,
        check=False,
    )
except subprocess.TimeoutExpired:
    raise SystemExit(1)
raise SystemExit(0 if proc.returncode == 0 else 1)
PY
}

find_reachable_worker() {
  python3 - "$TPU_NAME" "$PROJECT" "$ZONE" "$TARGET_WORKERS" "$TRANSPORT_FLAG" <<'PY'
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

tpu_name, project, zone, workers_raw, transport_flag = sys.argv[1:]
target_workers = [int(w) for w in workers_raw.split() if w.strip()]

def probe(worker: int):
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        f"--project={project}",
        f"--zone={zone}",
        f"--worker={worker}",
        transport_flag,
        "--ssh-flag=-o ConnectTimeout=5",
        "--ssh-flag=-o ConnectionAttempts=1",
        "--command=echo ok",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=12,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    return worker if proc.returncode == 0 else None

with ThreadPoolExecutor(max_workers=len(target_workers) or 1) as ex:
    futures = [ex.submit(probe, worker) for worker in target_workers]
    for fut in as_completed(futures):
        worker = fut.result()
        if worker is not None:
            print(worker)
            raise SystemExit(0)

raise SystemExit(1)
PY
}

delete_qr() {
  gcloud alpha compute tpus queued-resources delete "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --async \
    --quiet >/dev/null 2>&1 || true
}

delete_node() {
  gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --async \
    --quiet >/dev/null 2>&1 || true
}

launch_canary() {
  local worker="$1"
  mkdir -p "$(dirname "$TRAIN_SELECTION_MANIFEST_PATH")"
  TPU_NAME="$TPU_NAME" \
  ZONE="$ZONE" \
  WORKERS="$worker" \
  CHIPS=0 \
  LOCAL_CHIPS=1 \
  BOOTSTRAP_WORKERS="$worker" \
  SYNC_WORKERS="$worker" \
  REMOTE_ROOT="$REMOTE_ROOT" \
  LOCAL_WHEELHOUSE_TAR="$LOCAL_WHEELHOUSE_TAR" \
  LOCAL_HF_TAR="$LOCAL_HF_TAR" \
  HF_HOME="$HF_HOME" \
  HF_HUB_CACHE="$HF_HUB_CACHE" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  ALLOW_DIRTY=1 \
  RUN_PROFILE="$RUN_PROFILE" \
  RECIPE_MODE="$RECIPE_MODE" \
  RUN_OFFSET="$RUN_OFFSET" \
  RUN_LIMIT="$RUN_LIMIT" \
  SEED="$SEED" \
  TRAIN_SELECTION_MANIFEST_PATH="$TRAIN_SELECTION_MANIFEST_PATH" \
  MAX_TRAIN_EXAMPLES="$MAX_TRAIN_EXAMPLES" \
  MAX_EVAL_EXAMPLES="$MAX_EVAL_EXAMPLES" \
  TRAIN_PROBE_POOL_SIZE="$TRAIN_PROBE_POOL_SIZE" \
  TRAIN_PROBE_SAMPLES_PER_PROMPT="$TRAIN_PROBE_SAMPLES_PER_PROMPT" \
  TRAIN_PROBE_BATCH_SIZE="$TRAIN_PROBE_BATCH_SIZE" \
  GRPO_MAX_STEPS="$GRPO_MAX_STEPS" \
  GRPO_NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
  GRPO_GENERATION_BATCH_SIZE="$GRPO_GENERATION_BATCH_SIZE" \
  NUM_SAMPLES_PER_PROMPT="$NUM_SAMPLES_PER_PROMPT" \
  MAX_PROMPT_LEN="$MAX_PROMPT_LEN" \
  MAX_COMPLETION_LEN="$MAX_COMPLETION_LEN" \
  EXPLORATION_MAX_COMPLETION_LEN="$EXPLORATION_MAX_COMPLETION_LEN" \
  EVAL_MAX_COMPLETION_LEN="$EVAL_MAX_COMPLETION_LEN" \
  MAX_TOTAL_LEN="$MAX_TOTAL_LEN" \
  TMX_GRPO_GRAD_CKPT="${TMX_GRPO_GRAD_CKPT:-0}" \
  TMX_XLA_SAFE_GENERATE=1 \
  QUEUE_TAG="$QUEUE_TAG" \
  SESSION_PREFIX="$SESSION_PREFIX" \
  bash scripts/queue_gsm8k_recipe_search_tmux.sh
}

ssh_dead_count=0
node_missing_count=0
unhealthy_count=0

while true; do
  current_qr_state="$(qr_state)"
  current_node_state="$(node_state)"
  current_node_health="$(node_health)"

  printf '[%s] qr_state=%s node_state=%s node_health=%s ssh_dead_count=%s node_missing_count=%s unhealthy_count=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${current_qr_state:-NONE}" \
    "${current_node_state:-NONE}" \
    "${current_node_health:-NONE}" \
    "$ssh_dead_count" \
    "$node_missing_count" \
    "$unhealthy_count"

  case "${current_qr_state:-}" in
    FAILED|SUSPENDED|SUSPENDING)
      delete_qr
      delete_node
      ssh_dead_count=0
      node_missing_count=0
      unhealthy_count=0
      sleep 10
      continue
      ;;
  esac

  if [ -z "${current_qr_state:-}" ]; then
    ACCELERATOR_TYPE="$ACCELERATOR_TYPE" \
    bash scripts/recreate_v6e_qr.sh "$TPU_NAME" "$ZONE" "$RUNTIME_VERSION" || true
    ssh_dead_count=0
    node_missing_count=0
    unhealthy_count=0
    sleep "$POLL_SECONDS"
    continue
  fi

  if [ -z "${current_node_state:-}" ]; then
    if [ "${current_qr_state:-}" = "ACTIVE" ]; then
      node_missing_count=$((node_missing_count + 1))
      if [ "$node_missing_count" -ge "$NODE_MISSING_POLLS" ]; then
        echo "[recover] ${TPU_NAME} stayed ACTIVE without a TPU VM; recycling queue"
        delete_qr
        delete_node
        ssh_dead_count=0
        node_missing_count=0
        unhealthy_count=0
        sleep 10
        continue
      fi
    else
      node_missing_count=0
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  node_missing_count=0
  if [ -z "${current_node_health:-}" ]; then
    unhealthy_count=0
    sleep "$POLL_SECONDS"
    continue
  fi

  if [ "${current_node_health:-}" != "HEALTHY" ]; then
    unhealthy_count=$((unhealthy_count + 1))
    if [ "$unhealthy_count" -ge "$UNHEALTHY_POLLS" ]; then
      echo "[recover] ${TPU_NAME} stayed non-healthy (${current_node_health}); recycling slice"
      delete_qr
      delete_node
      ssh_dead_count=0
      node_missing_count=0
      unhealthy_count=0
      sleep 10
      continue
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  unhealthy_count=0

  if [ "${current_node_state:-}" = "READY" ] && [ "${current_node_health:-}" = "HEALTHY" ]; then
    reachable_worker="$(find_reachable_worker || true)"
    if [ -n "${reachable_worker:-}" ]; then
      if ssh_ready "$reachable_worker"; then
        echo "[launch] SSH is ready on worker ${reachable_worker}; launching ${RUN_PROFILE}/${RECIPE_MODE}"
        if launch_canary "$reachable_worker"; then
          exit 0
        fi
      fi
      echo "[launch] queue command failed; recycling ${TPU_NAME}"
      delete_qr
      delete_node
      ssh_dead_count=0
      node_missing_count=0
      unhealthy_count=0
      sleep 10
      continue
    fi

    ssh_dead_count=$((ssh_dead_count + 1))
    if [ "$ssh_dead_count" -ge "$SSH_DEAD_POLLS" ]; then
      echo "[recover] SSH stayed unavailable on ${TPU_NAME}; recycling slice"
      delete_qr
      delete_node
      ssh_dead_count=0
      node_missing_count=0
      unhealthy_count=0
      sleep 10
      continue
    fi
  fi

  sleep "$POLL_SECONDS"
done
